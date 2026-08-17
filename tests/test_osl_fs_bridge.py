"""Tests for the FreeSurfer/MNE source-reconstruction backend.

The parcellation tests run against the atlases osl-ephys actually ships, since
the point of this module is to reproduce osl-ephys' RHINO results without FSL.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from custom.osl import fs_bridge


PARCELLATION = "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"


@pytest.fixture(scope="module")
def atlas():
    """The shipped Glasser 52-parcel atlas, as a nibabel image."""
    import nibabel as nib

    from osl_ephys.source_recon import parcellation

    return nib.load(parcellation.find_file(PARCELLATION))


# ---------------------------------------------------------------------------
# File layout and path resolution
# ---------------------------------------------------------------------------


class TestFilenames:
    def test_layout_is_under_the_subject_directory(self):
        files = fs_bridge.get_fs_filenames("/out", "sub-007_ses-01")
        assert files["basedir"] == "/out/sub-007_ses-01/fs_src"
        assert files["fwd_model"] == "/out/sub-007_ses-01/fs_src/model-fwd.fif"
        assert files["parcdir"] == "/out/sub-007_ses-01/parc"

    def test_parc_directory_matches_the_rhino_backend(self):
        # osl-ephys' own beamform_and_parcellate writes to {outdir}/{subject}/parc,
        # so both backends' output lands in the same place.
        files = fs_bridge.get_fs_filenames("/out", "sub-007")
        assert files["parcdir"].endswith("/sub-007/parc")


class TestResolveSubjectsDir:
    def test_uses_the_explicit_value(self):
        assert fs_bridge._resolve_subjects_dir("/fs") == "/fs"

    def test_falls_back_to_the_environment(self, monkeypatch):
        monkeypatch.setenv("SUBJECTS_DIR", "/env/fs")
        assert fs_bridge._resolve_subjects_dir(None) == "/env/fs"

    def test_raises_when_neither_is_set(self, monkeypatch):
        monkeypatch.delenv("SUBJECTS_DIR", raising=False)
        with pytest.raises(ValueError, match="subjects_dir"):
            fs_bridge._resolve_subjects_dir(None)


class TestResolveTrans:
    def test_uses_the_freesurfer_convention(self, tmp_path):
        bem = tmp_path / "sub-007" / "bem"
        bem.mkdir(parents=True)
        expected = bem / "sub-007-trans.fif"
        expected.touch()
        assert fs_bridge._resolve_trans(None, str(tmp_path), "sub-007") == str(
            expected
        )

    def test_raises_with_a_pointer_to_the_coreg_stage(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="coreg"):
            fs_bridge._resolve_trans(None, str(tmp_path), "sub-007")


# ---------------------------------------------------------------------------
# Orientation options
# ---------------------------------------------------------------------------


class TestResolvePickOri:
    @pytest.mark.parametrize("pick_ori", ["max-power", "normal", None])
    def test_passes_scalar_mne_options_through(self, pick_ori):
        assert fs_bridge._resolve_pick_ori(pick_ori) == pick_ori

    def test_rejects_the_osl_only_option(self):
        # Silently substituting 'max-power' would change the estimator without
        # saying so, which is a scientific difference, not a formatting one.
        with pytest.raises(ValueError, match="max-power-pre-weight-norm"):
            fs_bridge._resolve_pick_ori("max-power-pre-weight-norm")

    def test_the_error_names_the_alternatives(self):
        with pytest.raises(ValueError, match="source_backend: rhino"):
            fs_bridge._resolve_pick_ori("max-power-pre-weight-norm")

    def test_rejects_vector_orientations(self):
        # A three-component estimate cannot be collapsed to a parcel time
        # course; caught here rather than as a shape error inside the
        # parcellation.
        with pytest.raises(ValueError, match="three components"):
            fs_bridge._resolve_pick_ori("vector")


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------


class TestReadMriHeadT:
    def test_returns_mri_to_head_unchanged(self, tmp_path):
        import mne

        trans = mne.transforms.Transform("mri", "head", np.eye(4))
        path = tmp_path / "a-trans.fif"
        mne.write_trans(path, trans)

        result = fs_bridge._read_mri_head_t(str(path))
        assert result["from"] == mne.io.constants.FIFF.FIFFV_COORD_MRI
        assert result["to"] == mne.io.constants.FIFF.FIFFV_COORD_HEAD

    def test_inverts_a_head_to_mri_transform(self, tmp_path):
        import mne

        # mne.coreg.Coregistration produces head -> mri, but mne.head_to_mni
        # wants mri -> head, so the direction has to be normalised.
        matrix = np.eye(4)
        matrix[:3, 3] = [0.01, 0.02, 0.03]
        trans = mne.transforms.Transform("head", "mri", matrix)
        path = tmp_path / "b-trans.fif"
        mne.write_trans(path, trans)

        result = fs_bridge._read_mri_head_t(str(path))
        assert result["from"] == mne.io.constants.FIFF.FIFFV_COORD_MRI
        assert result["to"] == mne.io.constants.FIFF.FIFFV_COORD_HEAD
        np.testing.assert_allclose(result["trans"], np.linalg.inv(matrix), atol=1e-12)


class TestTransformToMni:
    def _fake_forward(self, rr):
        return {"src": [{"rr": rr, "vertno": np.arange(len(rr))}]}

    def test_maps_dipoles_onto_the_grid_by_nearest_neighbour(self, monkeypatch):
        # A 10 mm grid of four points, and dipoles sitting exactly on three of
        # them: those three should receive their dipole's time course, and the
        # fourth should stay at zero.
        grid = np.array(
            [[0.0, 10.0, 20.0, 30.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        dipoles_mni = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])

        monkeypatch.setattr(
            fs_bridge, "_mni_grid_from_reference", lambda *a, **k: grid
        )
        monkeypatch.setattr(
            fs_bridge.mne, "head_to_mni", lambda *a, **k: dipoles_mni
        )
        monkeypatch.setattr(fs_bridge, "_read_mri_head_t", lambda path: None)

        timeseries = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        fwd = self._fake_forward(np.zeros((3, 3)))

        out, coords = fs_bridge._transform_to_mni(
            fwd,
            timeseries,
            subject="sub-007",
            subjects_dir="/fs",
            trans_path="/fs/trans.fif",
            parcellation_file=PARCELLATION,
            reference_brain="parcellation",
            spatial_resolution=10,
        )

        assert out.shape == (4, 2)
        np.testing.assert_allclose(out[0], [1.0, 1.0])
        np.testing.assert_allclose(out[1], [2.0, 2.0])
        np.testing.assert_allclose(out[2], [3.0, 3.0])
        # No dipole within 10 mm of the fourth grid point.
        np.testing.assert_allclose(out[3], [0.0, 0.0])
        np.testing.assert_allclose(coords, grid)

    def test_preserves_the_trial_dimension(self, monkeypatch):
        grid = np.array([[0.0, 10.0], [0.0, 0.0], [0.0, 0.0]])
        dipoles_mni = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

        monkeypatch.setattr(
            fs_bridge, "_mni_grid_from_reference", lambda *a, **k: grid
        )
        monkeypatch.setattr(
            fs_bridge.mne, "head_to_mni", lambda *a, **k: dipoles_mni
        )
        monkeypatch.setattr(fs_bridge, "_read_mri_head_t", lambda path: None)

        timeseries = np.zeros((2, 5, 7))  # dipoles x times x trials
        out, _ = fs_bridge._transform_to_mni(
            self._fake_forward(np.zeros((2, 3))),
            timeseries,
            subject="sub-007",
            subjects_dir="/fs",
            trans_path="/fs/trans.fif",
            parcellation_file=PARCELLATION,
            reference_brain="parcellation",
            spatial_resolution=10,
        )
        assert out.shape == (2, 5, 7)


# ---------------------------------------------------------------------------
# Parcellation, without FSL
# ---------------------------------------------------------------------------


class TestDropInterpolationDust:
    def test_zeroes_negligible_weights(self):
        import nibabel as nib

        data = np.zeros((2, 2, 2, 1))
        data[0, 0, 0, 0] = 1.0
        data[1, 1, 1, 0] = 1e-30
        img = nib.Nifti1Image(data, np.eye(4))

        out = fs_bridge._drop_interpolation_dust(img).get_fdata()
        assert out[0, 0, 0, 0] == 1.0
        assert out[1, 1, 1, 0] == 0.0

    def test_keeps_real_probabilistic_weights(self):
        import nibabel as nib

        data = np.zeros((2, 2, 2, 1))
        data[0, 0, 0, 0] = 1.0
        data[1, 1, 1, 0] = 0.05
        img = nib.Nifti1Image(data, np.eye(4))

        out = fs_bridge._drop_interpolation_dust(img).get_fdata()
        assert out[1, 1, 1, 0] == pytest.approx(0.05)

    def test_thresholds_each_parcel_against_its_own_peak(self):
        import nibabel as nib

        data = np.zeros((2, 2, 2, 2))
        data[0, 0, 0, 0] = 1000.0  # parcel 0 is on a much larger scale
        data[1, 1, 1, 0] = 1e-9
        data[0, 0, 0, 1] = 1e-3  # parcel 1 is small but entirely real
        img = nib.Nifti1Image(data, np.eye(4))

        out = fs_bridge._drop_interpolation_dust(img).get_fdata()
        assert out[1, 1, 1, 0] == 0.0
        assert out[0, 0, 0, 1] == pytest.approx(1e-3)

    def test_handles_a_3d_image(self):
        import nibabel as nib

        data = np.zeros((2, 2, 2))
        data[0, 0, 0] = 1.0
        data[1, 1, 1] = 1e-30
        out = fs_bridge._drop_interpolation_dust(
            nib.Nifti1Image(data, np.eye(4))
        ).get_fdata()
        assert out.shape == (2, 2, 2)
        assert out[1, 1, 1] == 0.0


class TestResampleToIsotropic:
    def test_produces_isotropic_voxels(self, atlas):
        resampled = fs_bridge._resample_to_isotropic(atlas, 8)
        np.testing.assert_allclose(np.abs(np.diag(resampled.affine)[:3]), 8.0)

    def test_preserves_the_mni_sform_code(self, atlas):
        # nilearn stamps code 2 (ALIGNED_ANAT), which osl-ephys' own get_sform
        # rejects outright.
        resampled = fs_bridge._resample_to_isotropic(atlas, 8)
        assert int(resampled.header["sform_code"]) == int(atlas.header["sform_code"])
        assert int(resampled.header["sform_code"]) in (1, 4)


class TestNiiPointcloud:
    def test_returns_mm_coordinates_and_values(self):
        import nibabel as nib

        data = np.zeros((3, 3, 3))
        data[1, 1, 1] = 5.0
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        coords, values = fs_bridge._nii_pointcloud(nib.Nifti1Image(data, affine))

        assert coords.shape == (3, 1)
        np.testing.assert_allclose(coords[:, 0], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(values, [5.0])

    def test_requires_volindex_for_4d(self):
        import nibabel as nib

        img = nib.Nifti1Image(np.ones((2, 2, 2, 3)), np.eye(4))
        with pytest.raises(ValueError, match="volindex"):
            fs_bridge._nii_pointcloud(img)


class TestMniGridFromReference:
    def test_derives_a_regular_grid_from_the_parcellation(self):
        coords = fs_bridge._mni_grid_from_reference("parcellation", PARCELLATION, 8)

        assert coords.shape[0] == 3
        assert coords.shape[1] > 1000
        # The grid step osl-ephys later infers from these coordinates must be
        # the one we asked for, since it drives the parcellation resampling.
        assert int(fs_bridge.rhino_utils.get_gridstep(coords.T) / 1000) == 8

    def test_grid_lies_within_a_plausible_mni_bounding_box(self):
        coords = fs_bridge._mni_grid_from_reference("parcellation", PARCELLATION, 8)
        assert coords[0].min() > -100 and coords[0].max() < 100
        assert coords[1].min() > -140 and coords[1].max() < 100
        assert coords[2].min() > -100 and coords[2].max() < 110


class TestResampleParcellation:
    def test_round_trips_the_shipped_atlas(self, tmp_path, atlas):
        # The atlas is already at 8 mm, so resampling it onto its own grid must
        # give back exactly the same parcels. This is the check that the
        # nilearn substitution for `flirt` is faithful.
        coords = fs_bridge._mni_grid_from_reference("parcellation", PARCELLATION, 8)
        matrix = fs_bridge._resample_parcellation(
            PARCELLATION, coords, str(tmp_path)
        )

        source = atlas.get_fdata()
        expected = (source != 0).sum(axis=(0, 1, 2))
        actual = (matrix != 0).sum(axis=0)

        assert matrix.shape == (coords.shape[1], 52)
        np.testing.assert_array_equal(np.sort(actual), np.sort(expected))

    def test_parcels_do_not_overlap_for_a_binary_atlas(self, tmp_path):
        coords = fs_bridge._mni_grid_from_reference("parcellation", PARCELLATION, 8)
        matrix = fs_bridge._resample_parcellation(
            PARCELLATION, coords, str(tmp_path)
        )
        assert (matrix != 0).sum(axis=1).max() == 1

    def test_no_parcel_is_empty(self, tmp_path):
        coords = fs_bridge._mni_grid_from_reference("parcellation", PARCELLATION, 8)
        matrix = fs_bridge._resample_parcellation(
            PARCELLATION, coords, str(tmp_path)
        )
        assert (matrix != 0).sum(axis=0).min() > 0

    def test_writes_the_resampled_parcellation_for_provenance(self, tmp_path):
        coords = fs_bridge._mni_grid_from_reference("parcellation", PARCELLATION, 8)
        fs_bridge._resample_parcellation(PARCELLATION, coords, str(tmp_path))
        assert list(tmp_path.glob("*_8mm.nii.gz"))

    def test_output_feeds_osl_parcel_timeseries(self, tmp_path):
        # The whole point: the matrix must be usable by osl-ephys' own parcel
        # time-course maths, so both backends produce comparable output.
        from osl_ephys.source_recon import parcellation

        coords = fs_bridge._mni_grid_from_reference("parcellation", PARCELLATION, 8)
        matrix = fs_bridge._resample_parcellation(
            PARCELLATION, coords, str(tmp_path)
        )

        rng = np.random.RandomState(0)
        voxel_ts = rng.randn(coords.shape[1], 50)
        parcel_ts, weights, assignments = parcellation._get_parcel_timeseries(
            voxel_ts, matrix, method="spatial_basis"
        )

        assert parcel_ts.shape == (52, 50)
        assert np.isfinite(parcel_ts).all()
        assert weights.shape == (coords.shape[1], 52)
        assert assignments.shape == (coords.shape[1], 52)


# ---------------------------------------------------------------------------
# Orthogonalisation
# ---------------------------------------------------------------------------


class TestOrthogonalise:
    def test_none_is_a_passthrough(self):
        data = np.arange(12.0).reshape(3, 4)
        for value in (None, "none", "None"):
            np.testing.assert_array_equal(
                fs_bridge._orthogonalise(data, value, PARCELLATION, None), data
            )

    def test_symmetric_decorrelates_the_parcels(self):
        rng = np.random.RandomState(0)
        source = rng.randn(1, 500)
        # Three parcels that are all near-copies of one source: symmetric
        # orthogonalisation should knock their mutual correlation right down.
        data = np.repeat(source, 3, axis=0) + 0.1 * rng.randn(3, 500)

        result = fs_bridge._orthogonalise(data, "symmetric", PARCELLATION, None)

        before = np.abs(np.corrcoef(data)[np.triu_indices(3, 1)]).mean()
        after = np.abs(np.corrcoef(result)[np.triu_indices(3, 1)]).mean()
        assert result.shape == data.shape
        assert after < before

    def test_local_requires_a_neighbour_distance(self):
        with pytest.raises(ValueError, match="neighbour_distance"):
            fs_bridge._orthogonalise(
                np.zeros((3, 4)), "local", PARCELLATION, None
            )

    def test_rejects_an_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown orthogonalisation"):
            fs_bridge._orthogonalise(np.zeros((3, 4)), "sideways", PARCELLATION, None)


# ---------------------------------------------------------------------------
# Beamforming
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sphere_setup():
    """A synthetic OPM-like array, sphere BEM, volume source space and epochs.

    Enough to exercise the real MNE beamformer without a FreeSurfer subject.
    """
    import mne

    n_channels = 40
    rng = np.random.RandomState(0)

    info = mne.create_info(
        [f"MEG{i:03d}" for i in range(n_channels)], 200.0, ["mag"] * n_channels
    )
    for i, ch in enumerate(info["chs"]):
        theta, phi, radius = 2 * np.pi * i / n_channels, np.pi / 3, 0.11
        loc = np.zeros(12)
        loc[:3] = [
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ]
        loc[3:6] = loc[:3] / np.linalg.norm(loc[:3])
        tangent = np.cross(loc[3:6], [0, 0, 1.0])
        tangent /= np.linalg.norm(tangent)
        loc[6:9] = tangent
        loc[9:12] = np.cross(loc[3:6], tangent)
        ch["loc"] = loc
    info["dev_head_t"] = mne.transforms.Transform("meg", "head", np.eye(4))

    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.09, verbose=False)
    src = mne.setup_volume_source_space(
        pos=20.0, sphere=sphere, sphere_units="m", verbose=False
    )
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere, meg=True, eeg=False, verbose=False
    )

    raw = mne.io.RawArray(
        rng.randn(n_channels, 4000) * 1e-12, info, verbose=False
    )
    events = np.column_stack(
        [np.arange(200, 3800, 200), np.zeros(18, int), np.ones(18, int)]
    )
    epochs = mne.Epochs(
        raw,
        events,
        {"a": 1},
        tmin=-0.2,
        tmax=0.3,
        baseline=None,
        preload=True,
        verbose=False,
    )
    return SimpleNamespace(info=info, fwd=fwd, raw=raw, epochs=epochs)


class TestBeamforming:
    def _filters(self, setup, data, is_epochs, **kwargs):
        import mne

        options = dict(
            reg=0.05,
            noise_cov=None,
            pick_ori="max-power",
            weight_norm="unit-noise-gain-invariant",
            rank={"mag": 20},
            reduce_rank=True,
        )
        options.update(kwargs)
        cov = fs_bridge._compute_data_cov(data, is_epochs, "shrunk")
        return mne.beamformer.make_lcmv(data.info, setup.fwd, cov, **options)

    def test_epochs_stack_as_dipoles_times_trials(self, sphere_setup):
        filters = self._filters(sphere_setup, sphere_setup.epochs, True)
        out = fs_bridge._apply_lcmv(sphere_setup.epochs, filters, True)

        n_sources = sphere_setup.fwd["src"][0]["nuse"]
        assert out.shape == (n_sources, len(sphere_setup.epochs.times), 18)
        assert np.isfinite(out).all()

    def test_continuous_data_stacks_as_dipoles_times(self, sphere_setup):
        filters = self._filters(sphere_setup, sphere_setup.raw, False)
        out = fs_bridge._apply_lcmv(sphere_setup.raw, filters, False)

        n_sources = sphere_setup.fwd["src"][0]["nuse"]
        assert out.shape == (n_sources, len(sphere_setup.raw.times))
        assert np.isfinite(out).all()

    @pytest.mark.parametrize(
        "weight_norm", ["unit-noise-gain-invariant", "unit-noise-gain", "nai"]
    )
    def test_configured_weight_norms_work_without_a_noise_covariance(
        self, sphere_setup, weight_norm
    ):
        # 'nai' derives its noise level from the covariance's noise subspace,
        # so it needs either a reduced rank or regularisation -- both of which
        # the shipped config sets.
        filters = self._filters(
            sphere_setup, sphere_setup.epochs, True, weight_norm=weight_norm
        )
        out = fs_bridge._apply_lcmv(sphere_setup.epochs, filters, True)
        assert np.isfinite(out).all()

    def test_covariance_estimators_from_the_config(self, sphere_setup):
        for method in ("empirical", "shrunk"):
            cov = fs_bridge._compute_data_cov(sphere_setup.epochs, True, method)
            assert cov.data.shape == (40, 40)

    def test_output_feeds_the_mni_transform(self, sphere_setup, monkeypatch):
        # The beamformer output must slot straight into the morph step, which
        # indexes it by dipole along axis 0.
        filters = self._filters(sphere_setup, sphere_setup.epochs, True)
        out = fs_bridge._apply_lcmv(sphere_setup.epochs, filters, True)

        n_sources = sphere_setup.fwd["src"][0]["nuse"]
        grid = np.zeros((3, 5))
        grid[0] = np.arange(5) * 8.0
        monkeypatch.setattr(
            fs_bridge, "_mni_grid_from_reference", lambda *a, **k: grid
        )
        monkeypatch.setattr(
            fs_bridge.mne,
            "head_to_mni",
            lambda *a, **k: np.zeros((n_sources, 3)),
        )
        monkeypatch.setattr(fs_bridge, "_read_mri_head_t", lambda path: None)

        morphed, coords = fs_bridge._transform_to_mni(
            sphere_setup.fwd,
            out,
            subject="sub-007",
            subjects_dir="/fs",
            trans_path="/fs/trans.fif",
            parcellation_file=PARCELLATION,
            reference_brain="parcellation",
            spatial_resolution=8,
        )
        assert morphed.shape == (5, len(sphere_setup.epochs.times), 18)
        assert coords.shape == (3, 5)


class TestRegistry:
    def test_wrappers_are_registered_under_their_config_names(self):
        assert [f.__name__ for f in fs_bridge.SOURCE_EXTRA_FUNCS] == [
            "fs_coregister",
            "fs_forward_model",
            "fs_beamform_and_parcellate",
        ]
