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
# Beamformer rank
# ---------------------------------------------------------------------------


PARCELLATION_N = 52
"""Parcels in the Glasser52 atlas used throughout these tests."""


class TestResolveRank:
    """``rank: data`` mirrors run_beamformer.resolve_rank."""

    def test_an_explicit_rank_is_passed_through(self):
        assert fs_bridge._resolve_rank({"mag": 60}, None) == {"mag": 60}
        assert fs_bridge._resolve_rank("info", None) == "info"
        assert fs_bridge._resolve_rank(None, None) is None

    def test_data_takes_the_minimum_of_the_info_and_data_estimates(
        self, monkeypatch
    ):
        # The info rank knows the Maxwell basis but not what ICA removed, so
        # the data estimate has to be able to pull it down.
        calls = []

        def fake_compute_rank(data, rank=None, tol=None, tol_kind=None, verbose=None):
            calls.append(rank)
            return {"mag": 102} if rank == "info" else {"mag": 83}

        monkeypatch.setattr("mne.compute_rank", fake_compute_rank)
        assert fs_bridge._resolve_rank("data", object()) == {"mag": 83}

    def test_the_info_rank_caps_a_data_estimate_that_ran_away(self, monkeypatch):
        # tol='auto' style over-estimates must not raise the rank above what
        # the Maxwell basis can support.
        def fake_compute_rank(data, rank=None, tol=None, tol_kind=None, verbose=None):
            return {"mag": 102} if rank == "info" else {"mag": 185}

        monkeypatch.setattr("mne.compute_rank", fake_compute_rank)
        assert fs_bridge._resolve_rank("data", object()) == {"mag": 102}

    def test_the_data_estimate_stands_alone_without_proc_history(
        self, monkeypatch
    ):
        def fake_compute_rank(data, rank=None, tol=None, tol_kind=None, verbose=None):
            if rank == "info":
                raise ValueError("no proc_history")
            return {"mag": 185}

        monkeypatch.setattr("mne.compute_rank", fake_compute_rank)
        assert fs_bridge._resolve_rank("data", object()) == {"mag": 185}

    def test_the_data_estimate_uses_an_explicit_relative_tolerance(
        self, monkeypatch
    ):
        # tol='auto' is ~1e-13 relative and counts every direction SSS/ICA
        # nulled, so it must not be what gets used.
        seen = {}

        def fake_compute_rank(data, rank=None, tol=None, tol_kind=None, verbose=None):
            if rank != "info":
                seen["tol"], seen["tol_kind"] = tol, tol_kind
            return {"mag": 83}

        monkeypatch.setattr("mne.compute_rank", fake_compute_rank)
        fs_bridge._resolve_rank("data", object())
        assert seen == {"tol": 1e-6, "tol_kind": "relative"}


class TestOrthogonalisationRankCheck:
    """Symmetric orthogonalisation needs rank >= n_parcels; fail before beamforming."""

    def test_a_rank_below_the_parcel_count_is_rejected(self):
        with pytest.raises(ValueError, match="linearly independent"):
            fs_bridge._check_orthogonalisation_rank(
                {"mag": 40}, "symmetric", PARCELLATION
            )

    def test_a_sufficient_rank_passes(self):
        fs_bridge._check_orthogonalisation_rank(
            {"mag": PARCELLATION_N}, "symmetric", PARCELLATION
        )

    def test_other_orthogonalisations_are_not_checked(self):
        # 'local' and None do not require full-rank parcel time courses.
        for orthogonalisation in ("local", None):
            fs_bridge._check_orthogonalisation_rank(
                {"mag": 40}, orthogonalisation, PARCELLATION
            )

    def test_a_rank_mne_resolves_itself_is_not_checked(self):
        # 'info'/None are resolved inside MNE, so there is no number to test.
        fs_bridge._check_orthogonalisation_rank("info", "symmetric", PARCELLATION)

    def test_the_parcel_count_comes_from_the_atlas(self):
        assert fs_bridge._n_parcels(PARCELLATION) == PARCELLATION_N


# ---------------------------------------------------------------------------
# Report payloads
# ---------------------------------------------------------------------------


class TestDropMissingPlots:
    """osl-ephys' report copies plots by key without checking for None."""

    def test_a_plot_that_was_not_rendered_is_dropped(self):
        payload = fs_bridge._drop_missing_plots(
            {"coregister": True, "coreg_plot": None}
        )
        assert payload == {"coregister": True}

    def test_a_rendered_plot_is_kept(self):
        payload = fs_bridge._drop_missing_plots(
            {"coreg_plot": "sub-007/fs_src/coreg.html"}
        )
        assert payload == {"coreg_plot": "sub-007/fs_src/coreg.html"}

    def test_non_plot_keys_keep_their_none_values(self):
        # e.g. n_init_coreg, which the report renders as "None".
        payload = fs_bridge._drop_missing_plots(
            {"n_init_coreg": None, "n_epochs": None}
        )
        assert payload == {"n_init_coreg": None, "n_epochs": None}


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
        cov = fs_bridge._compute_data_cov(
            data, is_epochs, "shrunk", options["rank"]
        )
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
            cov = fs_bridge._compute_data_cov(
                sphere_setup.epochs, True, method, None
            )
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


# ---------------------------------------------------------------------------
# Memory: decimation, budget, and the fused beamform-onto-grid path
# ---------------------------------------------------------------------------


class TestDecimate:
    """`decim` divides the source array, which is what sets peak memory."""

    def test_none_and_one_leave_the_data_untouched(self, sphere_setup):
        for decim in (None, 1):
            out = fs_bridge._decimate(sphere_setup.epochs, decim, True, [1, 32])
            assert out is sphere_setup.epochs

    def test_epochs_are_decimated(self, sphere_setup):
        epochs = sphere_setup.epochs.copy()
        n_times = len(epochs.times)

        out = fs_bridge._decimate(epochs, 2, True, [1, 32])

        assert out.info["sfreq"] == pytest.approx(sphere_setup.epochs.info["sfreq"] / 2)
        assert len(out.times) == int(np.ceil(n_times / 2))

    def test_continuous_data_is_resampled(self, sphere_setup):
        raw = sphere_setup.raw.copy()
        out = fs_bridge._decimate(raw, 2, False, [1, 32])
        assert out.info["sfreq"] == pytest.approx(sphere_setup.raw.info["sfreq"] / 2)

    def test_rejects_a_factor_that_would_alias_the_retained_band(self, sphere_setup):
        # 200 Hz / 4 = 50 Hz, below twice the 32 Hz low-pass.
        with pytest.raises(ValueError, match="would alias"):
            fs_bridge._decimate(sphere_setup.epochs.copy(), 4, True, [1, 32])

    def test_reports_the_largest_safe_factor(self, sphere_setup):
        with pytest.raises(ValueError, match="largest safe factor here is 3"):
            fs_bridge._decimate(sphere_setup.epochs.copy(), 4, True, [1, 32])

    def test_rejects_a_non_positive_factor(self, sphere_setup):
        with pytest.raises(ValueError, match="positive integer"):
            fs_bridge._decimate(sphere_setup.epochs.copy(), 0, True, None)

    def test_no_band_limit_means_no_alias_check(self, sphere_setup):
        out = fs_bridge._decimate(sphere_setup.epochs.copy(), 4, True, None)
        assert out.info["sfreq"] == pytest.approx(sphere_setup.epochs.info["sfreq"] / 4)


class TestMemoryBudget:
    def test_reads_slurm_mem_per_node_as_megabytes(self, monkeypatch):
        monkeypatch.setattr(fs_bridge, "_cgroup_memory_limit", lambda: None)
        monkeypatch.setenv("SLURM_MEM_PER_NODE", "131072")
        assert fs_bridge._memory_budget() == 131072 * 1024 ** 2

    def test_takes_the_smaller_of_cgroup_and_slurm(self, monkeypatch):
        monkeypatch.setattr(fs_bridge, "_cgroup_memory_limit", lambda: 8 * 1024 ** 3)
        monkeypatch.setenv("SLURM_MEM_PER_NODE", "131072")
        assert fs_bridge._memory_budget() == 8 * 1024 ** 3

    def test_none_when_nothing_reports_a_limit(self, monkeypatch):
        monkeypatch.setattr(fs_bridge, "_cgroup_memory_limit", lambda: None)
        monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
        assert fs_bridge._memory_budget() is None

    def test_cgroup_sentinel_values_read_as_unlimited(self, monkeypatch, tmp_path):
        # cgroup v2 writes "max"; v1 writes a number near 2**63.
        for value in ("max", str(2 ** 63 - 1)):
            path = tmp_path / "memory.max"
            path.write_text(value)
            monkeypatch.setattr(fs_bridge.Path, "read_text", lambda self: value)
            monkeypatch.setattr(
                "builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError())
            )
            assert fs_bridge._cgroup_memory_limit() is None


class TestSourceMemoryCheck:
    """np.zeros maps lazily, so an oversized array OOM-kills instead of raising."""

    def test_raises_when_the_array_cannot_fit(self, monkeypatch):
        monkeypatch.setattr(fs_bridge, "_memory_budget", lambda: 8 * 1024 ** 3)
        # 2859 voxels x 1201 samples x 3200 trials -- the shape that killed
        # sub-044 with no traceback.
        with pytest.raises(MemoryError, match="Source reconstruction needs"):
            fs_bridge._check_source_memory(2859, n_times=1201, n_trials=3200)

    def test_the_error_names_decim_as_the_lever(self, monkeypatch):
        monkeypatch.setattr(fs_bridge, "_memory_budget", lambda: 1024 ** 3)
        with pytest.raises(MemoryError, match="decim"):
            fs_bridge._check_source_memory(2859, n_times=1201, n_trials=3200)

    def test_passes_when_the_array_fits(self, monkeypatch):
        monkeypatch.setattr(fs_bridge, "_memory_budget", lambda: 128 * 1024 ** 3)
        fs_bridge._check_source_memory(2859, n_times=201, n_trials=3200)

    def test_warns_when_the_array_dominates_the_budget(self, monkeypatch, caplog):
        monkeypatch.setattr(fs_bridge, "_memory_budget", lambda: 128 * 1024 ** 3)
        with caplog.at_level("WARNING", logger=fs_bridge.logger.name):
            # sub-044's real shape: 41 GiB of source array in a 128 GiB
            # job, which fits, but only just once the parcellation makes its
            # own copy of it.
            fs_bridge._check_source_memory(2859, n_times=1201, n_trials=3200)
        assert "source reconstruction will use at least" in caplog.text

    def test_silent_when_the_budget_is_unknown(self, monkeypatch):
        monkeypatch.setattr(fs_bridge, "_memory_budget", lambda: None)
        fs_bridge._check_source_memory(2859, n_times=1201, n_trials=3200)


class TestApplyLcmvToMni:
    """The fused path must match beamform-then-morph exactly, at half the peak."""

    @staticmethod
    def _mapping(monkeypatch, setup, n_grid=5):
        grid = np.zeros((3, n_grid))
        grid[0] = np.arange(n_grid) * 8.0
        n_sources = setup.fwd["src"][0]["nuse"]

        monkeypatch.setattr(
            fs_bridge, "_mni_grid_from_reference", lambda *a, **k: grid
        )
        monkeypatch.setattr(
            fs_bridge.mne, "head_to_mni", lambda *a, **k: np.zeros((n_sources, 3))
        )
        monkeypatch.setattr(fs_bridge, "_read_mri_head_t", lambda path: None)

        return fs_bridge._mni_mapping(
            setup.fwd,
            subject="sub-007",
            subjects_dir="/fs",
            trans_path="/fs/trans.fif",
            parcellation_file=PARCELLATION,
            reference_brain="parcellation",
            spatial_resolution=8,
        )

    @staticmethod
    def _filters(setup, data, is_epochs):
        import mne

        cov = fs_bridge._compute_data_cov(data, is_epochs, "shrunk", {"mag": 20})
        return mne.beamformer.make_lcmv(
            data.info,
            setup.fwd,
            cov,
            reg=0.05,
            noise_cov=None,
            pick_ori="max-power",
            weight_norm="unit-noise-gain-invariant",
            rank={"mag": 20},
            reduce_rank=True,
        )

    def test_matches_the_two_step_route_on_epochs(self, sphere_setup, monkeypatch):
        mapping = self._mapping(monkeypatch, sphere_setup)
        filters = self._filters(sphere_setup, sphere_setup.epochs, True)

        fused = fs_bridge._apply_lcmv_to_mni(
            sphere_setup.epochs, filters, True, mapping
        )
        stepwise, _ = fs_bridge._transform_to_mni(
            sphere_setup.fwd,
            fs_bridge._apply_lcmv(sphere_setup.epochs, filters, True),
            subject="sub-007",
            subjects_dir="/fs",
            trans_path="/fs/trans.fif",
            parcellation_file=PARCELLATION,
            reference_brain="parcellation",
            spatial_resolution=8,
        )

        assert fused.shape == stepwise.shape
        np.testing.assert_allclose(fused, stepwise, rtol=1e-6)

    def test_matches_the_two_step_route_on_continuous_data(
        self, sphere_setup, monkeypatch
    ):
        mapping = self._mapping(monkeypatch, sphere_setup)
        filters = self._filters(sphere_setup, sphere_setup.raw, False)

        fused = fs_bridge._apply_lcmv_to_mni(sphere_setup.raw, filters, False, mapping)
        stepwise, _ = fs_bridge._transform_to_mni(
            sphere_setup.fwd,
            fs_bridge._apply_lcmv(sphere_setup.raw, filters, False),
            subject="sub-007",
            subjects_dir="/fs",
            trans_path="/fs/trans.fif",
            parcellation_file=PARCELLATION,
            reference_brain="parcellation",
            spatial_resolution=8,
        )

        assert fused.shape == stepwise.shape
        np.testing.assert_allclose(fused, stepwise, rtol=1e-6)

    def test_output_is_the_single_precision_source_dtype(
        self, sphere_setup, monkeypatch
    ):
        mapping = self._mapping(monkeypatch, sphere_setup)
        filters = self._filters(sphere_setup, sphere_setup.epochs, True)
        out = fs_bridge._apply_lcmv_to_mni(
            sphere_setup.epochs, filters, True, mapping
        )
        assert out.dtype == fs_bridge._SOURCE_DTYPE

    def test_grid_points_with_no_nearby_dipole_stay_at_zero(
        self, sphere_setup, monkeypatch
    ):
        # The synthetic dipoles all sit at the MNI origin, so only the first
        # grid point is within one 8 mm step.
        mapping = self._mapping(monkeypatch, sphere_setup)
        filters = self._filters(sphere_setup, sphere_setup.epochs, True)

        out = fs_bridge._apply_lcmv_to_mni(
            sphere_setup.epochs, filters, True, mapping
        )
        assert np.any(out[0] != 0)
        np.testing.assert_array_equal(out[1:], 0)

    def test_raises_on_an_empty_epochs_object(self, sphere_setup, monkeypatch):
        mapping = self._mapping(monkeypatch, sphere_setup)
        filters = self._filters(sphere_setup, sphere_setup.epochs, True)

        empty = sphere_setup.epochs[:0]
        with pytest.raises(ValueError, match="No epochs to beamform"):
            fs_bridge._apply_lcmv_to_mni(empty, filters, True, mapping)
