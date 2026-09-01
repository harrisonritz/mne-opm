"""Tests for dipole sign flipping and the group-level stage."""

from __future__ import annotations

import json
import textwrap

import mne
import numpy as np
import pytest

from custom.osl import group as group_stage
from custom.osl import sign_flip
from custom.osl._config import load_config


N_PARCELS = 6
N_TIMES = 40
SFREQ = 100.0


def make_parc_epochs(n_epochs=8, n_parcels=N_PARCELS, seed=0, flips=None):
    """Parcellated Epochs named the way osl-ephys' convert2mne_epochs names them."""
    rng = np.random.RandomState(seed)
    info = mne.create_info(
        [f"parcel_{i}" for i in range(n_parcels)], SFREQ, ["misc"] * n_parcels
    )

    # A shared source per parcel plus noise, so subjects' covariances are
    # genuinely comparable and sign flipping has something to lock on to.
    base = np.random.RandomState(99).randn(n_parcels, N_TIMES)
    data = np.stack(
        [base + 0.3 * rng.randn(n_parcels, N_TIMES) for _ in range(n_epochs)]
    )
    if flips is not None:
        data = data * np.asarray(flips)[np.newaxis, :, np.newaxis]

    events = np.column_stack(
        [
            np.arange(n_epochs) * N_TIMES + 10,
            np.zeros(n_epochs, int),
            np.tile([201, 202], n_epochs // 2)[:n_epochs],
        ]
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"response/left": 201, "response/right": 202},
        tmin=-0.1,
        verbose=False,
    )


def write_subject(outdir, subject, **kwargs):
    """Write a subject's parcellated epochs where the pipeline expects them."""
    path = outdir / subject / "parc"
    path.mkdir(parents=True, exist_ok=True)
    epochs = make_parc_epochs(**kwargs)
    epochs.save(path / "lcmv-parc-epo.fif", overwrite=True, verbose=False)
    return epochs


# ---------------------------------------------------------------------------
# Paths and channels
# ---------------------------------------------------------------------------


class TestPaths:
    def test_parc_file_matches_what_the_source_stage_writes(self):
        assert sign_flip.parc_file("/out", "sub-007", epoched=True).endswith(
            "/out/sub-007/parc/lcmv-parc-epo.fif"
        )
        assert sign_flip.parc_file("/out", "sub-007", epoched=False).endswith(
            "/out/sub-007/parc/lcmv-parc-raw.fif"
        )

    def test_epoched_path_carries_the_source_method_prefix(self):
        # The upstream bug this module exists for: osl-ephys' own apply_flips
        # reads 'parc-epo.fif' with no prefix, which never exists.
        path = sign_flip.parc_file("/out", "sub-007", epoched=True)
        assert path.endswith("lcmv-parc-epo.fif")
        assert not path.endswith("/parc-epo.fif")

    def test_sflip_file_naming(self):
        assert sign_flip.sflip_file("/out", "sub-007", epoched=True).endswith(
            "/out/sub-007/sub-007_sflip_lcmv-parc-epo.fif"
        )

    def test_available_subjects_reports_only_what_exists(self, tmp_path):
        write_subject(tmp_path, "sub-001")
        found = sign_flip.available_subjects(
            tmp_path, ["sub-001", "sub-002"], epoched=True
        )
        assert found == ["sub-001"]


class TestParcelChannels:
    def test_returns_names_for_the_parcel_convention(self):
        epochs = make_parc_epochs()
        assert sign_flip.parcel_channels(epochs) == [
            f"parcel_{i}" for i in range(N_PARCELS)
        ]

    def test_resolves_the_misc_fallback_to_names(self):
        # Older parcel files have no 'parcel_X' naming, and osl-ephys returns
        # the literal string 'misc' for them.
        info = mne.create_info(["a", "b"], SFREQ, ["misc"] * 2)
        raw = mne.io.RawArray(np.zeros((2, 10)), info, verbose=False)
        assert sign_flip.parcel_channels(raw) == ["a", "b"]


# ---------------------------------------------------------------------------
# Sign flipping
# ---------------------------------------------------------------------------


class TestSignFlipping:
    def test_find_template_picks_one_of_the_subjects(self, tmp_path):
        for i in range(3):
            write_subject(tmp_path, f"sub-{i:03d}", seed=i)

        template = sign_flip.find_template(
            tmp_path, ["sub-000", "sub-001", "sub-002"], n_embeddings=1
        )
        assert template in {"sub-000", "sub-001", "sub-002"}

    def test_find_template_needs_two_subjects(self, tmp_path):
        write_subject(tmp_path, "sub-000")
        with pytest.raises(ValueError, match="two or more subjects"):
            sign_flip.find_template(tmp_path, ["sub-000"], n_embeddings=1)

    def test_apply_flips_writes_a_separate_file(self, tmp_path):
        write_subject(tmp_path, "sub-000")
        flips = np.ones(N_PARCELS)

        outfile = sign_flip.apply_flips(tmp_path, "sub-000", flips)

        assert outfile.endswith("sub-000_sflip_lcmv-parc-epo.fif")
        # The input is untouched, so re-running never compounds.
        assert (tmp_path / "sub-000" / "parc" / "lcmv-parc-epo.fif").exists()

    def test_apply_flips_negates_the_marked_parcels(self, tmp_path):
        original = write_subject(tmp_path, "sub-000")
        flips = np.ones(N_PARCELS)
        flips[[1, 3]] = -1

        outfile = sign_flip.apply_flips(tmp_path, "sub-000", flips)
        flipped = mne.read_epochs(outfile, verbose=False)

        expected = original.get_data() * flips[np.newaxis, :, np.newaxis]
        np.testing.assert_allclose(flipped.get_data(), expected, rtol=1e-6)

    def test_apply_flips_is_idempotent_on_the_input(self, tmp_path):
        write_subject(tmp_path, "sub-000")
        flips = np.array([1.0, -1, 1, -1, 1, -1])

        first = mne.read_epochs(
            sign_flip.apply_flips(tmp_path, "sub-000", flips), verbose=False
        ).get_data()
        second = mne.read_epochs(
            sign_flip.apply_flips(tmp_path, "sub-000", flips), verbose=False
        ).get_data()

        np.testing.assert_allclose(first, second)

    def test_apply_flips_rejects_a_length_mismatch(self, tmp_path):
        write_subject(tmp_path, "sub-000")
        with pytest.raises(ValueError, match="flips for"):
            sign_flip.apply_flips(tmp_path, "sub-000", np.ones(N_PARCELS + 1))

    def test_flip_subject_recovers_a_known_flip(self, tmp_path):
        # sub-001 is sub-000 with parcels 1 and 4 inverted; the search should
        # find those and undo them.
        truth = np.ones(N_PARCELS)
        truth[[1, 4]] = -1
        write_subject(tmp_path, "sub-000", seed=0)
        write_subject(tmp_path, "sub-001", seed=0, flips=truth)

        result = sign_flip.flip_subject(
            outdir=tmp_path,
            subject="sub-001",
            template="sub-000",
            n_embeddings=1,
            n_init=2,
            n_iter=300,
            max_flips=3,
        )

        assert result["error"] is None
        recovered = mne.read_epochs(result["outfile"], verbose=False).get_data()
        reference = mne.read_epochs(
            sign_flip.parc_file(tmp_path, "sub-000"), verbose=False
        ).get_data()
        # Correlation per parcel should now be positive across the board.
        for parcel in range(N_PARCELS):
            r = np.corrcoef(
                recovered[:, parcel].ravel(), reference[:, parcel].ravel()
            )[0, 1]
            assert r > 0, f"parcel {parcel} still inverted (r={r:.2f})"

    def test_flip_subject_leaves_the_template_unflipped(self, tmp_path):
        write_subject(tmp_path, "sub-000")
        write_subject(tmp_path, "sub-001", seed=1)

        result = sign_flip.flip_subject(
            outdir=tmp_path, subject="sub-000", template="sub-000", n_embeddings=1
        )

        assert result["error"] is None
        assert result["n_flipped"] == 0
        # The template still gets an sflip file, so the group stage can read
        # every subject from one place.
        assert mne.read_epochs(result["outfile"], verbose=False)

    def test_flip_subject_reports_failure_rather_than_raising(self, tmp_path):
        result = sign_flip.flip_subject(
            outdir=tmp_path, subject="sub-missing", template="sub-000"
        )
        assert result["error"] is not None
        assert result["outfile"] is None


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------


GROUP_CONFIG = """
pipeline:
  subject: "000"
  task: TSX
  bids_root: {bids_root}
  outdir: {outdir}
  subject_label: "sub-{{subject}}"
  source_input: epochs
group:
  sign_flip: {{n_embeddings: 1, n_init: 1, n_iter: 50, max_flips: 2}}
  n_workers: 1
  conditions: [response/left, response/right]
  contrasts:
    - name: responseHand
      conditions: [response/left, response/right]
      weights: [0.5, -0.5]
"""


@pytest.fixture
def group_cfg(tmp_path):
    outdir = tmp_path / "osl"
    outdir.mkdir()
    for i in range(3):
        write_subject(outdir, f"sub-{i:03d}", seed=i)

    path = tmp_path / "cfg.yaml"
    path.write_text(
        textwrap.dedent(
            GROUP_CONFIG.format(bids_root=tmp_path / "bids", outdir=outdir)
        )
    )
    return load_config(path, env={}), outdir


class TestGroupStage:
    def test_runs_end_to_end(self, group_cfg):
        cfg, outdir = group_cfg
        assert group_stage.run(cfg) is True

        groupdir = outdir / "group"
        assert (groupdir / "template_subject.txt").exists()
        assert (groupdir / "group_parcel_evoked.npz").exists()
        assert (groupdir / "group_contrasts.npz").exists()

    def test_stacked_arrays_are_subjects_by_parcels_by_times(self, group_cfg):
        cfg, outdir = group_cfg
        group_stage.run(cfg)

        data = np.load(outdir / "group" / "group_parcel_evoked.npz")
        assert data["response__left"].shape == (3, N_PARCELS, N_TIMES)
        assert list(data["subjects"]) == ["sub-000", "sub-001", "sub-002"]
        assert len(data["parcels"]) == N_PARCELS
        assert len(data["times"]) == N_TIMES

    def test_contrast_is_the_weighted_combination(self, group_cfg):
        cfg, outdir = group_cfg
        group_stage.run(cfg)

        evoked = np.load(outdir / "group" / "group_parcel_evoked.npz")
        contrasts = np.load(outdir / "group" / "group_contrasts.npz")

        expected = 0.5 * evoked["response__left"] - 0.5 * evoked["response__right"]
        np.testing.assert_allclose(contrasts["responseHand"], expected)

    def test_template_summary_records_the_flips(self, group_cfg):
        cfg, outdir = group_cfg
        group_stage.run(cfg)

        summary = json.loads((outdir / "group" / "template_subject.txt").read_text())
        assert summary["template"] in {"sub-000", "sub-001", "sub-002"}
        assert set(summary["subjects"]) == {"sub-000", "sub-001", "sub-002"}
        assert summary["subjects"][summary["template"]]["n_flipped"] == 0

    def test_writes_figures(self, group_cfg):
        cfg, outdir = group_cfg
        group_stage.run(cfg)

        groupdir = outdir / "group"
        assert (groupdir / "sign_flip_summary.png").exists()
        assert (groupdir / "contrast_responseHand.png").exists()

    def test_honours_a_pinned_template(self, group_cfg):
        cfg, outdir = group_cfg
        cfg.group["template"] = "sub-002"
        group_stage.run(cfg)

        summary = json.loads((outdir / "group" / "template_subject.txt").read_text())
        assert summary["template"] == "sub-002"

    def test_discovers_subjects_when_none_are_listed(self, group_cfg):
        cfg, outdir = group_cfg
        assert "subjects" not in cfg.group
        group_stage.run(cfg)

        data = np.load(outdir / "group" / "group_parcel_evoked.npz")
        assert len(data["subjects"]) == 3

    def test_a_subject_missing_a_condition_becomes_nan(self, group_cfg, tmp_path):
        cfg, outdir = group_cfg

        # Rewrite one subject with only left responses.
        epochs = make_parc_epochs(n_epochs=4, seed=7)
        epochs = epochs["response/left"]
        epochs.save(
            outdir / "sub-002" / "parc" / "lcmv-parc-epo.fif",
            overwrite=True,
            verbose=False,
        )

        group_stage.run(cfg)
        data = np.load(outdir / "group" / "group_parcel_evoked.npz")

        index = list(data["subjects"]).index("sub-002")
        assert np.isnan(data["response__right"][index]).all()
        assert not np.isnan(data["response__left"][index]).any()

    def test_raises_without_a_group_section(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text("pipeline:\n  subject: '007'\n  task: T\n"
                        "  bids_root: /b\n  outdir: /o\n")
        cfg = load_config(path, env={})
        with pytest.raises(ValueError, match="No 'group' section"):
            group_stage.run(cfg)

    def test_raises_when_too_few_subjects_have_data(self, tmp_path):
        outdir = tmp_path / "osl"
        outdir.mkdir()
        write_subject(outdir, "sub-000")

        path = tmp_path / "cfg.yaml"
        path.write_text(
            textwrap.dedent(
                GROUP_CONFIG.format(bids_root=tmp_path / "bids", outdir=outdir)
            )
        )
        cfg = load_config(path, env={})

        with pytest.raises(ValueError, match="two or more subjects"):
            group_stage.run(cfg)

    def test_requires_epoched_source_input(self, group_cfg):
        cfg, _ = group_cfg
        cfg.pipeline.source_input = "raw"
        with pytest.raises(ValueError, match="source_input: epochs"):
            group_stage.run(cfg)


class TestContrasts:
    def test_skips_a_contrast_whose_condition_is_absent(self):
        stacked = {"a": np.ones((2, 3, 4))}
        out = group_stage._compute_contrasts(
            stacked,
            [{"name": "x", "conditions": ["a", "missing"], "weights": [1, -1]}],
        )
        assert out == {}

    def test_condition_names_survive_the_npz_key_round_trip(self):
        assert group_stage._safe_key("response/left") == "response__left"


class TestWorkerCount:
    """`group.n_workers` is a ceiling; the job's cores and subjects clamp it."""

    def test_caps_workers_at_the_cpu_budget(self, monkeypatch):
        monkeypatch.setattr(group_stage, "_available_cpus", lambda: 4)
        assert group_stage._resolve_n_workers(8, n_subjects=10) == 4

    def test_caps_workers_at_the_number_of_subjects(self, monkeypatch):
        monkeypatch.setattr(group_stage, "_available_cpus", lambda: 16)
        assert group_stage._resolve_n_workers(8, n_subjects=2) == 2

    def test_keeps_the_configured_count_when_it_fits(self, monkeypatch):
        monkeypatch.setattr(group_stage, "_available_cpus", lambda: 16)
        assert group_stage._resolve_n_workers(8, n_subjects=10) == 8

    def test_falls_back_to_the_config_when_cpus_are_unknown(self, monkeypatch):
        monkeypatch.setattr(group_stage, "_available_cpus", lambda: None)
        assert group_stage._resolve_n_workers(8, n_subjects=10) == 8

    def test_never_returns_fewer_than_one_worker(self, monkeypatch):
        monkeypatch.setattr(group_stage, "_available_cpus", lambda: 0)
        assert group_stage._resolve_n_workers(0, n_subjects=0) == 1

    def test_available_cpus_takes_the_smallest_signal(self, monkeypatch):
        monkeypatch.setattr(group_stage.os, "sched_getaffinity", lambda _: set(range(32)))
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
        monkeypatch.setenv("MAX_WORKERS", "12")
        assert group_stage._available_cpus() == 4

    def test_available_cpus_ignores_unset_and_junk_env_vars(self, monkeypatch):
        monkeypatch.setattr(group_stage.os, "sched_getaffinity", lambda _: set(range(6)))
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "")
        monkeypatch.delenv("MAX_WORKERS", raising=False)
        assert group_stage._available_cpus() == 6

    def test_sign_flip_all_runs_serially_once_clamped(self, monkeypatch, tmp_path):
        """Two subjects on one core must not start a Dask cluster."""
        monkeypatch.setattr(group_stage, "_available_cpus", lambda: 1)

        def boom(*args, **kwargs):
            raise AssertionError("should not have started a dask cluster")

        monkeypatch.setattr(group_stage, "_flip_with_dask", boom)
        monkeypatch.setattr(
            group_stage.sign_flip,
            "flip_subject",
            lambda subject, **kw: {"subject": subject, "n_flipped": 0, "error": None},
        )

        results = group_stage._sign_flip_all(
            tmp_path,
            ["sub-001", "sub-002"],
            {"n_workers": 8, "template": "sub-001"},
            epoched=True,
            source_method="lcmv",
            groupdir=tmp_path,
        )
        assert [r["subject"] for r in results] == ["sub-001", "sub-002"]
