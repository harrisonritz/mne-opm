"""Tests for preprocessing._config — configuration utilities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom.preprocessing._config import (
    ANALYSIS_CONFIG_FLAGS,
    ICA_ANALYSES,
    check_analysis_enabled,
    get_analysis_config_flag,
    normalize_analysis_key,
    validate_required_config,
)


# ---------------------------------------------------------------------------
# normalize_analysis_key
# ---------------------------------------------------------------------------

class TestNormalizeAnalysisKey:
    """Tests for removing underscores from analysis names."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("bad_segments", "badsegments"),
            ("regress_ref", "regressref"),
            ("manual_channel", "manualchannel"),
            ("auto_ica", "autoica"),
            ("apply_hfc", "applyhfc"),
            ("zca_filter", "zcafilter"),
            ("bad_epochs", "badepochs"),
            ("manual_ica", "manualica"),
            ("coreg", "coreg"),
        ],
    )
    def test_removes_underscores(self, raw: str, expected: str):
        assert normalize_analysis_key(raw) == expected

    def test_empty_string(self):
        assert normalize_analysis_key("") == ""

    def test_no_underscores(self):
        assert normalize_analysis_key("coreg") == "coreg"

    def test_multiple_underscores(self):
        assert normalize_analysis_key("a_b_c") == "abc"


# ---------------------------------------------------------------------------
# get_analysis_config_flag
# ---------------------------------------------------------------------------

class TestGetAnalysisConfigFlag:
    """Tests for mapping analysis keys to config flags."""

    def test_known_keys_return_flags(self):
        assert get_analysis_config_flag("regressref") == "_regress_ref"
        assert get_analysis_config_flag("applyhfc") == "_do_HFC"
        assert get_analysis_config_flag("zcafilter") == "_do_ZCA"
        assert get_analysis_config_flag("autoica") == "_auto_ica"
        assert get_analysis_config_flag("manualica") == "_manual_ica"
        assert get_analysis_config_flag("manualchannel") == "_manual_channels"

    def test_unknown_keys_return_none(self):
        assert get_analysis_config_flag("badsegments") is None
        assert get_analysis_config_flag("badchannels") is None
        assert get_analysis_config_flag("badepochs") is None
        assert get_analysis_config_flag("coreg") is None

    def test_nonexistent_key(self):
        assert get_analysis_config_flag("nonexistent") is None

    def test_all_flags_present(self):
        """Ensure every entry in ANALYSIS_CONFIG_FLAGS is a string."""
        for key, flag in ANALYSIS_CONFIG_FLAGS.items():
            assert isinstance(key, str)
            assert isinstance(flag, str)
            assert flag.startswith("_")


# ---------------------------------------------------------------------------
# check_analysis_enabled
# ---------------------------------------------------------------------------

class TestCheckAnalysisEnabled:
    """Tests for determining whether an analysis is enabled."""

    def test_always_enabled_analyses(self):
        """Analyses without config flags are always enabled."""
        cfg = SimpleNamespace()
        assert check_analysis_enabled(cfg, "badsegments") is True
        assert check_analysis_enabled(cfg, "badchannels") is True
        assert check_analysis_enabled(cfg, "badepochs") is True
        assert check_analysis_enabled(cfg, "coreg") is True

    def test_flag_gated_enabled(self):
        cfg = SimpleNamespace(_regress_ref=True)
        assert check_analysis_enabled(cfg, "regressref") is True

    def test_flag_gated_disabled(self):
        cfg = SimpleNamespace(_regress_ref=False)
        assert check_analysis_enabled(cfg, "regressref") is False

    def test_flag_gated_missing_defaults_false(self):
        cfg = SimpleNamespace()
        assert check_analysis_enabled(cfg, "regressref") is False

    def test_ica_requires_spatial_filter(self):
        cfg = SimpleNamespace(_auto_ica=True, spatial_filter="ica")
        assert check_analysis_enabled(cfg, "autoica") is True

    def test_ica_disabled_without_spatial_filter(self):
        cfg = SimpleNamespace(_auto_ica=True, spatial_filter=None)
        assert check_analysis_enabled(cfg, "autoica") is False

    def test_ica_disabled_with_wrong_spatial_filter(self):
        cfg = SimpleNamespace(_auto_ica=True, spatial_filter="ssp")
        assert check_analysis_enabled(cfg, "autoica") is False

    def test_ica_disabled_flag_false(self):
        cfg = SimpleNamespace(_auto_ica=False, spatial_filter="ica")
        assert check_analysis_enabled(cfg, "autoica") is False

    def test_manual_ica_requires_both(self):
        cfg = SimpleNamespace(_manual_ica=True, spatial_filter="ica")
        assert check_analysis_enabled(cfg, "manualica") is True

    def test_hfc_enabled(self):
        cfg = SimpleNamespace(_do_HFC=True)
        assert check_analysis_enabled(cfg, "applyhfc") is True

    def test_hfc_disabled(self):
        cfg = SimpleNamespace(_do_HFC=False)
        assert check_analysis_enabled(cfg, "applyhfc") is False

    def test_zca_enabled(self):
        cfg = SimpleNamespace(_do_ZCA=True)
        assert check_analysis_enabled(cfg, "zcafilter") is True


# ---------------------------------------------------------------------------
# validate_required_config
# ---------------------------------------------------------------------------

class TestValidateRequiredConfig:
    """Tests for config attribute validation."""

    def test_all_present(self):
        cfg = SimpleNamespace(a=1, b=2, c=3)
        # Should not raise
        validate_required_config(cfg, ["a", "b", "c"], "test")

    def test_missing_raises(self):
        cfg = SimpleNamespace(a=1)
        with pytest.raises(ValueError, match="Missing required config"):
            validate_required_config(cfg, ["a", "b"], "test")

    def test_missing_message_includes_names(self):
        cfg = SimpleNamespace()
        with pytest.raises(ValueError, match="xyz"):
            validate_required_config(cfg, ["xyz"], "analysis_name")

    def test_empty_required(self):
        cfg = SimpleNamespace()
        validate_required_config(cfg, [], "test")  # Should not raise


# ---------------------------------------------------------------------------
# ICA_ANALYSES constant
# ---------------------------------------------------------------------------

class TestICAAnalyses:
    def test_ica_analyses_contains_expected(self):
        assert "autoica" in ICA_ANALYSES
        assert "manualica" in ICA_ANALYSES

    def test_ica_analyses_does_not_contain_others(self):
        assert "badsegments" not in ICA_ANALYSES
        assert "regressref" not in ICA_ANALYSES
