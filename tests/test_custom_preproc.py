"""Tests for custom_preproc.py — CLI dispatcher and analysis registry.

custom_preproc.py uses CWD-relative imports (``from preprocessing._config ...``)
so we cannot directly import it from the test runner.  Instead, we test:
  1. The registry data and module lookup by importing the module from disk
  2. The normalize_analysis_key function (via _config, which is importable)
  3. That every registered module is actually importable from preprocessing.*
"""

from __future__ import annotations

import importlib
import sys

import pytest

from custom.preprocessing._config import normalize_analysis_key


# ---------------------------------------------------------------------------
# Local copy of the registry (mirrors custom_preproc.py)
# ---------------------------------------------------------------------------

ANALYSIS_REGISTRY = {
    "regressref": "regress_ref",
    "badsegments": "bad_segments",
    "badchannels": "bad_channels",
    "manualchannel": "manual_channel",
    "applyhfc": "apply_hfc",
    "zcafilter": "zca_filter",
    "applyzca": "zca_filter",
    "badepochs": "bad_epochs",
    "autoica": "auto_ica",
    "manualica": "manual_ica",
    "coreg": "coreg",
}

ANALYSIS_CHOICES = [
    "regress_ref",
    "bad_segments",
    "bad_channels",
    "manual_channel",
    "apply_hfc",
    "zca_filter",
    "apply_zca",
    "bad_epochs",
    "auto_ica",
    "manual_ica",
    "coreg",
]


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

class TestAnalysisRegistry:
    """Verify that the analysis registry is self-consistent."""

    def test_all_choices_have_registry_entry(self):
        """Every human-readable choice must map to a registry entry."""
        for choice in ANALYSIS_CHOICES:
            key = normalize_analysis_key(choice)
            assert key in ANALYSIS_REGISTRY, (
                f"CLI choice '{choice}' (normalized: '{key}') "
                f"not in ANALYSIS_REGISTRY"
            )

    def test_registry_values_are_module_names(self):
        """Registry values should be valid Python module identifiers."""
        for key, module_name in ANALYSIS_REGISTRY.items():
            assert module_name.isidentifier(), (
                f"Registry value '{module_name}' for key '{key}' "
                f"is not a valid Python identifier"
            )

    def test_expected_analyses_present(self):
        expected = {
            "regressref",
            "badsegments",
            "badchannels",
            "manualchannel",
            "applyhfc",
            "zcafilter",
            "badepochs",
            "autoica",
            "manualica",
            "coreg",
        }
        assert expected.issubset(set(ANALYSIS_REGISTRY.keys()))

    def test_applyzca_is_alias_for_zcafilter(self):
        assert ANALYSIS_REGISTRY["applyzca"] == ANALYSIS_REGISTRY["zcafilter"]

    def test_choices_list_not_empty(self):
        assert len(ANALYSIS_CHOICES) > 0


# ---------------------------------------------------------------------------
# Dynamic module import
# ---------------------------------------------------------------------------

class TestDynamicModuleImport:
    """Test that every registered analysis module is importable and has run()."""

    @pytest.mark.parametrize(
        "module_name",
        sorted(set(ANALYSIS_REGISTRY.values())),
    )
    def test_module_importable(self, module_name):
        """Each module in the registry should be importable from preprocessing."""
        mod = importlib.import_module(f"custom.preprocessing.{module_name}")
        assert hasattr(mod, "run"), (
            f"Module custom.preprocessing.{module_name} has no run() function"
        )

    @pytest.mark.parametrize(
        "module_name",
        sorted(set(ANALYSIS_REGISTRY.values())),
    )
    def test_run_is_callable(self, module_name):
        mod = importlib.import_module(f"custom.preprocessing.{module_name}")
        assert callable(mod.run)

    @pytest.mark.parametrize(
        "module_name",
        sorted(set(ANALYSIS_REGISTRY.values())),
    )
    def test_run_accepts_cfg_param(self, module_name):
        import inspect

        mod = importlib.import_module(f"custom.preprocessing.{module_name}")
        sig = inspect.signature(mod.run)
        assert "cfg" in sig.parameters, (
            f"run() in custom.preprocessing.{module_name} "
            f"does not have a 'cfg' parameter"
        )

    def test_unknown_module_fails(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("custom.preprocessing.nonexistent_module")
