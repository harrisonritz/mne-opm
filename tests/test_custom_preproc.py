"""Tests for custom_preproc.py — CLI dispatcher and analysis registry.

Now that custom_preproc.py uses absolute imports (``custom.preprocessing.*``),
we can import it directly as a library module.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from custom.custom_preproc import (
    ANALYSIS_CHOICES,
    ANALYSIS_REGISTRY,
    import_analysis_module,
)
from custom.preprocessing._config import normalize_analysis_key


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
            "regress",
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
# import_analysis_module
# ---------------------------------------------------------------------------

class TestImportAnalysisModule:
    """Tests for the import_analysis_module dispatcher."""

    def test_unknown_key_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown analysis"):
            import_analysis_module("nonexistent_key")

    def test_import_returns_callable(self):
        for key in ["badsegments", "badchannels", "regress", "applyhfc"]:
            run_func = import_analysis_module(key)
            assert callable(run_func), f"run() for {key} should be callable"

    def test_imported_run_has_cfg_param(self):
        run_func = import_analysis_module("badsegments")
        sig = inspect.signature(run_func)
        assert "cfg" in sig.parameters

    @pytest.mark.parametrize("key", list(ANALYSIS_REGISTRY.keys()))
    def test_all_registered_modules_importable(self, key):
        """Every module in the registry should be importable."""
        run_func = import_analysis_module(key)
        assert callable(run_func)


# ---------------------------------------------------------------------------
# Dynamic module import (direct)
# ---------------------------------------------------------------------------

class TestDynamicModuleImport:
    """Test that every registered analysis module is importable via importlib."""

    @pytest.mark.parametrize(
        "module_name",
        sorted(set(ANALYSIS_REGISTRY.values())),
    )
    def test_module_importable(self, module_name):
        mod = importlib.import_module(f"custom.preprocessing.{module_name}")
        assert hasattr(mod, "run")

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
        mod = importlib.import_module(f"custom.preprocessing.{module_name}")
        sig = inspect.signature(mod.run)
        assert "cfg" in sig.parameters

    def test_unknown_module_fails(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("custom.preprocessing.nonexistent_module")
