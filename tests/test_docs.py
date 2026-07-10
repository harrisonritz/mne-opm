"""Tests for documentation infrastructure and public API docstring coverage.

Ensures that all public API members exported by custom.preprocessing have
docstrings, and that the Sphinx docs configuration is valid.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import custom.preprocessing as pp


# ---------------------------------------------------------------------------
# Docstring coverage for public API
# ---------------------------------------------------------------------------

# Collect all names from __all__ that are callable or classes
_PUBLIC_NAMES = [
    name
    for name in pp.__all__
    if not name.startswith("_") and not isinstance(getattr(pp, name, None), type(pp))
]


class TestDocstringCoverage:
    """All public API members must have a docstring."""

    @pytest.mark.parametrize("name", _PUBLIC_NAMES)
    def test_public_member_has_docstring(self, name):
        obj = getattr(pp, name)
        if callable(obj):
            assert obj.__doc__ is not None and obj.__doc__.strip(), (
                f"Public API member '{name}' is missing a docstring"
            )
        elif isinstance(obj, type):
            assert obj.__doc__ is not None and obj.__doc__.strip(), (
                f"Public API class '{name}' is missing a docstring"
            )

    def test_public_modules_have_docstrings(self):
        """Each analysis module should have a module-level docstring."""
        module_names = [
            "regress",
            "bad_segments",
            "bad_channels",
            "manual_channel",
            "apply_hfc",
            "zca_filter",
            "bad_epochs",
            "auto_ica",
            "manual_ica",
            "coreg",
        ]
        for mod_name in module_names:
            mod = importlib.import_module(f"custom.preprocessing.{mod_name}")
            assert mod.__doc__ is not None and mod.__doc__.strip(), (
                f"Module custom.preprocessing.{mod_name} is missing a docstring"
            )

    def test_private_utility_modules_have_docstrings(self):
        """Private utility modules should also have docstrings."""
        for mod_name in ["_base", "_config", "_io", "_bids_utils"]:
            mod = importlib.import_module(f"custom.preprocessing.{mod_name}")
            assert mod.__doc__ is not None and mod.__doc__.strip(), (
                f"Module custom.preprocessing.{mod_name} is missing a docstring"
            )


# ---------------------------------------------------------------------------
# Documentation infrastructure
# ---------------------------------------------------------------------------

_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


class TestDocsInfrastructure:
    """Verify that docs configuration files and key pages exist."""

    def test_conf_py_exists(self):
        assert (_DOCS_DIR / "conf.py").is_file()

    def test_index_exists(self):
        assert (_DOCS_DIR / "index.md").is_file()

    def test_installation_exists(self):
        assert (_DOCS_DIR / "installation.md").is_file()

    def test_usage_exists(self):
        assert (_DOCS_DIR / "usage.md").is_file()

    def test_tutorials_exist(self):
        tutorials = _DOCS_DIR / "tutorials"
        assert (tutorials / "index.md").is_file()
        assert (tutorials / "quickstart.md").is_file()
        assert (tutorials / "configuration.md").is_file()
        assert (tutorials / "preprocessing.md").is_file()

    def test_api_pages_exist(self):
        api = _DOCS_DIR / "api"
        assert (api / "index.md").is_file()
        assert (api / "preprocessing.md").is_file()
        assert (api / "utilities.md").is_file()

    def test_requirements_txt_exists(self):
        assert (_DOCS_DIR / "requirements.txt").is_file()

    def test_readthedocs_yaml_exists(self):
        assert (Path(__file__).resolve().parent.parent / ".readthedocs.yaml").is_file()

    def test_conf_py_imports_cleanly(self):
        """conf.py should be importable without errors."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "docs_conf", str(_DOCS_DIR / "conf.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.project == "mne-opm"
        assert "myst_parser" in mod.extensions
        assert "sphinx.ext.autodoc" in mod.extensions
