"""Extended tests for custom_preproc.py — CLI parsing and main().

Covers parse_args(), main() with mocked config loading, and the
header/footer formatting.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from custom.custom_preproc import (
    ANALYSIS_CHOICES,
    ANALYSIS_REGISTRY,
    import_analysis_module,
    parse_args,
)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Test CLI argument parsing."""

    def test_required_args(self, monkeypatch):
        """--analysis and --config are required."""
        monkeypatch.setattr(sys, "argv", ["custom_preproc.py"])
        with pytest.raises(SystemExit):
            parse_args()

    def test_valid_args(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "custom_preproc.py",
                "--analysis",
                "bad_channels",
                "--config",
                "/tmp/config.py",
            ],
        )
        args = parse_args()
        assert args.analysis == "bad_channels"
        assert args.config == "/tmp/config.py"

    def test_analysis_choices_validated(self, monkeypatch):
        """Invalid analysis names should be rejected by argparse."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "custom_preproc.py",
                "--analysis",
                "nonexistent",
                "--config",
                "/tmp/config.py",
            ],
        )
        with pytest.raises(SystemExit):
            parse_args()

    def test_all_choices_accepted(self, monkeypatch):
        """Every ANALYSIS_CHOICES value should be accepted by argparse."""
        for choice in ANALYSIS_CHOICES:
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "custom_preproc.py",
                    "--analysis",
                    choice,
                    "--config",
                    "/tmp/config.py",
                ],
            )
            args = parse_args()
            assert args.analysis == choice


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    """Test the main() entry point with mocked dependencies."""

    @patch("custom.custom_preproc.parse_args")
    @patch("custom.custom_preproc.load_config")
    @patch("custom.custom_preproc.import_analysis_module")
    def test_main_dispatches(self, mock_import, mock_config, mock_args):
        """main() should parse args, load config, import module, and call run."""
        from custom.custom_preproc import main

        mock_args.return_value = SimpleNamespace(
            analysis="bad_channels",
            config="/tmp/config.py",
        )
        mock_cfg = SimpleNamespace(
            subjects=["001"], sessions=["01"], task="restingstate"
        )
        mock_config.return_value = mock_cfg

        mock_run_fn = MagicMock()
        mock_import.return_value = mock_run_fn

        main()

        mock_config.assert_called_once_with("/tmp/config.py")
        mock_import.assert_called_once()
        mock_run_fn.assert_called_once_with(mock_cfg)

    @patch("custom.custom_preproc.parse_args")
    @patch("custom.custom_preproc.load_config")
    @patch("custom.custom_preproc.import_analysis_module")
    def test_main_prints_header(self, mock_import, mock_config, mock_args, capsys):
        """main() should print formatted header and footer."""
        from custom.custom_preproc import main

        mock_args.return_value = SimpleNamespace(
            analysis="apply_hfc",
            config="/tmp/config.py",
        )
        mock_config.return_value = SimpleNamespace()
        mock_import.return_value = MagicMock()

        main()

        captured = capsys.readouterr()
        assert "apply_hfc" in captured.out.lower() or "applyhfc" in captured.out.lower()


# ---------------------------------------------------------------------------
# main() — error handling branches
# ---------------------------------------------------------------------------


class TestMainErrorHandling:
    """Test that main() handles errors gracefully."""

    @patch("custom.custom_preproc.parse_args")
    @patch("custom.custom_preproc.load_config")
    @patch("custom.custom_preproc.import_analysis_module")
    def test_main_file_not_found(self, mock_import, mock_config, mock_args, capsys):
        from custom.custom_preproc import main

        mock_args.return_value = SimpleNamespace(
            analysis="bad_channels", config="/tmp/config.py"
        )
        mock_config.return_value = SimpleNamespace()
        mock_import.return_value = MagicMock(
            side_effect=FileNotFoundError("No such file")
        )

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "File not found" in captured.out

    @patch("custom.custom_preproc.parse_args")
    @patch("custom.custom_preproc.load_config")
    @patch("custom.custom_preproc.import_analysis_module")
    def test_main_value_error(self, mock_import, mock_config, mock_args, capsys):
        from custom.custom_preproc import main

        mock_args.return_value = SimpleNamespace(
            analysis="bad_channels", config="/tmp/config.py"
        )
        mock_config.return_value = SimpleNamespace()
        mock_import.return_value = MagicMock(side_effect=ValueError("Bad config"))

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "Configuration error" in captured.out

    @patch("custom.custom_preproc.parse_args")
    @patch("custom.custom_preproc.load_config")
    @patch("custom.custom_preproc.import_analysis_module")
    def test_main_generic_error(self, mock_import, mock_config, mock_args, capsys):
        from custom.custom_preproc import main

        mock_args.return_value = SimpleNamespace(
            analysis="bad_channels", config="/tmp/config.py"
        )
        mock_config.return_value = SimpleNamespace()
        mock_import.return_value = MagicMock(
            side_effect=RuntimeError("Unexpected error")
        )

        result = main()
        captured = capsys.readouterr()
        assert result == 1
        assert "Analysis failed" in captured.out


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Test the load_config function."""

    def test_missing_config_raises(self):
        from custom.preprocessing._config import load_config

        with pytest.raises(FileNotFoundError, match="not found"):
            load_config("/nonexistent/path/config.py")


# ---------------------------------------------------------------------------
# import_analysis_module — edge cases
# ---------------------------------------------------------------------------


class TestImportAnalysisModuleExtended:
    def test_applyzca_and_zcafilter_return_same_module(self):
        """applyzca and zcafilter should resolve to the same run function."""
        run1 = import_analysis_module("applyzca")
        run2 = import_analysis_module("zcafilter")
        assert run1.__module__ == run2.__module__

    def test_all_registered_run_functions_take_cfg(self):
        """Every registered module's run() must accept a cfg parameter."""
        import inspect

        for key in ANALYSIS_REGISTRY:
            run_fn = import_analysis_module(key)
            sig = inspect.signature(run_fn)
            assert "cfg" in sig.parameters, (
                f"run() for {key} does not accept 'cfg' parameter"
            )
