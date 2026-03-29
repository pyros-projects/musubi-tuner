from __future__ import annotations

import io
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr
from pathlib import Path


class NorthstarCliTest(unittest.TestCase):
    def test_train_parser_accepts_run_config_path(self) -> None:
        from northstar.cli.train import build_parser

        args = build_parser().parse_args(["examples/flux2_smoke.toml"])

        self.assertEqual(args.run_config, "examples/flux2_smoke.toml")

    def test_main_loads_valid_config_and_reports_invalid_config(self) -> None:
        from northstar.cli.train import main

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "valid.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    name = "flux2-smoke"
                    architecture = "flux2"

                    [model]
                    version = "klein-base-9b"
                    dit = "/models/flux2.safetensors"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(main([str(config_path)]), 0)

            invalid_path = Path(temp_dir) / "invalid.toml"
            invalid_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    name = "flux2-smoke"
                    architecture = "flux2"
                    bad_field = true

                    [model]
                    version = "klein-base-9b"
                    dit = "/models/flux2.safetensors"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main([str(invalid_path)])

        self.assertEqual(exit_code, 1)
        self.assertIn("run.bad_field", stderr.getvalue())

    def test_main_reports_missing_file_and_toml_parse_errors_cleanly(self) -> None:
        from northstar.cli.train import main

        missing_stderr = io.StringIO()
        with redirect_stderr(missing_stderr):
            missing_exit_code = main(["/definitely/missing.toml"])

        self.assertEqual(missing_exit_code, 1)
        self.assertIn("not found", missing_stderr.getvalue().lower())

        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = Path(temp_dir) / "broken.toml"
            invalid_path.write_text("[run\nname = \"oops\"\n", encoding="utf-8")

            parse_stderr = io.StringIO()
            with redirect_stderr(parse_stderr):
                parse_exit_code = main([str(invalid_path)])

        self.assertEqual(parse_exit_code, 1)
        self.assertIn("toml", parse_stderr.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
