from __future__ import annotations

import unittest
from pathlib import Path


class NorthstarExamplesTest(unittest.TestCase):
    def test_example_configs_load(self) -> None:
        from northstar.config.loader import load_run_config

        repo_root = Path(__file__).resolve().parent.parent
        flux2_config = repo_root / "northstar" / "examples" / "flux2_smoke.toml"
        zimage_config = repo_root / "northstar" / "examples" / "zimage_smoke.toml"

        flux2 = load_run_config(flux2_config)
        zimage = load_run_config(zimage_config)

        self.assertEqual(flux2.run.architecture, "flux2")
        self.assertEqual(zimage.run.architecture, "zimage")


if __name__ == "__main__":
    unittest.main()
