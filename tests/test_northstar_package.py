import importlib
import unittest


class NorthstarPackageTest(unittest.TestCase):
    def test_northstar_packages_are_importable(self) -> None:
        self.assertIsNotNone(importlib.import_module("northstar"))
        self.assertIsNotNone(importlib.import_module("northstar.cli"))
        self.assertIsNotNone(importlib.import_module("northstar.config"))
        self.assertIsNotNone(importlib.import_module("northstar.core"))
        self.assertIsNotNone(importlib.import_module("northstar.interop"))


if __name__ == "__main__":
    unittest.main()
