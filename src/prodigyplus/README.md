# Vendored `prodigy-plus-schedule-free`

This directory vendors the upstream `prodigy-plus-schedule-free` Python package so the
current Musubi tree can use `ProdigyPlusScheduleFree` without relying on an extra local
pip install.

Current vendored source:

- package: `prodigy-plus-schedule-free`
- version: `2.0.1`
- upstream: `https://github.com/LoganBooker/prodigy-plus-schedule-free`
- license: Apache-2.0

Files copied from the local package cache:

- `__init__.py`
- `core_optimiser.py`
- `prodigy_plus_schedulefree.py`
- `LICENSE`

If this package is updated later, keep the import surface stable:

- `prodigyplus.prodigy_plus_schedulefree.ProdigyPlusScheduleFree`

That path is already used by Musubi optimizer loading and tests.
