"""Shared pytest fixtures/helpers for the whole suite.

`use_numpy/` and `use_manif/` each have a script with the same bare module
name (`pose_graph.py`, `bundle_adjustment.py`, ...), so importing "the
`pose_graph` module" is ambiguous unless we're careful: a naive
`sys.path.insert` + `import pose_graph` would cache whichever backend gets
imported first under `sys.modules["pose_graph"]`, and every later import of
the *other* backend's `pose_graph` would silently return that same, wrong
cached module. `import_from` below avoids that by cleaning up every module
name it touches immediately after the import, so each call starts fresh.
"""

import importlib
import sys
from pathlib import Path

import pytest
import scipy.spatial.transform  # noqa: F401 -- see warm-up note below

REPO_ROOT = Path(__file__).resolve().parent.parent
USE_NUMPY_DIR = REPO_ROOT / "use_numpy"
USE_MANIF_DIR = REPO_ROOT / "use_manif"

# utils.py lives only at the repo root and never collides with anything else,
# so it's safe to put on sys.path once for the whole test session.
sys.path.insert(0, str(REPO_ROOT))

# Warm up scipy (and the numpy FFT extension it pulls in transitively) here, before
# `_import_from` below ever runs: its `modules_before`/`modules_after` diff deletes every
# module gained during an isolated import, which works for pure-Python modules but not for
# numpy's compiled `_pocketfft_umath` extension -- reloading it after deletion raises
# "ImportError: cannot load module more than once per process". A script that imports scipy
# (e.g. use_manif/pointcloud_pose_tracking.py's run_vanilla_kf, via
# scipy.spatial.transform.Rotation) would otherwise trip this on its *second* isolated
# import. Importing it once here, before any test runs, keeps it permanently outside every
# `_import_from` call's delta so it's never a candidate for deletion.


def _import_from(directory, module_name):
    """Imports `module_name` from `directory` in isolation, so that two
    directories containing a same-named module (e.g. use_numpy/pose_graph.py
    and use_manif/pose_graph.py) never collide in `sys.modules`.
    Arguments:
        directory: pathlib.Path to prepend to sys.path for the duration of the import
        module_name: bare module name to import (no package prefix)
    Returns:
        the imported module object
    """
    directory = str(directory)
    modules_before = set(sys.modules)
    sys.path.insert(0, directory)
    try:
        importlib.invalidate_caches()
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(directory)
        for name in set(sys.modules) - modules_before:
            del sys.modules[name]


# Exposed as fixtures (not plain module-level imports) so every test file
# gets them regardless of how deep it is under tests/ -- pytest fixtures
# resolve through conftest.py files up the directory tree automatically,
# unlike a bare `from conftest import ...`, which depends on tests/ itself
# happening to already be on sys.path.
@pytest.fixture(scope="session")
def use_numpy_dir():
    return USE_NUMPY_DIR


@pytest.fixture(scope="session")
def use_manif_dir():
    return USE_MANIF_DIR


@pytest.fixture
def import_module():
    return _import_from
