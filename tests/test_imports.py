import importlib

import pytest


@pytest.mark.parametrize("name", ["plantrel", "plantrel.hashing", "plantrel.repro", "plantrel.metrics",
                                  "plantrel.bootstrap", "plantrel.freeze", "plantrel.staging",
                                  "plantrel.entrypoints", "numpy", "scipy", "pandas", "yaml", "PIL"])
def test_core_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", ["plantrel.checkpoint", "torch", "torchvision"])
def test_torch_imports(name):
    pytest.importorskip("torch")
    importlib.import_module(name)
