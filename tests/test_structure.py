import importlib


def test_application_modules():
    modules = [
        "app.core",
        "app.acquisition",
        "app.dsp",
        "app.ml",
        "app.storage",
        "app.gui",
    ]

    for module in modules:
        assert importlib.import_module(module) is not None