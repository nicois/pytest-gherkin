"""Pytest plugin entry point. Used for any fixtures needed."""

import pytest
from . import gherkin


def pytest_addhooks(pluginmanager):
    """Register plugin hooks."""
    # from . import hooks
    # pluginmanager.add_hookspecs(hooks)
    pass


def pytest_collect_file(parent, path):
    if path.ext == ".feature":
        return gherkin.FeatureFile(path, parent)
