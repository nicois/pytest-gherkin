# -*- coding: utf-8 -*-
from collections import namedtuple
import pytest
from pytest import fixture
from pytest_gherkin import action


@action("There are <n> solar systems")
def initial_ss(n: int, galaxy):
    for _ in range(n):
        galaxy.add_ss()


@action("I discover another solar system")
def discover_ss(galaxy, solar_system_factory):
    galaxy.solar_systems.append(solar_system_factory(n_planets=1))


@action("I have <n> solar systems")
def count_ss(galaxy, n: int):
    assert len(galaxy.solar_systems) == n


def test_galaxy_fixture(galaxy):
    """
    Just making sure that registering a fixture with pytest_gherkin
    also implicitly registers it with pytest
    """
    galaxy.add_ss()
    assert len(galaxy.solar_systems) == 1
