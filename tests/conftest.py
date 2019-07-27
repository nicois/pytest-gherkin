from collections import namedtuple
from pytest import fixture

pytest_plugins = "pytester"

SolarSystem = namedtuple("SolarSystem", ["planets", "stars"])


@fixture(scope="function")
def basket():
    return dict()


# this also exercises the 'bare' decorator
@fixture
def solar_system_factory():
    def create_solar_system(*, n_planets, n_stars=1):
        return SolarSystem(planets=n_planets, stars=n_stars)

    return create_solar_system


@fixture(scope="function")
def galaxy(solar_system_factory, client):
    class Galaxy:
        def __init__(self):
            self.solar_systems = []

        def add_ss(self):
            self.solar_systems.append(solar_system_factory(n_planets=1))

    return Galaxy()
