# -*- coding: utf-8 -*-
import re
from os.path import join, dirname
from os import listdir
import itertools
from datetime import datetime, timedelta
from collections import defaultdict, namedtuple
from itertools import product
from inspect import signature, Parameter
from gherkin.parser import Parser
from pprint import pformat
import logging
import pytest
from _pytest.fixtures import FuncFixtureInfo


LOGGER = logging.getLogger(__file__)
_AVAILABLE_ACTIONS = dict()
_FIXTURES = dict()
_ACTION_REGEX = re.compile(r"<([^>]+)>")
MARKS = set()


class FeatureFile(pytest.File):
    def _get_example_sets(self, examples_list):
        """
        Scenarios can have multiple tables, I guess
        the idea is to make permutations of each row.
        """
        result = []
        seen_keys = set()
        for example in examples_list:
            table_keys = [cell["value"] for cell in example["tableHeader"]["cells"]]
            table_keys_set = set(table_keys)
            duplicated_keys = table_keys_set & seen_keys
            assert not duplicated_keys, f"Found some duplicated_keys: {duplicated_keys}"
            seen_keys |= table_keys_set
            table_value_dicts = []
            for table_row in example["tableBody"]:
                table_value_dicts.append(
                    dict(
                        zip(table_keys, [cell["value"] for cell in table_row["cells"]])
                    )
                )
            result.append(table_value_dicts)
        return product(*result)

    def collect(self):
        parser = Parser()
        with self.fspath.open() as handle:
            feature = parser.parse(handle.read())

            # Group the feature's children by type
            children = defaultdict(list)
            for child in feature["feature"].get("children", []):
                children[child["type"]].append(child)

            backgrounds = children.get("Background", [])

            self.obj = dict()

            for scenario_index, scenario_outline in enumerate(
                children["ScenarioOutline"]
            ):
                for example in self._get_example_sets(scenario_outline["examples"]):
                    example_values = "-".join([v for d in example for v in d.values()])

                    function = ScenarioOutline(
                        name=scenario_outline["name"] + ": " + example_values,
                        parent=self,
                        spec=scenario_outline,
                        scenario_index=scenario_index,
                        example=example,
                        backgrounds=backgrounds,
                    )
                    for mark in MARKS:
                        function = getattr(pytest.mark, mark)(function)
                    yield function

            for scenario_index, scenario_outline in enumerate(
                children["Scenario"], -1000000
            ):
                function = ScenarioOutline(
                    name=scenario_outline["name"],
                    parent=self,
                    spec=scenario_outline,
                    scenario_index=scenario_index,
                    backgrounds=backgrounds,
                )
                for mark in MARKS:
                    function = getattr(pytest.mark, mark)(function)
                yield function


# A modified of the class method from FixtureManager, to
# regenerate the fixtureinfo object from a list of arguments,
# instead of by inspecting a function
def getfixtureinfo(self, node, func, cls, *, argnames):

    usefixtures = itertools.chain.from_iterable(
        mark.args for mark in node.iter_markers(name="usefixtures")
    )
    initialnames = tuple(usefixtures) + tuple(argnames)
    fm = node.session._fixturemanager
    initialnames, names_closure, arg2fixturedefs = fm.getfixtureclosure(
        initialnames, node, ignore_args=self._get_direct_parametrize_args(node)
    )
    return FuncFixtureInfo(
        tuple(argnames), initialnames, names_closure, arg2fixturedefs
    )


class ScenarioOutline(pytest.Function):
    def __init__(
        self, *, name, parent, spec, backgrounds, scenario_index, example=None
    ):
        super().__init__(name, parent)
        self.spec = spec
        self.example = example or []
        self.backgrounds = backgrounds
        self.scenario_index = scenario_index

    def _getobj(self):
        def wrapper(request):
            # assert False, repr(request)
            # self._rrequest = request
            self._runtest(request=request)

        return wrapper

    def _runtest(self, request):
        try:
            context = dict()
            steps = []
            for background in self.backgrounds:
                steps.extend(background["steps"])
            steps.extend(self.spec["steps"])
            for step in steps:
                # Identify the action which will handle this step, along
                # with the parameter names which need to be passed in
                # explicitly from the gherkin feature file
                stripped_step = _ACTION_REGEX.sub("<>", step["text"])
                step_arg_names = [
                    match.group(1) for match in _ACTION_REGEX.finditer(step["text"])
                ]
                if stripped_step not in _AVAILABLE_ACTIONS:
                    raise GherkinException(self, "Undefined step", stripped_step)
                action = _AVAILABLE_ACTIONS[stripped_step]

                # This defines how examples are mapped into keywords for the
                # step function
                argument_map = dict(zip(step_arg_names, action.argument_names))

                # add any extra args requested, which aren't specifically provided in params
                desired_kwargs = set(signature(action.function).parameters)
                arguments = {
                    argument_map[example_key]: example_value
                    for example_table in self.example
                    for example_key, example_value in example_table.items()
                    if argument_map.get(example_key) in desired_kwargs
                }

                # Respect any literal values in the step, assuming that any token
                # which didn't match an example is literal
                for literal_argument_value, argument_name in argument_map.items():
                    if argument_name not in arguments:
                        arguments[argument_name] = literal_argument_value

                for context_key, context_value in context.items():
                    if context_key not in arguments and context_key in desired_kwargs:
                        arguments[context_key] = context_value
                desired_kwargs -= set(arguments)

                # Manually refresh the request object, as though
                # we were a function with these argument names
                self._fixtureinfo = getfixtureinfo(
                    self=self.session._fixturemanager,
                    node=self,
                    func=self.obj,
                    cls=self.cls,
                    argnames=desired_kwargs,
                )
                self.fixturenames = self._fixtureinfo.names_closure
                self._initrequest()
                for desired_kwarg in desired_kwargs:
                    fixturedef, = self.session._fixturemanager._arg2fixturedefs[
                        desired_kwarg
                    ]
                    self._request._compute_fixture_value(fixturedef)
                    print(desired_kwarg, "is", fixturedef)
                    arguments[desired_kwarg] = fixturedef
                # The following line raises an exception, as galaxy STILL isn't available
                self._request.getfixturevalue("galaxy")
                result = apply_type_hints_to_arguments(
                    item=self, function=action.function, **arguments
                )
                if isinstance(result, dict):
                    # Update the context available for future steps with the result of this step
                    context.update(result)
        except:
            LOGGER.exception("oops")
            raise

    def repr_failure(self, excinfo):
        """ called when self.runtest() raises an exception. """
        if isinstance(excinfo.value, GherkinException):
            return ": ".join(excinfo.value.args[1:3])
        return f"Unexpected exception: {excinfo.value}"

    def reportinfo(self):
        return self.fspath, 0, "Scenario: %s" % self.name


def identity(anything):
    return anything


def apply_type_hints_to_arguments(*, item, function, **kw):
    sig = signature(function)
    parms = sig.parameters
    new_arguments = {
        k: v if parms[k].annotation == Parameter.empty else parms[k].annotation(v)
        for k, v in kw.items()
    }
    try:
        sig.bind(**new_arguments)
    except TypeError as ex:
        raise GherkinException(
            item, f"Cannot execute {function.__name__}", ", ".join(ex.args)
        )
    return function(**new_arguments)


def map_arguments(*, arguments, typemap, default_type=identity):
    if typemap is None:
        return arguments
    return {k: typemap.get(k, default_type)(v) for k, v in arguments.items()}


class GherkinException(Exception):
    pass


Action = namedtuple("Action", ["function", "argument_names"])


def action(name):
    arg_names = [match.group(1) for match in _ACTION_REGEX.finditer(name)]
    stripped_name = _ACTION_REGEX.sub("<>", name)

    def decorator(fn):
        _AVAILABLE_ACTIONS[stripped_name] = Action(
            function=fn, argument_names=arg_names
        )
        return fn

    return decorator


def register(raw_fn=None):
    """
    Used to expose a pytest fixture without also
    re-registering it with pytest, which pytest
    would get upset about
    """

    def decorator(pytest_fixture):
        scope = pytest_fixture._pytestfixturefunction.scope
        assert scope in ["function", "session"]
        name = pytest_fixture.__name__
        fn = pytest_fixture.__wrapped__
        assert name not in _FIXTURES
        _FIXTURES[name] = (fn, scope)

    if raw_fn is not None:
        return decorator(pytest_fixture=raw_fn)
    return decorator


def add_mark(mark):
    MARKS.add(mark)


def fixture(raw_fn=None, *, scope="function"):
    assert scope in ["function", "session"]

    def decorator(fn):
        name = fn.__name__
        assert name not in _FIXTURES
        _FIXTURES[name] = (fn, scope)
        # Also register it with the normal pytest fixture, in case
        # anything else also wants it
        return pytest.fixture(scope=scope)(fn)

    if raw_fn is not None:
        return decorator(fn=raw_fn)
    return decorator


def retrieve_fixture(*, name, scope, _cache=dict()):
    fn, fn_scope = _FIXTURES[name]
    scope_key = (fn, scope[fn_scope])
    if scope_key not in _cache:
        kwargs = {
            param: retrieve_fixture(name=param, scope=scope)
            for param in signature(fn).parameters
        }
        result = fn(**kwargs)
        assert result is not None, f"Fixture {name} returned no value!"
        _cache[scope_key] = result
    return _cache[scope_key]
