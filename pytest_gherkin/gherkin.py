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
            self._runtest(request=request)

        return wrapper

    def _runtest(self, request):
        assert len(_AVAILABLE_ACTIONS) > 0
        print("available actions is ", len(_AVAILABLE_ACTIONS), " long")
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
                print(f"action is {action}")
                print("cx is", context)

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

                # Remove any items in desired kwargs which have been
                # fulfilled by the arguments collected so far
                desired_kwargs -= set(arguments)
                print("desired", desired_kwargs)
                print("arg", arguments)

                """
                # FIXME: I am not sure I should be doing this here
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
                """

                # Any kwargs this step needs, which haven't been
                # fulfilled via the context or literal values,
                # need to come from normal pytest fixtures.
                self._resolve_arguments(desired_kwargs=desired_kwargs, context=context)

                # Apply any variables in our context which are desired and
                # not yet provided
                print("cx", context)
                for context_key, context_value in context.items():
                    if context_key not in arguments and context_key in desired_kwargs:
                        arguments[context_key] = context_value

                result = apply_type_hints_to_arguments(
                    item=self, function=action.function, **arguments
                )
                if isinstance(result, dict):
                    # Update the context available for future steps with the result of this step
                    context.update(result)
        except Exception as ex:
            LOGGER.exception(f"oops: {ex}")
            raise

    def _resolve_arguments(self, *, desired_kwargs, context) -> None:
        """
        Ensure the context has values for each of the desired kwargs.
        It's OK if there are items in the context which are not required;
        they will be ignored.
        """
        remaining_kwargs = set(desired_kwargs) - set(context)
        """
        # Make sure there are pytest fixtures for each unsatisfied
        # kwargs, and get the actual function
        fixture_map = {
            fixture_name: self.session._fixturemanager._arg2fixturedefs[
                fixture_name
            ].func
            for fixture_name in desired_kwargs
        }
        """

        # Find a fixture which doesn't depend on any other fixtures
        while remaining_kwargs:  # ie: is not empty
            print("x", remaining_kwargs)
            for remaining_kwarg in remaining_kwargs:
                value = self._request.getfixturevalue(remaining_kwarg)
                context[remaining_kwarg] = value
                remaining_kwargs.remove(remaining_kwarg)
                print(f"added {value} to {remaining_kwarg!r}")
                break
            else:
                raise Exception("Circular reference detected")

            """
            # value = self._request._compute_fixture_value(fixturedef)
            # value = self._request.getfixturevalue(remaining_kwarg)
            # value = fixturedef.func()
            print(remaining_kwarg, "is", fixturedef, " with val ", value)
            arguments[remaining_kwarg] = value
            """

    '''
    def repr_failure(self, excinfo):
        """ called when self.runtest() raises an exception. """
        if isinstance(excinfo.value, GherkinException):
            return ": ".join(excinfo.value.args[1:3])
        return f"Unexpected exception: {excinfo.value}"
    '''

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
        print(f"adding action {name} to AA with {fn}")
        return fn

    return decorator


def add_mark(mark):
    MARKS.add(mark)
