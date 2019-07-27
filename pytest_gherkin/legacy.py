from functools import wraps
from .gherkin import action


def parse(s):
    return s.replace("{", "<").replace("}", ">")


def legacy_step(text, target_fixture=None):
    def decorator(fn):
        result_key = target_fixture or fn.__name__

        @action(text)
        @wraps(fn)
        def wrapper(**kw):
            return {result_key: fn(**kw)}

        return wrapper

    return decorator


given = legacy_step
when = legacy_step
then = legacy_step
