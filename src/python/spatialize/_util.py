import numpy as np
from rich.progress import track

from spatialize import SpatializeError


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class per_call:
    """Default of a :func:`signature_overload` argument, computed anew at every call."""

    def __init__(self, factory):
        self.factory = factory


# the documented default of every `seed` and `folding_seed`: a random integer in [1000, 10000)
random_seed = per_call(lambda: int(np.random.randint(1000, 10000)))


def _default(value):
    return value.factory() if isinstance(value, per_call) else value


def signature_overload(pivot_arg, common_args, specific_args):
    def outer_function(func):
        def inner_function(*args, **kwargs):
            pivot_key, pivot_default_value, pivot_desc = pivot_arg

            if pivot_key not in common_args:
                common_args[pivot_key] = pivot_default_value

            # if a common argument is needed for the pivot argument
            # and is not in kwargs then add it with its declared
            # default value
            for arg in common_args.keys():
                if arg not in kwargs:
                    kwargs[arg] = _default(common_args[arg])

            # get the specific args for the current pivot key
            pk = kwargs[pivot_key]

            if pk not in specific_args:
                raise SpatializeError(f"{pivot_desc.capitalize()} '{pk}' not supported")

            spec_args = specific_args[pk]

            # if the specific argument is needed for the pivot argument
            # and is not in kwargs then add it with its declared
            # default value
            for arg in spec_args.keys():
                if arg not in kwargs:
                    kwargs[arg] = _default(spec_args[arg])

            # check that all arguments are consistent
            # with the pivot key
            for arg in kwargs.keys():
                if arg != pivot_key and arg not in spec_args and arg not in common_args:
                    raise SpatializeError(f"Argument '{arg}' not recognized for '{pk}' {pivot_desc.lower()}")

            return func(*args, **kwargs)

        return inner_function

    return outer_function


def get_progress_bar(list_like_obj, desc):
    if in_notebook():
        it = tqdm(range(len(list_like_obj)), desc=desc)
    else:
        it = track(range(len(list_like_obj)), description=desc)
    return it


class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


