"""Features under study, whose interface and results may change between versions without notice.

What lives here works and is tested, while its theory or its design is still being settled. Each
feature warns once per session, the first time it is used, with an :class:`ExperimentalWarning`,
which can be silenced with :mod:`warnings` (``warnings.simplefilter("ignore", ExperimentalWarning)``).

Modules
-------
:mod:`spatialize.futures.esmi`
    Spatial entropy and mutual information of the predictive laws.
:mod:`spatialize.futures.coesi`
    Co-estimation, a variable predicted from several others through two stages of ensembles.
"""
import warnings


class ExperimentalWarning(UserWarning):
    """A feature of :mod:`spatialize.futures` is in use: its interface and results may change."""


_warned = set()


def _experimental(feature):
    """Warn once per session that ``feature`` belongs to :mod:`spatialize.futures`."""
    if feature in _warned:
        return
    _warned.add(feature)
    warnings.warn(f"{feature} is experimental (spatialize.futures): its interface and results may change "
                  "between versions without notice.", ExperimentalWarning, stacklevel=3)
