class SpatializeError(Exception):
    pass

from ._version import __version__
from ._util import SingletonType, in_notebook
from .result import GridSearchResult, EstimationResult
from . import session

__all__ = ["gs", "gs.idw", "gs.esi", "gs.esi.aggfunction"]
