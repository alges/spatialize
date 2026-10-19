"""Deprecated name of :mod:`spatialize.gs.cat_esi.scoring`, renamed in version 1.3 and to be removed in 1.4.

Importing this module returns :mod:`spatialize.gs.cat_esi.scoring` itself, with a warning.
"""
import sys
import warnings

from spatialize.gs.cat_esi import scoring as _renamed

warnings.warn("spatialize.gs.cat_esi.score_functions was renamed spatialize.gs.cat_esi.scoring in version 1.3; the old name will be removed in 1.4",
              FutureWarning, stacklevel=2)
sys.modules[__name__] = _renamed
