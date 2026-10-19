"""Deprecated name of :mod:`spatialize.gs.esi.functionals`, renamed in version 1.3 and to be removed in 1.4.

Importing this module returns :mod:`spatialize.gs.esi.functionals` itself, with a warning.
"""
import sys
import warnings

from spatialize.gs.esi import functionals as _renamed

warnings.warn("spatialize.gs.esi.aggfunction was renamed spatialize.gs.esi.functionals in version 1.3; the old name will be removed in 1.4",
              FutureWarning, stacklevel=2)
sys.modules[__name__] = _renamed
