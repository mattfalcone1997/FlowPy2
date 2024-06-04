import logging
import scipy

from typing import Callable
from scipy.integrate import (simpson,
                             cumulative_trapezoid)

if scipy.__version__ >= "1.12":
    from scipy.integrate import cumulative_simpson

logger = logging.getLogger(__name__)

def _validate_integration(val):
    set_default_integration(val)
    return val

def _validate_cumulat_integration(val):
    set_default_cumulat_integration(val)
    return val

_rc_params = {'default_integrate': 'scipy_simps',
              'default_cumulat_integrate': 'scipy_cumtrapz',}
_rc_validators = {'default_integrate': _validate_integration,
                  'default_cumulat_integrate': _validate_cumulat_integration}

_integrations = dict()
_cumulative_integrations = dict()

_default_integration = [None]
_default_cumulat_integration = [None]


def set_default_integration(name: str):
    if name not in _integrations:
        raise ValueError("Integrate method is invalid")

    _default_integration[0] = name

def set_default_cumulat_integration(name: str):
    if name not in _cumulative_integrations:
        raise ValueError("Cumulative integrate method is invalid")

    _default_cumulat_integration[0] = name

def register_integration(name: str,
                         integrate_method: Callable,
                         force: bool=False):
    if name in _integrations and not force:
        raise ValueError(f"Method {name} already registered")

    _integrations[name] = integrate_method

def register_cumulat_integration(name: str,
                                integrate_method: Callable,
                                force: bool=False):
    
    if name in _cumulative_integrations and not force:
        raise ValueError(f"Method {name} already registered")

    _cumulative_integrations[name] = integrate_method

def integrate(*args, method=None, **kwargs):
    if method is None:
        method = _default_integration[0]

    if method not in _integrations:
        raise ValueError(f"Invalid integration method {method}")

    return _integrations[method](*args, **kwargs)

def cumulative_integrate(*args, method=None, **kwargs):
    if method is None:
        method = _default_cumulat_integration[0]

    if method not in _cumulative_integrations:
        raise ValueError(f"Invalid integration method {method}")

    return _cumulative_integrations[method](*args, **kwargs)

register_integration('scipy_simps', simpson)
register_cumulat_integration('scipy_cumtrapz', cumulative_trapezoid)

if scipy.__version__ >= "1.12":
    register_cumulat_integration('scipy_cumsimps', cumulative_simpson)


