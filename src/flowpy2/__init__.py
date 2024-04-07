from .flow_type import (FlowType,
                        register_flow_type,
                        get_flow_type)

from .coords import CoordStruct
from .indexers import CompIndexer, TimeIndexer
from .datastruct import DataStruct
from .flowstruct import FlowStructND

from .gradient import (first_derivative,
                       second_derivative,
                       set_default_gradient,
                       register_gradient,
                       reset_default,
                       return_gradients)
from .arrays import HAVE_CUPY
from .io import HAVE_NETCDF4

from matplotlib import RcParams

def _get_validators():
    validators = {}
    from .gradient import _rc_validators
    for k, v in _rc_validators.items():
        validators[f'gradient.{k}'] = v

    from .io import _rc_validators
    for k, v in _rc_validators.items():
        validators[f'io.{k}'] = v

    from .arrays import _rc_validators
    for k, v in _rc_validators.items():
        validators[f'arrays.{k}'] = v
    return validators

def _get_rc_defaults():
    defaults = {}
    from .gradient import _rc_params
    for k, v in _rc_params.items():
        defaults[f'gradient.{k}'] = v

    from .io import _rc_params
    for k, v in _rc_params.items():
        defaults[f'io.{k}'] = v

    from .arrays import _rc_params
    for k, v in _rc_params.items():
        defaults[f'arrays.{k}'] = v

    return defaults


rcParams = RcParams()
rcParams.validate = _get_validators()
dict.update(rcParams,_get_rc_defaults())