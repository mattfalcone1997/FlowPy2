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
                       return_gradients)

from .integrate import (register_cumulat_integration,
                        register_integration)
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

    from .integrate import _rc_validators
    for k, v in _rc_validators.items():
        validators[f'integrate.{k}'] = v

    return validators


def _get_rc_defaults(rcparams: RcParams):
    from .gradient import _rc_params
    for k, v in _rc_params.items():
        rcparams[f'gradient.{k}'] = v

    from .io import _rc_params
    for k, v in _rc_params.items():
        rcparams[f'io.{k}'] = v

    from .arrays import _rc_params
    for k, v in _rc_params.items():
        rcparams[f'arrays.{k}'] = v

    from .integrate import _rc_params
    for k, v in _rc_params.items():
        rcparams[f'integrate.{k}'] = v


rcParams = RcParams()
rcParams.validate = _get_validators()
_get_rc_defaults(rcParams)
