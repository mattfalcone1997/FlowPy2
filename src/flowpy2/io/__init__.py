from .netcdf import HAVE_NETCDF4
from ._common import (valid_tag_checks,
                      cls_from_tag)

def _validate_tag_checks(val): return val in valid_tag_checks

_rc_params = {'tag_check' : 'weak'}
_rc_validators = {'tag_check': _validate_tag_checks}
