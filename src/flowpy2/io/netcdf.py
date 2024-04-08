
import logging
import warnings
from ..utils import find_stack_level
import inspect
from ._common import (valid_tag_checks,
                      make_tag,
                      weak_tag_check)
try:
    import netCDF4
except ImportError:
    HAVE_NETCDF4 = False
else:
    HAVE_NETCDF4 = True

logger = logging.getLogger(__name__)


class netCDF4TagError(KeyError):
    pass


class netCDF4TagWarning(UserWarning):
    pass



def set_type_tag(cls, g, tag_attr='type_tag'):
    ref_tag = make_tag(cls)
    setattr(g, tag_attr, ref_tag)




def validate_tag(cls, g, tag_check, tag_attr='type_tag'):

    if not isinstance(g, (netCDF4.Dataset, netCDF4.Group)):
        raise TypeError("Invalid type")

    if tag_check not in valid_tag_checks:
        raise ValueError(f"Invalid tag check type: {tag_check}. "
                         f"Must be in {' '.join(valid_tag_checks)}")

    ref_tag = make_tag(cls)
    try:
        tag = getattr(g, tag_attr)
    except AttributeError:
        if tag_check != 'nocheck':
            raise netCDF4TagError(f"No tag found. Potentially "
                                  "invalid HDF5 File or Group.") from None
        else:
            logger.debug("No tag found on group.")
            return

    if ref_tag != tag:
        msg = (f"Tags do not match {tag} vs {ref_tag}. You can change "
               "tag check through keyword or rcParams")

        if tag_check == 'strict':
            raise netCDF4TagError("Strick check: "+msg)

        elif tag_check == 'warn':
            warnings.warn(msg,
                          category=netCDF4TagWarning,
                          stacklevel=find_stack_level())
        elif tag_check == 'weak':
            cls = weak_tag_check(cls,
                            tag,
                            netCDF4TagError)
            
        elif tag_check == 'nocheck':
            logger.debug("Tags do not match. You can change "
                         "tag check through keyword or rcParams")

    return cls

def make_dataset(fn_or_obj, mode=None, key=None):
    if isinstance(fn_or_obj, (str, bytes)):
        if mode is None:
            raise ValueError("Mode must be given if "
                             "str or bytes passed")

        f = netCDF4.Dataset(fn_or_obj, mode)
    elif isinstance(fn_or_obj, (netCDF4.Dataset, netCDF4.Group)):
        if mode is not None:
            raise ValueError("Mode cannot be given if File or "
                             "Group object passed")

        f = fn_or_obj

    else:
        raise TypeError("Invalid type")

    return f if key is None else f.createGroup(key)


def _get_name_file_group(g):
    if isinstance(g, netCDF4.Group):
        return f"group {g.name} in file {g.filepath()}"

    elif isinstance(g, netCDF4.Dataset):
        return f"in file {g.filepath()}"


def close(f):
    if f.parent is None:
        f.close()
    else:
        close(f.parent)


def access_dataset(fn_or_obj: str,
                   key: str = None):
    if isinstance(fn_or_obj, (str, bytes)):
        f = netCDF4.Dataset(fn_or_obj, 'r')
    elif isinstance(fn_or_obj, (netCDF4.Dataset, netCDF4.Group)):
        f = fn_or_obj
    else:
        raise TypeError("Invalid type")

    if key is not None:
        if key not in f.groups:
            raise KeyError(f"Key {key} not present "
                           f"in {_get_name_file_group(f)}")
        return f.groups[key]
    else:
        return f
