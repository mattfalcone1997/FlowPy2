import h5py
import os
import numpy as np
import copy
import warnings
import logging
from typing import Union

from scipy import io
from numbers import Number

from ..utils import find_stack_level
from ._common import (valid_tag_checks,
                      make_tag,
                      weak_tag_check)
logger = logging.getLogger(__name__)

H5_Group_File = (h5py.File, h5py.Group)


class HDF5TagError(KeyError):
    pass


class HDF5TagWarning(UserWarning):
    pass



def set_type_tag(cls: type, g: Union[h5py.File, h5py.Group]):
    ref_tag = make_tag(cls)
    g.attrs['type_tag'] = ref_tag




def validate_tag(cls: type, g: Union[h5py.File, h5py.Group], tag_check):

    if not isinstance(g, H5_Group_File):
        raise TypeError("Invalid type")

    if tag_check not in valid_tag_checks:
        raise ValueError(f"Invalid tag check type: {tag_check}. "
                         f"Must be in {' '.join(valid_tag_checks)}")

    ref_tag = make_tag(cls)
    try:
        tag = g.attrs['type_tag']
    except KeyError:
        if tag_check != 'nocheck':
            raise HDF5TagError(f"No tag found in file {g.file.name} at {g.name}. "
                               "Potentially invalid HDF5 File or Group.") from None
        else:
            logger.debug("No tag found on group.")
            return

    if ref_tag != tag:
        msg = (f"Tags do not match {ref_tag} vs {tag}. "
               "You can change tag check through keyword "
               "or rcParams")

        if tag_check == 'strict':
            raise HDF5TagError("Strick check: "+msg)

        elif tag_check == 'warn':
            warnings.warn(msg,
                          category=HDF5TagWarning,
                          stacklevel=find_stack_level())

        elif tag_check == 'weak':
            cls = weak_tag_check(cls,
                                tag,
                                HDF5TagError)

        elif tag_check == 'nocheck':
            logger.debug(msg)

    return cls


def _get_name_file_group(g: Union[h5py.File, h5py.Group]):
    if isinstance(g, h5py.Group):
        return f"group {g.name} in file {g.file.filename}"

    elif isinstance(g, h5py.File):
        return f"in file {g.filename}"


def make_group(fn_or_obj: str,
               mode: str = None,
               key: str = None) -> Union[h5py.File, h5py.Group]:

    if isinstance(fn_or_obj, (str, bytes)):
        if mode is None:
            raise ValueError("Mode must be given if "
                             "str or bytes passed")

        f = h5py.File(fn_or_obj, mode)

    elif isinstance(fn_or_obj, (h5py.File, h5py.Group)):
        if mode is not None:
            raise ValueError("Mode cannot be given if File or "
                             "Group object passed")
        f = fn_or_obj
    else:
        raise TypeError("Invalid type")

    return f if key is None else f.create_group(key)


def access_group(fn_or_obj: str,
                 key: str = None) -> Union[h5py.File, h5py.Group]:
    if isinstance(fn_or_obj, (str, bytes)):
        f = h5py.File(fn_or_obj, 'r')
    elif isinstance(fn_or_obj, (h5py.File, h5py.Group)):
        f = fn_or_obj
    else:
        raise TypeError("Invalid type")

    if key is not None:
        if key not in f.keys():
            raise KeyError(f"Key {key} not present "
                           f"in {_get_name_file_group(f)}")
        return f[key]
    else:
        return f
