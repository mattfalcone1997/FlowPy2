import numpy as np
import warnings
import logging
from typing import Union, Literal
import os

from ..utils import find_stack_level
from ._common import (valid_tag_checks,
                      make_tag,
                      weak_tag_check)
import weakref
import h5py
logger = logging.getLogger(__name__)

class HDF5TagError(KeyError):
    pass


class HDF5TagWarning(UserWarning):
    pass

class _hdf5_cache:
    def __init__(self):
        self._file_obj = dict()

    def _purge_keys(self):
        keys = list(self._file_obj.keys())
        for k in keys:
            if not self._file_obj[k]:
                del self._file_obj[k]

    def get(self, filename, mode, *args, **kwargs):
        self._purge_keys()
        obj = self._file_obj.get(filename, None)

        if obj is None:
            h5_obj = h5py.File(filename,
                               mode,
                               *args,
                               **kwargs)
            
            self._file_obj[filename] = h5_obj
            owner = True
        else:
            if obj.mode == 'r' and mode != 'r':
                raise ValueError("Cannot open reopen an already opened file "
                                    f"'{filename}'in mode 'r' as"
                                    " writeable. Close it first.")
            if mode == 'w':
                warnings.warn(f"A {filename} already open, but reopened "
                              "in mode 'w'. Overwriting. ",
                              category=UserWarning,
                              stacklevel=find_stack_level())
                obj.close()
                return self.get(filename, mode, *args, **kwargs)
            
            h5_obj = obj
            owner = False
            
        return h5_obj, owner
class hdfHandler:
    _file_cache = _hdf5_cache()
    def __init__(self,
                 fn_or_obj: Union[str, bytes,h5py.Group, h5py.File],
                 mode: Literal['r', 'w', 'a', 'r+']=None,
                 key: str= None, 
                 *args,
                 **kwargs):
        
        self.__owner = False
        msg = "Cannot use mode if str or bytes isn't passed"
        if isinstance(fn_or_obj, (str, bytes)):
            if mode is None:
                mode = 'r'

            self._file, self.__owner = self._file_cache.get(fn_or_obj,
                                              mode=mode,
                                              *args,
                                              **kwargs)
            
            base_group = self._file

        elif isinstance(fn_or_obj, h5py.File):
            if mode is not None:
                raise ValueError(msg)
            
            self._file = fn_or_obj
            base_group = self._file

        elif isinstance(fn_or_obj, h5py.Group):
            if mode is not None:
                raise ValueError(msg)
            
            self._file = fn_or_obj.file
            base_group = fn_or_obj

        elif isinstance(fn_or_obj, hdfHandler):
            if mode is not None:
                raise ValueError(msg)
            
            self._file = fn_or_obj._file
            base_group = fn_or_obj._current_group
        else:
            raise TypeError(f"invalid type for {type(self).__name__}")
            

        if key is None:
            self._current_group = base_group
        else:
           self._current_group = base_group.require_group(key)
    
    def __del__(self):
        if self.__owner:
            self._file.close()

    @property
    def attrs(self):
        return self._current_group.attrs
    
    
    @property
    def filename(self):
        return self._file.filename
    
    @property
    def name(self):
        return self._current_group.name
    
    @property
    def groupname(self):
        return os.path.basename(self.name)
    

    def keys(self):
        return self._current_group.keys()
    
    def create_group(self, name: str):       
        return hdfHandler(self._current_group,
                          key = name)
    
    def __getitem__(self, key):
        obj = self._current_group[key]
        if isinstance(obj, h5py.Group):
            return hdfHandler(obj)
        else:
            return obj
    
    def create_dataset(self,*args,**kwargs):
        
        return self._current_group.create_dataset(*args, **kwargs)

    def read_dataset(self,
                     name: str, 
                     index=None):
        
        if index is None:
            index = slice(None)

        dset = self._current_group[name]
        return dset[index]

    def set_type_tag(self, cls: type):
        ref_tag = make_tag(cls)
        self.attrs['type_tag'] = ref_tag

    def validate_tag(self, cls: type, tag_check):

        if tag_check not in valid_tag_checks:
            raise ValueError(f"Invalid tag check type: {tag_check}. "
                            f"Must be in {' '.join(valid_tag_checks)}")

        ref_tag = make_tag(cls)
        try:
            tag = self.attrs['type_tag']
        except KeyError:
            if tag_check != 'nocheck':
                raise HDF5TagError(f"No tag found in file {self.filename} "
                                   f"at {self.groupname}. "
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
    
    def close(self):
        self._file.close()