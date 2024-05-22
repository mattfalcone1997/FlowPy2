import numpy as np
import warnings
import logging
from typing import Union, Literal


from ..utils import find_stack_level
from ._common import (valid_tag_checks,
                      make_tag,
                      weak_tag_check)
import weakref
import tables
logger = logging.getLogger(__name__)

class HDF5TagError(KeyError):
    pass


class HDF5TagWarning(UserWarning):
    pass

class _hdf_attribute_dict:
    def __init__(self, obj: tables.Node) -> None:
        self._obj = weakref.ref(obj)

    def __getitem__(self, key):
        try:
            return self._obj()._v_attrs.__getitem__(key)
        except KeyError:
            raise KeyError(f"No attribute {key}") from None
    
    def __setitem__(self, key, value):
        try:
            return self._obj()._v_attrs.__setitem__(key, value)
        except KeyError:
            raise KeyError(f"No attribute {key}") from None

    def __delitem__(self, key):
        try:
            return self._obj()._v_attrs.__delitem__(key)
        except KeyError:
            raise KeyError(f"No attribute {key}") from None
        
    def keys(self):
        return self._obj()._v_attrs._f_list()

class hdfHandler:
    def __init__(self,
                 fn_or_obj: Union[str, bytes,tables.Group, tables.File],
                 mode: Literal['r', 'w', 'a', 'r+']=None,
                 key: str= None):
        
        self.__owner = False
        msg = "Cannot use mode if str or bytes isn't passed"
        if isinstance(fn_or_obj, (str, bytes)):
            if mode is None:
                mode = 'r'

            self._file = tables.open_file(fn_or_obj,
                                          mode=mode)
            
            base_group = self._file.root
            self.__owner=True

        elif isinstance(fn_or_obj, tables.File):
            if mode is not None:
                raise ValueError(msg)
            
            self._file = fn_or_obj
            base_group = self._file.root

        elif isinstance(fn_or_obj, tables.Group):
            if mode is not None:
                raise ValueError(msg)
            
            self._file = fn_or_obj._v_file
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
           self._current_group = self._dynamic_get_group(base_group, key)
                
    @staticmethod
    def _dynamic_get_group(group: tables.Group,key: str):
        elements = key.split('/')
        for element in elements:
            if element in group:
                group = group._f_get_child(element)
            else:
                group = tables.Group(group, element, new=True)

        return group
    
    def __del__(self):
        if self.__owner:
            self._file.close()

    @property
    def attrs(self):
        return _hdf_attribute_dict(self._current_group)
    
    
    @property
    def filename(self):
        return self._file.filename
    
    @property
    def groupname(self):
        return self._current_group._v_name
    
    @property
    def full_groupname(self):
        return self._current_group._v_pathname
    
    def keys(self):
        return [x._v_name for x in self._current_group._f_list_nodes()]
    
    def create_group(self, name: str):       
        return hdfHandler(self._current_group,
                          key = name)
    
    def __getitem__(self, key):
        obj = getattr(self._current_group, key)
        if isinstance(obj, tables.Group):
            return hdfHandler(self._current_group[key])
        else:
            return obj
    
    def create_dataset(self,
                       name: str,
                       data: np.ndarray,
                       compression: str=None,
                       compress_level: int=9):
        
        if compression is None:
            self._file.create_array(self._current_group,
                                    name,
                                    data)
            
        else:
            filters=tables.Filters(complevel=compress_level,
                                   complib=compression)
            
            self._file.create_carray(self._current_group,
                                     name=name,
                                     filters=filters,
                                     obj=data)

    def read_dataset(self,
                     name: str, 
                     index=None):
        
        if index is None:
            index = slice(None)

        return getattr(self._current_group, name)[index]

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