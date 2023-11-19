"""
This module contains classes which handles the 
data arrays and their manipulation and all
io processing on these array to hdf5, netCDF etc.
Allows abstraction for cupy arrays
"""
import numpy as np
import numbers
import logging

from abc import ABC, abstractmethod
from numpy.lib.mixins import NDArrayOperatorsMixin
from typing import Iterable, Type, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    _cupy_avail = cp.is_available()
    logger.debug(f"cupy availble: Device {cp.cuda.Device().attributes}")

except ImportError:
    _cupy_avail = False
    logger.debug("cupy unavailble")


# _GROUP_NP_FUNCS = {}

class GroupArray(NDArrayOperatorsMixin):

    _NP_FUNCS = {}
    _options = {}

    _array_class = None
    _array_creator = None

    _HANDLED_TYPES = (numbers.Number,)
    _ALLOWED_METHODS = ('__call__')
    _NOT_ALLOWED_UFUNCS = ()
    _NOT_ALLOWED_KWARGS = ('axis', 'out', 'axes')

    @classmethod
    def implements(cls, np_func):
        def decorator(func):
            logger.debug(
                f"Registering {np_func.__name__} with {func.__name__}")
            cls._NP_FUNCS[np_func] = func
            return func
        return decorator

    @classmethod
    def register(cls, array_cls: type, array_creator: Callable):
        def _group_cls(f_cls):
            logger.debug(f_cls, cls, array_cls)

            class array(f_cls):
                _array_class = array_cls
                _array_creator = array_creator

            cls._options[array_cls] = array
            return array
        return _group_cls

    @classmethod
    def get(cls, array_type: type):
        logger.debug(f"array_type: {array_type}. "
                     f"Available type: {cls._options}")
        if array_type not in cls._options:
            raise TypeError("Type not valid")

        return cls._options[array_type]

    def __init__(self, data: Iterable[np.ndarray], dtype=None, copy=False):

        logger.debug(f"Group array with {self._array_class} "
                     f"and {self._array_creator}")

        self._data = np.array([self._array_creator(
                              d.astype(dtype=dtype, copy=copy), copy=False) for d in data],
                              dtype=object,
                              copy=False)

    @property
    def array_type(self):
        return self._array_class

    @property
    def dtype(self):
        return self._data[0].dtype

    @property
    def data(self):
        return self._data

    @abstractmethod
    def asnumpy(self):
        pass

    def __getitem__(self, key):
        data = self._data.__getitem__(key)

        if data.dtype == object:
            logger.debug("getitem returned array with object dtype")
            return self.__class__(data)
        else:
            logger.debug("getitem returned an element")
            return data

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)

    def __str__(self):
        return self._data.__str__() + self._data[0].dtype.type.__name__

    def __setitem__(self, key, values):
        if isinstance(key, int):
            if not isinstance(values, self._array_class):
                raise TypeError("Element can only be "
                                f"set with {self._array_class}")
        else:
            if not hasattr(values, '__iter__'):
                raise ValueError("Values must be iterable")

            if not all(isinstance(value, self._array_class) for value in values):
                raise TypeError("All elements must be arrays")

        self._data.__setitem__(key, values)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        for x in inputs:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            types = self._HANDLED_TYPES + (self.__class__, self.array_type)

            if not isinstance(x, types):
                return NotImplemented

        actual_inputs = [inp._data
                         if isinstance(inp, self.__class__) else inp
                         for inp in inputs]

        logger.debug("Actual input types: %s" %
                     ([type(inp) for inp in actual_inputs]))
        logger.debug("ufunc: %s %s. Available type: %s" %
                     (ufunc.__name__, method, ufunc.types))

        func = getattr(ufunc, method)
        try:
            data = func(*actual_inputs, **kwargs)
        except TypeError as e:
            if ufunc.nargs == 2:
                logger.debug("Actual input types for exception: %s" %
                             ([inp.dtype for inp in actual_inputs[0]]))
                data = [func(inp) for inp in actual_inputs[0]]
            else:
                raise TypeError(e) from None

        return self.__class__(data)

    def __array_function__(self, func, types, args, kwargs):
        if func not in self._NP_FUNCS:
            logger.debug(f"{func.__name__} not implemented. "
                         "Implemented funcs: {_GROUP_NP_FUNCS.keys()}")
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects

        return self._NP_FUNCS[func](*args, **kwargs)

    def concat(self, other_group):

        if type(self) != type(other_group):
            raise TypeError("Merging groups must be the same type")

        self._data = np.concatenate([self._data,
                                    other_group._data],
                                    axis=0,
                                    dtype=object)

    def copy(self):
        cls = self.__class__
        return cls(self._data.copy())

    def __deepcopy__(self, memo):
        return self.copy()


@GroupArray.implements(np.allclose)
def all_close(array1, array2, *args, **kwargs):
    for val1, val2 in zip(array1, array2):
        if not np.allclose(val1, val2, *args, **kwargs):
            return False

    return True


@GroupArray.implements(np.array_equal)
def array_equal(array1, array2, *args, **kwargs):
    for val1, val2 in zip(array1, array2):
        if not np.array_equal(val1, val2, *args, **kwargs):
            return False

    return True


def array_type(array_cls, array_creator):
    def class_dec(cls):
        class array(cls):
            _array_class = array_cls
            _array_creator = array_creator

        return array
    return class_dec


@GroupArray.register(np.ndarray, np.array)
class npGroupArray(GroupArray):
    @contextmanager
    def asnumpy(self):
        return self


if _cupy_avail:
    @GroupArray.register(cp.ndarray, cp.array)
    class cpGroupArray(GroupArray):
        @contextmanager
        def asnumpy(self):
            return npGroupArray([cp.asnumpy(d) for d in self])
