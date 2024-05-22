import numpy as np
import numbers
import logging

import flowpy2 as fp
from .io import hdf5
from abc import ABC, abstractmethod, abstractclassmethod
from numpy.lib.mixins import NDArrayOperatorsMixin
from typing import Iterable, Type, Callable

from matplotlib.rcsetup import validate_string
logger = logging.getLogger(__name__)

try:
    import cupy as cp
    HAVE_CUPY = cp.is_available()
    logger.debug(f"cupy availble: Device {cp.cuda.Device().attributes}")

except ImportError:
    HAVE_CUPY = False
    logger.debug("cupy unavailble")

_rc_params = {'backend': 'numpy'}
_rc_validators = {'backend': validate_string}

class ArrayBackends:
    def __init__(self):
        self._array_types = {}
        self._array_creators = {}
        self._np_casters = {}

    def set(self, key: str, array_type: Type, array_creator: Callable,
            np_caster: Callable):
        """ 
        each element requires a tuple with the array type, 
        creator and a method to cast to numpy except numpy
        """

        if key in self._array_types:
            raise ValueError(f"Key {key} already present")

        test = [1., 2., 3.]
        creator_test = array_creator(test)

        if type(creator_test) != array_type:
            raise TypeError("creator did not product type "
                            f"{array_type.__name__}")

        if type(np_caster(creator_test)) != np.ndarray:
            raise TypeError("Numpy caster did not produce numpy array")

        self._array_types[key] = array_type
        self._array_creators[key] = array_creator
        self._np_casters[key] = np_caster

    def get_type(self, key: str):
        return self._array_types[key]

    def get_creator(self, key: str):
        return self._array_creators[key]

    def get_caster(self, key: str):
        return self._np_casters[key]


array_backends = ArrayBackends()
array_backends.set('numpy', np.ndarray, np.array, lambda x: x)
if HAVE_CUPY:
    array_backends.set('cupy', cp.ndarray, cp.array, cp.asnumpy)


class CommonArrayExtensions(NDArrayOperatorsMixin, ABC):
    _NP_FUNCS = {}

    _HANDLED_TYPES = (numbers.Number,)
    _ALLOWED_METHODS = ('__call__')
    _NOT_ALLOWED_UFUNCS = ()
    _NOT_ALLOWED_KWARGS = ('axis', 'out', 'axes')
    _array_attr = None

    def __new__(cls, *args, **kwargs):
        if cls._array_attr is None:
            raise AttributeError("_array_attr must be set")

        if not isinstance(cls._array_attr, str):
            raise TypeError("_array_attr must be type str")

        logging.debug(f"_array_attr is {cls._array_attr}")

        return super().__new__(cls)

    @classmethod
    def implements(cls, np_func):
        def decorator(func):
            if cls.__name__ not in cls._NP_FUNCS:
                cls._NP_FUNCS[cls.__name__] = {}
                logger.debug(f"Creating dictionary for class {cls.__name__}")

            logger.debug(
                f"Registering {np_func.__name__} with {func.__name__}")
            cls._NP_FUNCS[cls.__name__][np_func.__name__] = func
            return func
        return decorator

    def _preprocess_array_ufunc(self, ufunc, method, *inputs, **kwargs):
        for x in inputs:
            if isinstance(x, CommonArrayExtensions) \
                    and type(x) != self.__class__:
                raise TypeError("Operations on subclasses of "
                                "ArrayExtensionsBase are strictly typed")

        actual_inputs = [getattr(inp, self._array_attr)
                         if isinstance(inp, self.__class__) else inp
                         for inp in inputs]

        return actual_inputs

    @abstractmethod
    def _postprocess_array_ufunc(*args, **kwargs):
        pass

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        for x in inputs:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            types = self._HANDLED_TYPES + (self.__class__, )

            if not isinstance(x, types):
                return NotImplemented

        actual_inputs = self._preprocess_array_ufunc(ufunc, method,
                                                     *inputs, **kwargs)

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

        return self._postprocess_array_ufunc(data)

    def __array_function__(self, func, types, args, kwargs):
        cls_name = self.__class__.__name__
        if cls_name not in self._NP_FUNCS:
            logger.debug(f"{cls_name} is not present in _NP_FUNC dictionary "
                         "not implement calls for this class")
            return NotImplemented

        if func.__name__ not in self._NP_FUNCS[self.__class__.__name__]:
            logger.debug(f"{func.__name__} not implemented for {cls_name}. "
                         f"Implemented funcs: {self._NP_FUNCS[cls_name].keys()}")
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects

        return self._NP_FUNCS[cls_name][func.__name__](*args, **kwargs)

    def _process_extra_args(self, **kwargs):
        pass

    # def _get_internal_args(self, **kwargs):
    #     pass

    @staticmethod
    def _hdf5_read_hook(hdf_obj: hdf5.hdfHandler):
        return {}

    def _hdf5_write_hook(self, hdf_obj: hdf5.hdfHandler):
        pass

    @staticmethod
    def _netcdf_read_hook(netcdf_dataset):
        return {}

    def _netcdf_write_hook(self, netcdf_dataset):
        pass

    @classmethod
    def _create_struct(cls,**kwargs):
        return cls(**kwargs)

    def _init_update_kwargs(self,kwargs: dict, key: str, value):
        if key not in kwargs:
            kwargs[key] = value

    @abstractmethod
    def _init_args_from_kwargs(self,*args,**kwargs):
        pass