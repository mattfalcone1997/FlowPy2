from __future__ import annotations

import numpy as np
import copy
import logging
from numpy.typing import DTypeLike
from numpy.lib.mixins import NDArrayOperatorsMixin
from numbers import Number
from typing import Iterable, Union, List, Optional
from abc import ABC, abstractmethod
import warnings

from .utils import find_stack_level
from .io import hdf5, netcdf

logger = logging.getLogger(__name__)


class IndexInitialisationError(Exception):
    pass


class IndexBase(ABC):
    @abstractmethod
    def key_present(self, key):
        pass

    def __getitem__(self, index: int):
        if isinstance(index, int):
            return self._index[index]
        elif isinstance(index, slice):
            return self.__class__(self._index.__getitem__(index))
        elif isinstance(index, list):
            return self.__class__(np.array(self._index)[index])

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return "%s(%s)" % (name, self._index)

    def __str__(self) -> str:
        name = self.__class__.__name__
        return "%s(%s)" % (name, self._index)

    def __eq__(self, other_index) -> bool:
        if type(other_index) != self.__class__:
            return False
        return all(x == y for x, y in zip(self, other_index))

    def __len__(self):
        return len(self._index)

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        return self.__class__(self._index)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        obj = self.__class__(d)
        self.__dict__ = obj.__dict__


class CompIndexer(IndexBase):
    def __init__(self, index: Iterable[str]):

        self._index = self._verify_index(index)
        self.__accessor_update = True
        self.__update_accessor_dict()

    def __update_accessor_dict(self):
        if self.__accessor_update:
            self.__accessor_dict = dict(zip(self._index,
                                        range(len(self._index))))
        self.__accessor_update = False

    def _verify_index(self, index: Iterable[str]) -> List[str]:
        if isinstance(index, str):
            raise TypeError("Container for input index cannot "
                            f"be str for {self.__class__.__name}")

        for ind in index:
            if not isinstance(ind, str):
                raise IndexInitialisationError("Inputs must be of type str "
                                               f"not {type(ind)}")

        if len(index) != len(set(index)):
            raise ValueError("Indices cannot be repeated")

        return copy.deepcopy(list(index))

    def key_present(self, key):
        if isinstance(key, str):
            return key in self._index
        elif hasattr(key, '__iter__'):
            return all(k in self._index for k in key)
        else:
            raise TypeError("Invalid key")

    def get(self, key) -> int:
        self.__update_accessor_dict()
        if isinstance(key, str):
            try:
                return self.__accessor_dict[key]
            except KeyError as e:
                raise KeyError(f"Component {key} not in indexer") from None

        elif isinstance(key, list):
            try:
                return [self.__accessor_dict[k] for k in key]
            except KeyError:
                raise KeyError("Some of the components in "
                               f"{key} not present") from None

        elif isinstance(key, slice):
            try:
                start = 0 if key.start is None else self.__accessor_dict[key.start]
            except KeyError:
                raise KeyError("Start of component slice"
                               f"{key.start} not present") from None

            try:
                stop = len(self._index) if key.stop is None\
                    else self.__accessor_dict[key.stop]+1
            except KeyError:
                raise KeyError("End of component slice"
                               f"{key.stop} not present") from None

            if key.step is not None:
                raise ValueError("step not allowed for slice indexing")

            if start >= stop:
                raise ValueError("Slice start must be ordered "
                                 "before slice stop")

            return slice(start, stop)

        else:
            raise KeyError("Invalid key: must be str, list or slice")

    def get_other(self, key):
        index = self.get(key)
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            index = range(index.start, index.stop)

        indexer = list(range(len(self._index)))
        for i in index:
            indexer.remove(i)
        return indexer

    def append(self, key: str):
        if not isinstance(key, str):
            raise TypeError("key must be of type str")

        if key in self._index:
            raise ValueError("key already found in index")

        self._index.append(key)
        self.__accessor_update = True

    def extend(self, index):
        if not isinstance(index, CompIndexer):
            index = CompIndexer(index)

        if any(key in self._index for key in index):
            raise ValueError("key already in index")

        self._index.extend(index._index)

        self.__accessor_update = True

    def remove(self, keys: Union[Iterable[str], str, slice]):
        if isinstance(keys, slice):
            index = self.get(keys)
            keys = self._index[index]
        elif isinstance(keys, str):
            keys = [keys]

        for key in keys:
            self._index.remove(key)

        self.__accessor_update = True

    def replace_keys(self, key_pairs):
        for k, v in key_pairs.items():
            try:
                index = self._index.index(k)
            except ValueError:
                raise KeyError(f"Key {k} not found") from None

            if not isinstance(v, str):
                raise TypeError("Replacement key must be of type str")

            if self.key_present(v):
                raise KeyError(f"Key {v} already present")
            self._index[index] = v
            self.__accessor_update = True

    def to_array(self):
        return np.array(self._index, dtype=np.string_)

    def to_hdf(self, h5_obj: hdf5.H5_Group_File, key: str):
        h5_obj.create_dataset(key,
                              data=np.array(self._index,
                                            dtype=np.string_))

    @classmethod
    def from_hdf(cls, h5_obj: hdf5.H5_Group_File, key: str):
        return cls([key.decode('utf-8') for key in h5_obj[key]])

    # def create_iterable_key(self, keys):
    #     if isinstance(keys, str):
    #         return [keys]
    #     if isinstance(keys, slice):
    #         start = 0 if keys.start is None else self.__accessor_dict(
    #             keys.start)
    #         stop = len(self._index) if keys.stop is None else self.__accessor_dict(
    #             keys.stop)
    #         step = 1 if keys.step is None else self.__accessor_dict(keys.step)
    #         return list(range(start, stop, step))
    #     elif all(isinstance(key, str) for key in keys):
    #         return True
    #     else:
    #         raise TypeError("key must be ")


class DtypeTruncationWarning(UserWarning):
    pass


class RoundTruncationWarning(UserWarning):
    pass


class TimeIndexer(IndexBase, NDArrayOperatorsMixin):
    _NP_FUNCS = {}
    _options = {}

    _array_class = None
    _array_creator = None

    _HANDLED_TYPES = (Number,)
    _ALLOWED_METHODS = ('__call__')
    _NOT_ALLOWED_UFUNCS = ()
    _NOT_ALLOWED_KWARGS = ('axis', 'out', 'axes')

    def __init__(self, index: Iterable[Number], decimals_round: Optional[int] = None, dtype: DTypeLike = 'f8'):

        self._decimals = decimals_round
        self._dtype = np.dtype(dtype).type

        self._index = self._verify_index(index)
        self.__accessor_update = True
        self.__update_accessor_dict()

    @property
    def decimals(self):
        return self._decimals

    def __update_accessor_dict(self):
        if self.__accessor_update:
            self.__accessor_dict = dict(zip(self._index,
                                        range(len(self._index))))
        self.__accessor_update = False

    def _verify_index(self, index: Iterable[str]) -> List[str]:

        if isinstance(index, Number):
            index = [index]

        for ind in index:
            if not isinstance(ind, Number):
                raise IndexInitialisationError("Inputs must be numbers")

        if len(index) != len(set(index)):
            raise ValueError("Indices cannot be repeated")

        index = np.array(index)

        diff = np.diff(index)
        if any(diff < 0):
            raise ValueError("Coordinates must be in ascending order")

        index_array = np.array(index)
        index_array_dtype = index_array.astype(self._dtype, copy=True)
        if not np.array_equal(index_array, index_array_dtype):
            warnings.warn("Possible truncation on type conversion detected",
                          category=DtypeTruncationWarning,
                          stacklevel=find_stack_level())

        if self._decimals is not None:
            index_array_round = np.around(index_array, decimals=self._decimals)
        else:
            index_array_round = index_array_dtype

        if not np.array_equal(index_array_round, index_array_dtype):
            warnings.warn("Possible truncation on rounding detected",
                          category=RoundTruncationWarning,
                          stacklevel=find_stack_level())

        return index_array_round

    def key_present(self, key):
        if isinstance(key, Number):
            return key in self._index
        elif hasattr(key, '__iter__'):
            return all(k in self._index for k in key)
        else:
            raise TypeError("Invalid key")

    def _check_key(self, key):
        if self._decimals is not None:
            key = np.round(key, self._decimals)
        return self._dtype(key)

    def get(self, key) -> int:
        self.__update_accessor_dict()

        if isinstance(key, Number):
            key = self._check_key(key)
            try:
                return self.__accessor_dict[key]
            except KeyError:
                raise KeyError(f"Time {key} not present in "
                               "indexer") from None

        elif isinstance(key, list):
            try:
                return [self.__accessor_dict[self._check_key(k)] for k in key]
            except KeyError:
                raise KeyError("Some of the times in "
                               f"{key} not present") from None

        elif isinstance(key, slice):
            if key.start is None:
                start = 0
            else:
                try:
                    start = self.__accessor_dict[self._check_key(key.start)]
                except KeyError:
                    raise KeyError("Start of time slice"
                                   f"{key.start} not present") from None

            if key.stop is None:
                stop = len(self._index)
            else:
                try:
                    stop = self.__accessor_dict[self._check_key(key.stop)] + 1
                except KeyError:
                    raise KeyError("End of time slice"
                                   f"{key.stop} not present") from None

            if key.step is not None:
                raise ValueError("step not allowed for slice indexing")

            if start >= stop:
                raise ValueError("Slice start must be ordered "
                                 "before slice stop")

            return slice(start, stop)
        else:
            raise KeyError("Invalid key: must be Number, list or slice")

    def get_other(self, key):
        index = self.get(key)

        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            index = range(index.start, index.stop)

        indexer = list(range(len(self._index)))
        for i in index:
            indexer.remove(i)
        return indexer

    def append(self, key: Number):
        if not isinstance(key, Number):
            raise TypeError("key must be of type str")

        if key in self._index:
            raise ValueError("key already found in index")

        index = list(self._index)
        index.append(key)

        self._index = self._verify_index(sorted(index))
        self.__accessor_update = True

    def extend(self, index: Iterable):
        index = sorted(list(self._index) + list(index))

        self._index = self._verify_index(index)
        self.__accessor_update = True

    @staticmethod
    def concat_indexers(*indices):
        # check validity of indices

        indices_list = [np.array(index) for index in indices]

        for index in indices_list[1:]:
            if np.equal(index, indices_list[0]).any():
                raise ValueError("Elements overlap")

        combined_index = np.concatenate(indices_list)
        starts = [0]*len(indices)
        for i in range(1, len(indices)):
            starts[i] = starts[-1] + len(indices[i-1])

        stops = [len(indices[0])]*len(indices)
        for i in range(1, len(indices)):
            stops[i] = stops[-1] + len(indices[i])

        sorted_indices = np.argsort(np.argsort(combined_index))
        out = tuple(list(sorted_indices[start:stop])
                    for start, stop in zip(starts, stops))

        return out

    def to_hdf(self, h5_obj: hdf5.H5_Group_File, key: str):
        self.__update_accessor_dict()

        d = h5_obj.create_dataset(key,
                                  data=self._index)
        if self._decimals is not None:
            d.attrs['decimals'] = self._decimals

    @classmethod
    def from_hdf(cls, h5_obj: hdf5.H5_Group_File, key: str):
        times = h5_obj[key][:]
        if 'decimals' in h5_obj.keys():
            decimals = h5_obj.attrs['decimals']
        else:
            decimals = None
        return cls(times, decimals)

    def remove(self, keys: Union[Iterable[str], Number, slice]):
        index = self.get_other(keys)

        self._index = self._verify_index(self._index[index])
        self.__accessor_update = True

    def to_netcdf(self, group):
        if not netcdf.HAVE_NETCDF4:
            raise ModuleNotFoundError("netCDF cannot be used")

        group.createDimension("time", self._index.size)
        dtype = self._index.dtype.kind + str(self._index.dtype.itemsize)
        var = group.createVariable("time", dtype, ("time",))
        var[:] = self._index
        var.units = "seconds"

        if self._decimals is not None:
            var.decimals = self._decimals

    @classmethod
    def from_netcdf(cls, group):
        if not netcdf.HAVE_NETCDF4:
            raise ModuleNotFoundError("netCDF cannot be used")

        if 'time' not in group.variables:
            return None

        time = group.variables['time'][:]
        if hasattr(group.variables['time'], 'decimals'):
            decimals = group.variables['time'].decimals
        else:
            decimals = None

        return cls(time, decimals)

    @classmethod
    def implements(cls, np_func):
        def decorator(func):
            logger.debug(
                f"Registering {np_func.__name__} with {func.__name__}")
            cls._NP_FUNCS[np_func] = func
            return func
        return decorator

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        for x in inputs:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            types = self._HANDLED_TYPES + (self.__class__, type(self._index))

            if not isinstance(x, types):
                return NotImplemented

        actual_inputs = [inp._index
                         if isinstance(inp, self.__class__) else inp
                         for inp in inputs]

        logger.debug("Actual input types: %s" %
                     ([type(inp) for inp in actual_inputs]))
        logger.debug("ufunc: %s %s. Available type: %s" %
                     (ufunc.__name__, method, ufunc.types))

        func = getattr(ufunc, method)
        data = func(*actual_inputs, **kwargs)

        decimals = self._decimals
        dtype = self._dtype

        return self.__class__(data, decimals_round=decimals, dtype=dtype)

    def __array_function__(self, func, types, args, kwargs):
        if func not in self._NP_FUNCS:
            logger.debug(f"{func.__name__} not implemented. "
                         "Implemented funcs: {_GROUP_NP_FUNCS.keys()}")
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects

        return self._NP_FUNCS[func](*args, **kwargs)
