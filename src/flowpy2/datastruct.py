import numpy as np
import numbers
import copy
import logging

from typing import Iterable, Union
from numpy.lib.mixins import NDArrayOperatorsMixin
from .arrays import CommonArrayExtensions, array_backends
from .indexers import CompIndexer
from abc import ABC, abstractmethod

from .io import hdf5
import warnings
from .utils import find_stack_level

logger = logging.getLogger(__name__)


class DataStruct(CommonArrayExtensions):
    _array_attr = '_data'

    def __init__(self,
                 data: Union[Iterable[np.ndarray], dict[str, np.ndarray]],
                 index: Iterable[str] = None,
                 dtype: Union[str, np.dtype] = None,
                 array_backend='numpy',
                 copy=False,
                 **kwargs):

        if isinstance(data, (list, np.ndarray)):
            if index is None:
                TypeError("If data is an array or list, index must be given")

            self._array_ini(data, index, dtype, array_backend, copy)

        elif isinstance(data, dict):
            if index is not None:
                TypeError("If data is a dict, index must not be given")

            self._dict_ini(data, dtype, array_backend, copy)

        else:
            raise TypeError("No valid initialisation found for "
                            f"type {type(data).__name__}")

        self._process_extra_args(**kwargs)

    @classmethod
    def from_hdf(cls, fn_or_obj, key=None, tag_check=None):
        g = hdf5.access_group(fn_or_obj, key)
        if tag_check is None:
            tag_check = 'strict'

        hdf5.validate_tag(cls, g, tag_check)

        index = CompIndexer.from_hdf(g, "index")
        d = g['data']
        data_array = d['data'][:]
        shapes = d['shapes']
        ndims = d['ndim'][:]

        start = 0
        stop = 0

        arr_stop = 0
        arr_start = 0

        data = []

        for ndim in ndims:
            stop += ndim
            shape = shapes[start:stop]
            arr_stop += np.prod(shape)
            item = data_array[arr_start:arr_stop].reshape(shape)
            data.append(item)
            start += ndim
            arr_start += np.prod(shape)

        array_backend = g.attrs['array_backend']

        return DataStruct(data, index=index, array_backend=array_backend)

    def to_hdf(self, fn_or_obj, mode=None, key=None):
        g = hdf5.make_group(fn_or_obj, mode, key)
        hdf5.set_type_tag(type(self), g)

        index_array = self._index.to_hdf(g, "index")

        d = g.create_group("data")
        data = np.concatenate([d.flatten() for d in self._data], axis=0)
        d.create_dataset("data", data=data)
        d.create_dataset("ndim", data=np.array([d.ndim for d in self._data]))

        shapes = np.concatenate([d.shape for d in self._data], axis=0)
        d.create_dataset("shapes", data=shapes)

        g.attrs['array_backend'] = self._array_backend

        if isinstance(fn_or_obj, hdf5.H5_Group_File):
            return g
        else:
            g.file.close()

    @property
    def index(self):
        return self._index

    def _array_ini(self,
                   array: Iterable[np.ndarray],
                   index: Iterable[str],
                   dtype: Union[str, np.dtype] = None,
                   array_backend='numpy',
                   copy=False):

        if len(array) != len(index):
            raise ValueError("Index and array must be the same length")
        self._index = CompIndexer(index)

        creator = array_backends.get_creator(array_backend)

        self._array_backend = array_backend

        array = [creator(arr, dtype=dtype, copy=copy) for arr in array]
        self._data = np.array(array, dtype=object)

    def _dict_ini(self,
                  dict_array: dict,
                  dtype: Union[str, np.dtype] = None,
                  array_backend='numpy',
                  copy=False):

        index = CompIndexer(dict_array.keys())
        array = list(dict_array.values())

        self._array_ini(array, index, dtype, array_backend, copy)

    def to_dict(self):
        return dict(zip(self._index, self._data))

    def __iter__(self):
        for key, val in zip(self._index, self._data):
            yield (key, val)

    def get(self, key):
        index = self._index.get(key)
        if isinstance(index, int):
            return self._data[index]

        elif isinstance(index, (slice, list)):
            data = self._data[index]
            new_index = self._index[index]
            return self._construct_dstruct(data, new_index)

        else:
            raise NotImplementedError("Loop fall through")

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, values):
        try:
            index = self._index.get(key)
            self._data[index] = values
        except KeyError:
            raise KeyError("Key must be present for setitem") from None

    def __delitem__(self, key):
        self.remove(key)

    def _postprocess_array_ufunc(self, data):
        return self._construct_dstruct(data, self.index)

    def _validate_inputs(self, inputs):
        super()._validate_inputs(inputs)

        for x in inputs:
            if isinstance(x, self.__class__):
                if x.index != self.index:
                    raise ValueError("Input of class {self.__class__.__name__}"
                                     " does not have matching inputs")

    def equals(self, other_datastruct):
        return np.allclose(self, other_datastruct)

    def __eq__(self, other_datastruct):
        return self.equals(other_datastruct)

    def __ne__(self, other_datastruct):
        return not self.equals(other_datastruct)

    def __len__(self):
        return len(self._data)

    def copy(self):
        cls = self.__class__
        return cls(copy.deepcopy(self.to_dict()))

    def __deepcopy__(self, memo):
        return self.copy()

    def __contains__(self, key):
        return key in self._index

    def _construct_dstruct(self, array, index):
        kwargs = self._get_internal_args(array, index)

        return self.__class__(**kwargs)

    def _get_internal_args(self, array, index):
        kwargs = {'data': array,
                  'index': index,
                  'array_backend': self._array_backend}

        return kwargs

    def concat(self, datastruct):
        if type(datastruct) != self.__class__:
            raise TypeError(f"Merging {self.__class__.__name}"
                            " must be of the same type")
        self._index.extend(datastruct._index)
        self._data = np.concatenate([self._data, datastruct._data], axis=0)

    def remove(self, keys):
        indexer = self._index.get_other(keys)

        self._data = self._data[indexer].copy()
        self._index.remove(keys)

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__str__()


@DataStruct.implements(np.allclose)
def allclose(dstruct1: DataStruct, dstruct2: DataStruct, *args, **kwargs):
    if dstruct1.index != dstruct2.index:
        logger.debug("Indices do not match")
        return False

    for d1, d2 in zip(dstruct1._data, dstruct2._data):
        if not np.allclose(d1, d2, *args, **kwargs):
            logger.debug("data do not match")
            return False

    return True


@DataStruct.implements(np.array_equal)
def array_equal(dstruct1: DataStruct, dstruct2: DataStruct, *args, **kwargs):
    if dstruct1.index != dstruct2.index:
        return False

    for d1, d2 in zip(dstruct1._data, dstruct2._data):
        if not np.array_equal(d1, d2, *args, **kwargs):
            return False

    return True
