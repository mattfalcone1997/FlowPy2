import numpy as np

from collections.abc import Mapping, Callable, Sequence
from typing import Any
from abc import ABC, abstractmethod
from .errors import InitialisationError


class DataAccessor(ABC):
    @abstractmethod
    def get(self, output_da=True, **kwargs):
        pass

    @abstractmethod
    def set(self, data, *kwargs):
        pass


class HomogeneousDA(DataAccessor):
    def __init__(self, data: np.ndarray, accessors: Mapping[Any, Callable], dtype: str | np.dtype = None, copy=False) -> None:
        """
            Data accessor class for data with same shape


        Parameters
        ----------
        data : np.ndarray
            Data for the accessor
        accessors : Mapping[Any, Callable]
            Dictionary from axis name to function which indexes axis
        dtype : str | np.dtype, optional
            Data type of incoming data, by defaults to existing datatype
        copy : bool, optional
            Create a copy or view, by default False

        Raises
        ------
        InitialisationError
            Dimensions of data and accessors must match
        InitialisationError
            The values of the accessor must be callable
        """

        if len(accessors) > data.ndim:
            raise InitialisationError("Length of accessors greater"
                                      " than dimensions of data")
        for v in accessors.values():
            if not callable(v):
                raise InitialisationError(
                    "Accessor mapping values must be callable")
        if dtype is None:
            dtype = data.dtype

        self._data = data.astype(dtype, copy=copy)
        self._accessors = dict(accessors)

    @property
    def dtype(self) -> np.dtype:
        """

        Returns
        -------
        np.dtype
            dtype of data
        """
        return self._data.dtype

    @property
    def shape(self) -> tuple[int]:
        """
        Returns
        -------
        tuple[int]
            shape of the data
        """
        return self._data.shape

    @property
    def ndim(self) -> int:
        """_summary_

        Returns
        -------
        int
            Number of dimensions
        """
        return self._data.ndim

    @property
    def size(self) -> int:
        """

        Returns
        -------
        int
            size of data
        """
        return self._data.size

    @property
    def data(self) -> np.ndarray:
        """
        Returns copy of the data

        """
        return self._data.copy()

    def get(self, output_da=True, **kwargs):
        """
        Returns DA or array based on keys provided

        Parameters
        ----------
        output_da : bool, optional
            Whether to output a new DA if possible, by default True

        Returns
        -------
        _type_
            _description_
        """
        indexer = self._create_indexer(kwargs)

        data = self._data[indexer]

        if isinstance(data, np.ndarray) and output_da:
            return self._create_da_from_get(data, indexer)
        else:
            return data

    def _create_indexer(self, accessor: dict):
        """
        Creates indexer for numpy array based on input dictionary

        Parameters
        ----------
        accessor : dict
            _description_

        Returns
        -------
        Iterable[int]
            _description_

        Raises
        ------
        DAAccessError
            _description_
        DAAccessError
            _description_
        """
        if not all(k in self._accessors for k in accessor):
            raise KeyError("Keys not found in DataAccessor")

        indexer = (slice(None))*self.ndim

        for i, (k, accessor) in enumerate(self._accessors.items()):
            if k in accessor:
                indexer[i] = accessor(accessor[k])

        indexer = self._conform_indexer(indexer)

        if not self._verify_array_indexer(indexer):
            raise DAAccessError(
                "Invalid indexer produced by accessors and keys")

        return indexer

    def _conform_indexer(self, indexer: tuple):
        indexer_copy = indexer.copy()
        for i, index in enumerate(indexer):
            if isinstance(index, slice):
                if index.stop == index.start+1:
                    indexer_copy[i] = index.start

        return indexer_copy

    def _verify_array_indexer(self, indices):
        for index in indices:
            if isinstance(index, (int, slice)):
                return True
            if isinstance(index, list):
                if all(isinstance(i, int) for i in index):
                    return True

        return False

    def _create_da_from_get(self, reduced_data, indexer):

        accessor = self._accessors.copy()
        for index, key in zip(indexer, accessor):
            if isinstance(index, int):
                del accessor[key]

        return self.__class__(reduced_data, accessor)

    def set(self, data, **kwargs):

        indexer = self._create_indexer(kwargs)

        if isinstance(data, np.ndarray):
            if self._data[indexer].shape != data.shape:
                raise DAAccessError("Input data does not conform"
                                    " to accessed shape")
            self._data[indexer] = data.astype(self.dtype)
        else:
            self._data[indexer] = data

    def remove(self, **kwargs):
        inverse_index = self._create_inverse_index(kwargs)

        accessor = self._accessors.copy()
        for index, key in zip(inverse_index, accessor):
            if isinstance(index, int):
                del accessor[key]

        self._data = self._data[inverse_index]
        self._accessors = accessor

    def _create_inverse_index(self, kwargs: dict):

        indexer = self._create_indexer(kwargs)
        full_index = tuple(list(range(l)) for l in self._data.shape)
        for i, index in enumerate(indexer):
            if isinstance(index, slice):
                stop = self.data.shape[i] if index.stop is None else index.stop
                start = 0 if index.start is None else index.start
                step = 1 if index.step is None else index.step

                indexer[i] = list(range(start, stop, step))

            elif isinstance(index, int):
                indexer[i] = [index]

        reverse_index = []
        for full, index in zip(full_index, indexer):
            reverse = list(set(index).difference(full))
            reverse_index.append(reverse)

        return reverse_index


class InHomogeneousDA(DataAccessor):
    def __init__(self, data: Sequence[np.ndarray],
                 accessor: Mapping[Any, Callable],
                 dtype: str | np.dtype = None,
                 copy=False):

        if not all(isinstance(d, np.ndarray) for d in data):
            raise InitialisationError("All elements of data must be "
                                      "numpy arrys")
        if not callable(accessor):
            raise InitialisationError("Accessor must be callable")

        self._data = np.array([d.astype(dtype, copy=copy) for d in data],
                              dtype=object)
        self._accessor = dict(accessor)

    def get(self, **kwargs):
        for k, v in kwargs:
            index = self._accessor[k](v)

        if not self._verify_array_indexer(index):
            raise DAAccessError(
                "Invalid indexer produced by accessors and keys")

        data = self._data[index]
        if data.dtype == object:
            return self._create_da_from_get(data, index)

    def _create_da_from_get(self, reduced_data, indexer):
        return self.__class__(reduced_data, indexer)

    def _verify_array_indexer(self, index):
        if isinstance(index, (int, slice)):
            return True
        if isinstance(index, list):
            if all(isinstance(i, int) for i in index):
                return True

        return False

    def set(self, data, **kwargs):
        for k, v in kwargs:
            index = self._accessor[k](v)

        self._data[index] = data

    def remove(self, **kwargs):
        inverse_index = self._create_inverse_index(kwargs)
        self._data = self._data[inverse_index]

    def _create_inverse_index(self, kwargs: dict):
        for k, v in kwargs:
            index = self._accessor[k](v)

        if not hasattr(index, '__iter__'):
            index = [index]

        full_index = tuple(range(self._data.size))
        return list(set(index).difference(full_index))

    def __iter__(self):
        for element in self._data:
            yield element

    def __len__(self):
        return len(self._data)

    def __eq__(self, otherDA):
        if len(self) != len(otherDA):
            return False
        for a, b in zip(self, otherDA):
            if not np.allclose(a, b):
                return False

        return True
