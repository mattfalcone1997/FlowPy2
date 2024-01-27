from __future__ import annotations
import numpy as np
from .coords import CoordStruct
from numpy.typing import ArrayLike
from typing import Iterable, Callable, Union, Mapping, List
from numbers import Number
from .indexers import CompIndexer, TimeIndexer
from matplotlib.axes import Axes
from .arrays import ArrayExtensionsBase, array_backends, ArrayBackends
from .flow_type import FlowType

import warnings
import logging
from .utils import find_stack_level

from .io import hdf5, netcdf

logger = logging.getLogger(__name__)


class FlowStructND(ArrayExtensionsBase):
    _array_attr = '_array'

    def __init__(self,
                 coorddata: CoordStruct,
                 array: ArrayLike,
                 comps: Iterable[str],
                 data_layout: Iterable[str],
                 times: Iterable[Number] = None,
                 time_decimals: int = None,
                 array_backend='numpy',
                 dtype: Union[str, type, None] = None,
                 attrs: Mapping = None,
                 copy=False) -> None:

        self._coords = self._set_coords(coorddata, data_layout)
        self._data_layout = tuple(data_layout)

        self._comps = CompIndexer(comps)
        if times is not None:
            times = TimeIndexer(times, time_decimals)
        self._times = times

        if attrs is None:
            attrs = {}

        if dtype is None:
            dtype = array.dtype.type

        self.attrs = dict(attrs)
        self._array = self._set_array(array, array_backend, dtype, copy)
        self._array_backend = array_backend

    def _set_coords(self, coorddata: CoordStruct, data_layout: Iterable[str]):
        if not isinstance(coorddata, CoordStruct):
            raise TypeError("coorddata must be an instance of CoordStruct")

        for d in data_layout:
            if d not in coorddata.index:
                raise ValueError(f"Element {d} in data_layout not "
                                 "in coorddata")

        coorddata = coorddata.copy()

        for k in coorddata.index:
            if k not in data_layout:
                del coorddata[k]

        return coorddata

    def _set_array(self, array: ArrayLike, array_backend: str, dtype: Union[str, type], copy: bool) -> ArrayLike:

        creator = array_backends.get_creator(array_backend)
        if type(array).__module__ != creator.__module__:
            array = creator(array, dtype=dtype)

        array = array.astype(dtype, copy=copy)

        if self._times is None:
            array = array.reshape((1, *array.shape))

        ndim = len(self._data_layout) + 2

        if array.ndim != ndim:
            raise ValueError("array must have dimension "
                             f"{ndim} not {array.ndim}")

        for i, d in enumerate(self._data_layout):
            coordsize = self._coords[d].size
            if array.shape[2+i] != coordsize:
                raise ValueError(f"Shape of dimension {i} of array "
                                 f"is invalid: {array.shape[2+i]} "
                                 f"vs {coordsize}")

        if self._times is not None:
            if len(self._times) != array.shape[0]:
                raise ValueError(f"Shape of dimension 0 of array "
                                 f"is invalid: {array.shape[0]} "
                                 f"vs {len(self._times)}")

        if array.shape[1] != len(self._comps):
            raise ValueError(f"Shape of dimension 1 of array "
                             f"is invalid: {array.shape[0]} "
                             f"vs {len(self._comps)}")

        return array

    def reduce(self, operation: Callable, axis: str):
        coords = self.coords.copy()
        coords.remove(axis)
        data_layout = list(self._data_layout)
        data_layout.remove(axis)

        for a in axis:
            axis_ind = list(self._data_layout).index(axis) + 2
            array = operation(self._array, axis=axis_ind)
        return self._construct_fstruct(coords, array, self.times, self.comps, data_layout)

    @property
    def flow_type(self) -> FlowType:
        return self._coords.flow_type

    @property
    def dtype(self) -> type:
        return self._array.dtype.type

    @property
    def coords(self) -> CoordStruct:
        return self._coords

    @property
    def times(self) -> np.ndarray:
        return np.array(self._times)

    @times.setter
    def times(self, values: np.array):
        if len(values) != len(self._times):
            raise ValueError("Times can only be set if "
                             "the array is the correct length")

        self._times = TimeIndexer(values)

    @property
    def comps(self) -> list[str]:
        return list(self._comps)

    @property
    def shape(self) -> tuple[int]:
        return self._array.shape[2:]

    @property
    def ndim(self) -> int:
        return len(self._coords)

    def get(self, *, time=None,
            comp=None,
            output_fs=True,
            squeeze=True,
            drop_coords=True,
            **coords_kw) -> Union[ArrayLike, FlowStructND]:

        indexer = [slice(None)]*self._array.ndim
        shape = [None]*self._array.ndim

        indexer[0], shape[0], out_fst = self._process_time_index(time)

        indexer[1], shape[1], out_fsc = self._process_comp_index(comp)

        out_fs = out_fsc or out_fst

        if coords_kw:
            coord_dict = {}
            data_layout = []
            for i, d in enumerate(self._data_layout):
                indexer[2+i], shape[2+i] = self._process_coord_index(d,
                                                                     coords_kw)
                if isinstance(indexer[2+i], (slice, list)) or not drop_coords:
                    coord_dict[d] = self._coords[d][indexer[2+i]]
                    data_layout.append(d)

            if drop_coords:
                drop_shapes = [2+i for i in range(len(indexer)-2)
                               if isinstance(indexer[i+2], np.integer)]
                for d in drop_shapes[::-1]:
                    del shape[d]
        else:
            coord_dict = self.coords.to_dict()
            data_layout = self._data_layout
            shape[2:] = self.shape

        array = self._array[tuple(indexer)].reshape(shape)

        if out_fs and output_fs:
            times = self.times[indexer[0]]
            comps = self._comps[indexer[1]]
            coords = self._coords.__class__(self.flow_type, coord_dict)

            return self._construct_fstruct(coords, array, times, comps, data_layout)

        else:
            return array if not squeeze else np.squeeze(array)

    def values(self):
        return self._array

    def _validate_inputs(self, inputs):
        super()._validate_inputs(inputs)
        for x in inputs:
            if isinstance(x, self.__class__):
                if isinstance(x, self.__class__):
                    self._check_fstruct_compat(x, True, True)

    def __array_ufunc__(self,  ufunc, method, *inputs, **kwargs):

        data = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

        return self._construct_fstruct(self.coords, data, self.times, self.comps, self._data_layout)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.get(time=key[0],
                            comp=key[1])
        elif self._times is None:
            return self.get(comp=key)
        else:
            raise KeyError("Invalid key")

    def _construct_fstruct(self, coords, array, times, comps, data_layout):
        args, kwds = self._get_internal_args_kwargs()

        fstruct = self.__class__(coords, array, comps, data_layout,
                                 times, *args, **kwds)

        return self._fstruct_promote(fstruct)

    def _get_internal_args_kwargs(self, **kwargs):
        kwds = dict(time_decimals=self._times.decimals,
                    array_backend=self._array_backend,
                    copy=False)
        kwds.update(kwargs)

        return (), kwds

    def _fstruct_promote(self, fstruct):
        return fstruct

    def _process_time_index(self, time):
        if time is None:
            return slice(None), len(self._times), len(self._times) > 1

        index = self._times.get(time)
        if isinstance(index, (slice, list)):
            out_fs = True
        else:
            index = [index]
            out_fs = False

        shape = self.times[index].size
        return index, shape, out_fs

    def _process_comp_index(self, comp):
        if comp is None:
            return slice(None), len(self._comps), len(self.comps) > 1

        index = self._comps.get(comp)

        if isinstance(index, (slice, list)):
            out_fs = True
        else:
            index = [index]
            out_fs = False

        shape = len(self._comps[index])
        return index, shape, out_fs

    def _process_coord_index(self, key, coords_kw):
        if key not in coords_kw:
            return slice(None), self._coords[key].size

        index = self._coords.coord_index(key, coords_kw[key])

        if isinstance(index, int):
            shape = 1
        else:
            shape = self._coords[key][index].size
        return index, shape

    def _check_fstruct_compat(self, fstruct: FlowStructND,
                              check_times: bool = False,
                              check_comps: bool = False):

        if self.coords != fstruct.coords:
            raise ValueError("Coordinates do not match")

        if self._data_layout != fstruct._data_layout:
            raise ValueError("Data layout do not match")

        if check_times:
            if not np.array_equal(self.times, fstruct.times):
                raise ValueError("Times do not match")

        if check_comps:
            if self.comps != fstruct.comps:
                raise ValueError("Components do not match")

    def concat_comps(self, fstructs: Union[FlowStructND, Iterable[FlowStructND]]) -> FlowStructND:
        if isinstance(fstructs, FlowStructND):
            fstructs = [fstructs]

        new_comps = self._comps.copy()
        arrays = [self._array]
        for fstruct in fstructs:
            self._check_fstruct_compat(fstruct, check_times=True)

            new_comps.extend(fstruct.comps.copy())
            arrays.append(fstruct._array)

        new_data = np.concatenate(arrays, axis=1)
        return self.__class__(self._coords,
                              new_data,
                              new_comps,
                              data_layout=self._data_layout,
                              times=self._times,
                              array_backend=self._array_backend)

    def concat_times(self, fstructs: Union[FlowStructND, Iterable[FlowStructND]]) -> FlowStructND:

        if isinstance(fstructs, FlowStructND):
            fstructs = [fstructs]

        new_time_indexer = self._times.copy()
        time_indexers = [self._times]
        for fstruct in fstructs:
            self._check_fstruct_compat(fstruct, check_comps=True)

            new_time_indexer.extend(fstruct._times)
            time_indexers.append(fstruct._times)

        shape = (len(new_time_indexer), len(self.comps), *self.shape)
        new_data = np.zeros(shape, dtype=self.dtype)

        indexers = self._times.concat_indexers(*time_indexers)

        new_data[indexers[0]] = self._array
        for indexer, fstruct in zip(indexers[1:], fstructs):
            new_data[indexer] = fstruct._array

        return self.__class__(self._coords,
                              new_data,
                              self._comps,
                              data_layout=self._data_layout,
                              times=new_time_indexer,
                              array_backend=self._array_backend)

    def copy(self) -> FlowStructND:
        return self.__class__(self._coords,
                              self._array,
                              self._comps,
                              self._data_layout,
                              self._times,
                              self._times.decimals,
                              self._array_backend,
                              copy=True)

    def __deepcopy__(self, memo):
        return self.copy()

    def Translate(self, **kwargs):
        for k in kwargs:
            if k not in self._data_layout:
                raise ValueError(f"{k} not in {self.__class__.__name__}")
            if not isinstance(kwargs[k], Number):
                raise TypeError("Invalid type")

            self._coords[k] += kwargs[k]

    def to_hdf(self, fn_or_obj: str, mode: str = None, key: str = None, compress=True):

        g = hdf5.make_group(fn_or_obj, mode, key)

        hdf5.set_type_tag(type(self), g)

        compression = 'gzip' if compress else None
        g.create_dataset("array", data=self._array,
                         compression=compression)

        self._comps.to_hdf(g, 'comps')
        self._times.to_hdf(g, 'times')

        self._coords.to_hdf(g, key='coords')

        g.create_dataset("data_layout",
                         data=np.array(self._data_layout, dtype=np.string_))

        g.attrs['array_backend'] = self._array_backend
        for k, v in self.attrs:
            g.attrs[k] = v

        if isinstance(fn_or_obj, hdf5.H5_Group_File):
            return g
        else:
            g.file.close()

    @classmethod
    def from_hdf(cls,
                 fn_or_obj: str,
                 key: str = None,
                 comps: Iterable[str] = None,
                 times: Iterable[Number] = None,
                 tag_check=None) -> FlowStructND:

        g = hdf5.access_group(fn_or_obj, key)
        if tag_check is None:
            tag_check = 'strict'

        hdf5.validate_tag(cls, g, tag_check)

        coords = CoordStruct.from_hdf(g, 'coords')
        array = g['array'][:]
        comps = CompIndexer.from_hdf(g, 'comps')
        times = TimeIndexer.from_hdf(g, 'times')

        attrs = dict(g.attrs)

        data_layout = tuple([key.decode('utf-8')
                            for key in g['data_layout'][:]])
        array_backend = attrs['array_backend']

        del attrs['array_backend']
        del attrs['type_tag']

        if not isinstance(fn_or_obj, hdf5.H5_Group_File):
            g.file.close()

        return FlowStructND(coords,
                            array,
                            comps,
                            data_layout,
                            times,
                            time_decimals=times.decimals,
                            array_backend=array_backend)

    def to_netcdf(self, fn_or_obj: str, mode: str = None, key: str = None, compress=True):

        if not netcdf.HAVE_NETCDF4:
            raise ModuleNotFoundError("Cannot use netcdf")

        g = netcdf.make_dataset(fn_or_obj, mode, key)
        netcdf.set_type_tag(type(self), g)

        g.array_backend = self._array_backend
        self._coords.to_netcdf(g)

        layout = []
        if self._times is not None:
            self._times.to_netcdf(g)
            layout.append('time')
        layout.extend(self._data_layout)

        dtype = self._array.dtype.kind + str(self._array.dtype.itemsize)
        for comp in self.comps:
            var = g.createVariable(comp, dtype, tuple(layout))
            var[:] = self.get(comp=comp).values()
        g.data_layout = self._data_layout
        g.comps = list(self._comps)
        netcdf.close(g)

    @classmethod
    def from_netcdf(cls, fn_or_obj: str,
                    key: str = None,
                    comps: Iterable[str] = None,
                    times: Iterable[Number] = None,
                    tag_check=None) -> FlowStructND:

        g = netcdf.access_dataset(fn_or_obj, key)
        if tag_check is None:
            tag_check = 'strict'

        netcdf.validate_tag(cls, g, tag_check)

        coords = CoordStruct.from_netcdf(g)
        times = TimeIndexer.from_netcdf(g)
        data_layout = g.data_layout

        comps = list(g.comps)
        array = [g.variables[comp][:] for comp in comps]
        axis = 1 if times is not None else 0

        array = np.stack(array, axis=axis)
        return FlowStructND(coords,
                            array,
                            comps,
                            data_layout,
                            times,
                            time_decimals=times.decimals,
                            array_backend=g.array_backend)

    def _get_plot_data(self, dir_plane, loc, output_dim):
        dim = self.ndim - output_dim
        if isinstance(loc, Number):
            if dim > 1:
                raise TypeError("Number can only be given if "
                                "flowstruct as dimension 2")

            coord = list(self._data_layout)
            for d in dir_plane:
                coord.remove(d)
            loc = {coord[0]: loc}

        if isinstance(loc, Mapping):
            if not all(key in self._data_layout for key in loc):
                raise ValueError("Mapping key must be in "
                                 "coordinates")
            if not all(isinstance(v, Number) for v in loc.values()):
                raise TypeError("Input coordinates for plotting "
                                "must be numbers")

            if any(d in dir_plane for d in loc):
                raise ValueError("Plot planes or directions cannot "
                                 "appear in loc")

        else:
            raise TypeError("Invalid type for location")

        if len(loc) != dim:
            raise ValueError(f"Size of loc should be {dim} not {len(loc)}")

        return loc

    def plot_line(self,
                  line: str,
                  loc: Union[Number, Mapping[str, Number]],
                  comp: str,
                  time: Number = None,
                  transform_ydata: Callable = None,
                  transform_xdata: Callable = None,
                  ax: Axes = None,
                  line_kw: Mapping = None,
                  fig_kw: Mapping = None) -> Axes:

        coord_kw = self._get_plot_data(line, loc, 1)
        data = self.get(time=time,
                        comp=comp,
                        output_fs=False,
                        squeeze=True,
                        **coord_kw)

        if transform_ydata is not None:
            data = transform_ydata(data)

        if line_kw is None:
            line_kw = {}

        return self.coords.plot_line(line, data,
                                     transform_xdata=transform_xdata,
                                     ax=ax,
                                     fig_kw=fig_kw,
                                     **line_kw)

    def _base_contour(self,
                      plane: str,
                      loc: Union[Number, Mapping[str, Number]],
                      comp: str,
                      time: Number = None,
                      transform_cdata: Callable = None) -> ArrayLike:

        coord_kw = self._get_plot_data(plane, loc, 2)

        data = self.get(time=time,
                        comp=comp,
                        output_fs=False,
                        squeeze=True,
                        **coord_kw)

        coords1 = self.coords.get(plane[0]).size
        coords2 = self.coords.get(plane[1]).size

        index1 = self._data_layout.index(plane[0])
        index2 = self._data_layout.index(plane[1])

        if index1 > index2:
            data = data.T

        if data.shape != (coords1, coords2):
            raise RuntimeError("Invalid shape returned "
                               "from plotting")
        if transform_cdata is not None:
            data = transform_cdata(data)

        return data

    def pcolormesh(self,
                   plane: Iterable[str],
                   loc: Union[Number, Mapping[str, Number]],
                   comp: str,
                   time: Number = None,
                   transform_ydata: Callable = None,
                   transform_xdata: Callable = None,
                   transform_cdata: Callable = None,
                   ax: Axes = None,
                   contour_kw: Mapping = None,
                   fig_kw: Mapping = None) -> Axes:

        data = self._base_contour(plane, loc, comp, time, transform_cdata)

        if contour_kw is None:
            contour_kw = {}

        return self.coords.pcolormesh(plane, data, ax=ax,
                                      transform_xdata=transform_xdata,
                                      transform_ydata=transform_ydata,
                                      fig_kw=fig_kw,
                                      **contour_kw)

    def contourf(self,
                 plane: Iterable[str],
                 loc: Union[Number, Mapping[str, Number]],
                 comp: str,
                 time: Number = None,
                 transform_ydata: Callable = None,
                 transform_xdata: Callable = None,
                 transform_cdata: Callable = None,
                 ax: Axes = None,
                 contour_kw: Mapping = None,
                 fig_kw: Mapping = None) -> Axes:

        data = self._base_contour(plane, loc, comp, time, transform_cdata)

        if contour_kw is None:
            contour_kw = {}

        return self.coords.contourf(plane, data, ax=ax,
                                    transform_xdata=transform_xdata,
                                    transform_ydata=transform_ydata,
                                    fig_kw=fig_kw,
                                    **contour_kw)

    def contour(self,
                plane: Iterable[str],
                loc: Union[Number, Mapping[str, Number]],
                comp: str,
                time: Number = None,
                transform_ydata: Callable = None,
                transform_xdata: Callable = None,
                transform_cdata: Callable = None,
                ax: Axes = None,
                contour_kw: Mapping = None,
                fig_kw: Mapping = None) -> Axes:

        data = self._base_contour(plane, loc, comp, time, transform_cdata)

        if contour_kw is None:
            contour_kw = {}

        return self.coords.contour(plane, data, ax=ax,
                                   transform_xdata=transform_xdata,
                                   transform_ydata=transform_ydata,
                                   fig_kw=fig_kw,
                                   **contour_kw)

    def first_derivative(self, comp: str,
                         axis: str,
                         time: float = None,
                         method: str = None) -> ArrayLike:
        axis_index = self._data_layout.index(axis) + 2
        data = self.get(time=time, comp=comp, squeeze=False, output_fs=False)

        return self.coords.first_derivative(axis, data, axis_index, method=method).squeeze()

    def second_derivative(self, comp: str,
                          axis: str,
                          time: float = None,
                          method: str = None) -> ArrayLike:
        axis_index = self._data_layout.index(axis) + 2
        data = self.get(time=time, comp=comp, squeeze=False, output_fs=False)

        return self.coords.second_derivative(axis, data, axis_index, method=method).squeeze()


@FlowStructND.implements(np.array_equal)
def array_equal(fstruct1: FlowStructND, fstruct2: FlowStructND, *args, **kwargs):
    try:
        fstruct1._check_fstruct_compat(fstruct2, True, True)
    except ValueError:
        return False

    return np.array_equal(fstruct1._array, fstruct2._array)


@FlowStructND.implements(np.allclose)
def allclose(fstruct1: FlowStructND, fstruct2: FlowStructND, *args, **kwargs):
    try:
        fstruct1._check_fstruct_compat(fstruct2, True, True)
    except ValueError:
        return False

    return np.allclose(fstruct1._array, fstruct2._array)
