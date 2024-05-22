from __future__ import annotations
import numpy as np
import flowpy2 as fp2
from .coords import CoordStruct
from numpy.typing import ArrayLike
from typing import Iterable, Callable, Union, Mapping, List, Tuple
from numbers import Number
from .indexers import CompIndexer, TimeIndexer
from matplotlib.axes import Axes
from .arrays import CommonArrayExtensions, array_backends, ArrayBackends
from .flow_type import FlowType

import warnings
import logging

import weakref
from .utils import find_stack_level

from .io import hdf5, netcdf

logger = logging.getLogger(__name__)


class FlowStructND(CommonArrayExtensions):
    _array_attr = '_array'
    _HANDLED_TYPES = (Number, np.ndarray)

    def __init__(self,
                 coorddata: CoordStruct,
                 array: ArrayLike,
                 comps: Iterable[str],
                 data_layout: Iterable[str] = None,
                 times: Iterable[Number] = None,
                 time_decimals: int = None,
                 array_backend: str = None,
                 dtype: Union[str, type, None] = None,
                 attrs: Mapping = None,
                 location: Mapping = None,
                 copy=False,
                 **kwargs) -> None:

        if data_layout is None:
            data_layout = tuple(coorddata.index)
        self._data_layout = tuple(data_layout)

        self._coords = self._set_coords(coorddata, data_layout)

        self._comps = CompIndexer(comps)
        if times is not None:
            times = TimeIndexer(times, time_decimals)
        self._times = times

        if attrs is None:
            attrs = {}

        if dtype is None:
            dtype = array.dtype.type

        self.attrs = dict(attrs)
        if location is None:
            location = {}

        for loc in location:
            if loc in self._data_layout:
                raise ValueError("Location cannot be "
                                 f"data layout: {loc}")

        self._location = location

        if array_backend is None:
            array_backend = fp2.rcParams['arrays.backend']

        self._set_array(array, array_backend, dtype, copy)

        self._array_backend = array_backend

        self._process_extra_args(**kwargs)

    def _set_coords(self, coorddata: CoordStruct, data_layout: Iterable[str]):
        if not isinstance(coorddata, CoordStruct):
            raise TypeError("coorddata must be an instance of CoordStruct")

        for d in data_layout:
            if d not in coorddata.index:
                raise ValueError(f"Element {d} in data_layout not "
                                 "in coorddata")

        coorddata = coorddata.copy()
        if not coorddata.is_consecutive:
            raise ValueError("CoordStruct must be consecutive is used "
                             f"in {type(self).__name__}")

        for k in coorddata.index:
            if k not in data_layout:
                del coorddata[k]

        return coorddata

    def _set_array(self, array: ArrayLike, array_backend: str, dtype: Union[str, type], copy: bool):

        creator = array_backends.get_creator(array_backend)
        if type(array).__module__ != creator.__module__:
            array = creator(array, dtype=dtype)

        array = array.astype(dtype, copy=copy)

        for i, d in enumerate(self._data_layout[::-1], 1):
            coordsize = self._coords[d].size
            if array.shape[-i] != coordsize:
                raise ValueError(f"Shape of dimension {array.ndim-i-1} "
                                 "of input array is invalid: "
                                 f"{array.shape[-i]} vs {coordsize}")

        ndim = len(self._data_layout)
        if not (array.ndim == ndim and len(self._comps)) == 1:
            loc = -ndim-1
            if array.shape[loc] != len(self._comps):
                raise ValueError(f"Shape of dimension 1 of array "
                                 f"is invalid: {array.shape[loc]} "
                                 f"vs {len(self._comps)}")

        time_size = 1 if self._times is None else len(self._times)
        comp_size = len(self._comps)

        coord_shape = [self._coords[d].size for d in self._data_layout]

        shape = (time_size, comp_size, *coord_shape)
        try:
            self._array = array.reshape(shape)
        except ValueError:
            if self._times is None:
                raise ValueError(f"Array must be shape {shape} "
                                 f"or {shape[1:]} not {array.shape}") from None
            else:
                raise ValueError(f"Array must be shape {shape} "
                                 f"not {array.shape}") from None

    def reduce(self, operation: Callable, axis: str):
        coords = self.coords.copy()
        coords.remove(axis)
        data_layout = list(self._data_layout)
        data_layout.remove(axis)

        for a in axis:
            axis_ind = list(self._data_layout).index(a) + 2
            array = operation(self._array, axis=axis_ind)
        kwargs = self._init_args_from_kwargs(coorddata=coords,
                                             array=array,
                                             data_layout=data_layout)
        return self._create_struct(**kwargs)

    @property
    def flow_type(self) -> FlowType:
        return self._coords.flow_type

    @property
    def location(self) -> Number:
        return self._location

    @property
    def dtype(self) -> type:
        return self._array.dtype.type

    @property
    def coords(self) -> CoordStruct:
        return self._coords

    @property
    def times(self) -> np.ndarray:
        if self._times is None:
            return None

        return np.array(self._times)

    @times.setter
    def times(self, values: Union[np.array, Number]):
        if isinstance(values, Number):
            values = np.array([values])

        if self._times is None and len(values) > 1:
            raise ValueError("If times is None values must be "
                             "1D or a Number")

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

    def _fast_get_single_time(self, comp):
        return self._array[0, self._comps.get(comp)]

    def _fast_get_many_time(self, time, comp):
        return self._array[self._times.get(time),
                           self._comps.get(comp)]

    def get(self, *, time=None,
            comp=None,
            output_fs=True,
            squeeze=True,
            drop_coords=True,
            **coords_kw) -> Union[ArrayLike, FlowStructND]:

        # Note this routine needs a rewrite
        if not coords_kw and squeeze and isinstance(comp, str):

            if self._times is None or len(self._times) == 1:
                return self._fast_get_single_time(comp)
            elif isinstance(time, (Number, str)):
                return self._fast_get_many_time(time, comp)

        indexer = [slice(None)]*self._array.ndim
        shape = [None]*self._array.ndim

        indexer[1], shape[1], out_fsc = self._process_comp_index(comp)

        if self._times is not None:
            indexer[0], shape[0], out_fst = self._process_time_index(time)
        else:
            out_fst = out_fsc

        out_fs = out_fsc or out_fst
        location = {}

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

                for i, d in enumerate(drop_shapes[::-1]):
                    del shape[d]
                    l = self._data_layout[d-2]
                    location[l] = self.coords[l][indexer[d]]

        else:
            coord_dict = self.coords.to_dict()
            data_layout = self._data_layout
            shape[2:] = self.shape

        if self._times is None:
            shape = shape[1:]

        array = self._array[tuple(indexer)].reshape(shape)

        if out_fs and output_fs:
            if self._times is not None:
                times = self.times[indexer[0]]
            else:
                times = None

            comps = self._comps[indexer[1]]
            coords = self._coords.__class__(self.flow_type.name,
                                            coord_dict)

            location_fs = location
            location_fs.update(self._location)
            kwargs = self._init_args_from_kwargs(coorddata=coords,
                                                 array=array,
                                                 times=times,
                                                 comps=comps,
                                                 data_layout=data_layout,
                                                 location=location_fs)
            return self._create_struct(**kwargs)

        else:
            return array if not squeeze else np.squeeze(array)

    @property
    def slice(self):
        return _fstruct_slicer(self)

    def values(self):
        return self._array

    def _preprocess_array_ufunc(self, ufunc, method, *inputs, **kwargs):
        actual_inputs = super()._preprocess_array_ufunc(ufunc, method, *inputs, **kwargs)
        for x in actual_inputs:
            if isinstance(x, self.__class__):
                if isinstance(x, self.__class__):
                    self._check_fstruct_compat(x, True, True)

        return actual_inputs

    def _postprocess_array_ufunc(self,  data):
        kwargs = self._init_args_from_kwargs(array=data)
        return self._create_struct(**kwargs)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.get(time=key[0],
                            comp=key[1])
        elif self._times is None:
            return self.get(comp=key)

        elif len(self._times) == 1:
            return self.get(comp=key)

        else:
            raise KeyError("Invalid key")

    def __setitem__(self, key, value: np.ndarray):

        if self.times is None:

            if key in self._comps:
                if self.shape != value.shape:
                    raise ValueError("Setting array not correct shape: "
                                     f"{self.shape} vs {value.shape}")

                index_comp, _, _ = self._process_comp_index(key)
                self._array[0, index_comp] = value
            else:
                if isinstance(key, str):
                    key = [key]
                kwargs = self._init_args_from_kwargs(array=value,
                                                     comps=key)
                fstruct = self._create_struct(**kwargs)
                self.concat_comps(fstruct, inplace=True)

        else:
            time = key[0]
            comp = key[1]

            if time in self._times and comp in self._comps:
                if self.shape != value.shape:
                    raise ValueError("Setting array not correct shape: "
                                     f"{self.shape} vs {value.shape}")

                index_comp, _, _ = self._process_comp_index(key[1])
                index_time, _, _ = self._process_time_index(key[0])
                self._array[index_time, index_comp] = value
            else:
                if isinstance(comp, str):
                    comp = [comp]

                if not hasattr(time, '__iter__'):
                    time = [time]

                kwargs = self._init_args_from_kwargs(array=value,
                                                     comps=comp,
                                                     times=time)
                fstruct = self._create_struct(**kwargs)
                self.concat(fstruct, inplace=True)

    def remove_times(self, times: Union[Number, Iterable[Number]]):

        index = self._times.get_other(times)
        self._array = self._array[index]
        self._times.remove(times)

    def remove_comps(self, comps: Union[str, Iterable[str]]):

        index = self._comps.get_other(comps)
        self._array = self._array[:, index]
        self._comps.remove(comps)

    def _init_args_from_kwargs(self, **kwargs):
        self._init_update_kwargs(kwargs, 'coorddata', self._coords)
        self._init_update_kwargs(kwargs, 'array', self._array)
        self._init_update_kwargs(kwargs, 'comps', self._comps)
        self._init_update_kwargs(kwargs, 'data_layout', self._data_layout)
        self._init_update_kwargs(kwargs, 'times', self._times)
        if self._times is not None:
            self._init_update_kwargs(
                kwargs, 'time_decimals', self._times.decimals)
        self._init_update_kwargs(kwargs, 'array_backend', self._array_backend)
        self._init_update_kwargs(kwargs, 'dtype', self.dtype)
        self._init_update_kwargs(kwargs, 'attrs', self.attrs)
        self._init_update_kwargs(kwargs, 'location', self._location)

        return kwargs

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
            return slice(None), len(self._comps), True

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
                              check_comps: bool = False,
                              raise_error: bool = True):

        compat = True
        if self.coords != fstruct.coords:
            if raise_error:
                raise ValueError("Coordinates do not match")
            compat = False
        if self._data_layout != fstruct._data_layout:
            if raise_error:
                raise ValueError("Data layout do not match")
            compat = False

        if check_times:
            if self._times != fstruct._times:
                if raise_error:
                    raise ValueError("Times do not match")
                compat = False

        if check_comps:
            if self.comps != fstruct.comps:
                if raise_error:
                    raise ValueError("Components do not match")
                compat = False

        return compat

    def concat_comps(self, fstructs: Union[FlowStructND, Iterable[FlowStructND]],
                     inplace=False) -> FlowStructND:
        if isinstance(fstructs, FlowStructND):
            fstructs = [fstructs]

        new_comps = self._comps.copy()
        arrays = [self._array]
        for fstruct in fstructs:
            self._check_fstruct_compat(fstruct, check_times=True)

            new_comps.extend(fstruct.comps.copy())
            arrays.append(fstruct._array)

        new_data = np.concatenate(arrays, axis=1)

        if inplace:
            logger.debug("Setting array inplace")

            self._comps = new_comps
            self._set_array(new_data,
                            self._array_backend,
                            dtype=self.dtype,
                            copy=False)
            return self
        else:
            logger.debug("Creating new flowstruct in concat")
            kwargs = self._init_args_from_kwargs(array=new_data,
                                                 comps=new_comps)
            return self._create_struct(**kwargs)

    def concat_times(self, fstructs: Union[FlowStructND, Iterable[FlowStructND]],
                     inplace=False) -> FlowStructND:

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

        if inplace:
            logger.debug("Setting array inplace")
            self._times = new_time_indexer
            self._set_array(new_data,
                            self._array_backend,
                            dtype=self.dtype,
                            copy=False)
            return self
        else:
            logger.debug("Creating new flowstruct in concat")
            kwargs = self._init_args_from_kwargs(array=new_data,
                                                 times=new_time_indexer)
            return self._create_struct(**kwargs)

    def concat(self, fstructs:  Union[FlowStructND, Iterable[FlowStructND]],
               inplace=False) -> FlowStructND:

        if type(fstructs) == type(self):
            fstructs = [fstructs]
        # check types

        if not all(type(self) == type(fstruct) for fstruct in fstructs):
            raise ValueError("fstructs must be the same "
                             "type or an iterable of them")

        if all(self._times == fstruct._times for fstruct in fstructs):
            return self.concat_comps(fstructs, inplace=inplace)

        elif all(self._comps == fstruct._comps for fstruct in fstructs):
            return self.concat_times(fstructs, inplace=inplace)

        else:
            raise ValueError(f"{type(self).__name__} must have same comps or "
                             "times to be concatenated")

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
        self._coords.Translate(**kwargs)

    def to_hdf(self, fn_or_obj: str, mode: str = None, key: str = None, compress=False):

        g = hdf5.hdfHandler(fn_or_obj, mode, key)

        g.set_type_tag(type(self))

        compression = 'zlib' if compress else None
        g.create_dataset("array",
                         data=self._array,
                         compression=compression)

        self._comps.to_hdf(g, 'comps')

        if self._times is not None:
            self._times.to_hdf(g, 'times')

        self._coords.to_hdf(g, key='coords')

        g.create_dataset("data_layout",
                         data=np.array(self._data_layout, dtype=np.string_))

        g.attrs['array_backend'] = self._array_backend
        for k, v in self.attrs:
            g.attrs[k] = v

        for k, v in self._location:
            g.attrs[f'location_{k}'] = v

        self._hdf5_write_hook(g)

        return g


    @classmethod
    def from_hdf(cls,
                 fn_or_obj: str,
                 key: str = None,
                 comps: Iterable[str] = None,
                 times: Iterable[Number] = None,
                 tag_check=None) -> FlowStructND:

        g = hdf5.hdfHandler(fn_or_obj, 'r', key)
        if tag_check is None:
            tag_check = fp2.rcParams['io.tag_check']

        real_cls = g.validate_tag(cls, tag_check)

        coords = CoordStruct.from_hdf(g, 'coords', tag_check=tag_check)
        array = g['array'][:]
        comps = CompIndexer.from_hdf(g, 'comps')
        
        times = TimeIndexer.from_hdf(g, 'times')
        decimals = times.decimals if times is not None else None

        attrs = dict(g.attrs)

        data_layout = tuple([key.decode('utf-8')
                            for key in g['data_layout'][:]])
        array_backend = attrs['array_backend']

        location_keys = [k.removeprefix('location_')
                         for k in attrs if 'location_' in k]
        location = {k: attrs[f'location_{k}'] for k in location_keys}

        kwargs = real_cls._hdf5_read_hook(g)

        for k in location_keys:
            del attrs[f'location_{k}']

        del attrs['array_backend']
        del attrs['type_tag']

        return real_cls._create_struct(coorddata=coords,
                                       array=array,
                                       comps=comps,
                                       data_layout=data_layout,
                                       times=times,
                                       decimals=decimals,
                                       array_backend=array_backend,
                                       attrs=attrs,
                                       location=location,
                                       **kwargs)

    def to_netcdf(self, fn_or_obj: str, mode: str = None, key: str = None, compress=True):

        if not netcdf.HAVE_NETCDF4:
            raise ModuleNotFoundError("Cannot use netcdf")

        g = netcdf.make_dataset(fn_or_obj, mode, key)
        netcdf.set_type_tag(type(self), g)

        self._coords.to_netcdf(g)

        layout = []
        if self._times is not None:
            self._times.to_netcdf(g)
            layout.append('time')

        layout.extend(self._data_layout)

        dtype = self._array.dtype.kind + str(self._array.dtype.itemsize)
        for comp in self.comps:
            var = g.createVariable(comp, dtype, tuple(layout))
            var[:] = self.get(comp=comp, output_fs=False)

        g.data_layout = self._data_layout
        g.comps = list(self._comps)

        self._netcdf_write_hook(g)

        g._array_backend = self._array_backend
        for k, v in self.attrs.items():
            setattr(g, f'_attr_{k}', v)

        for k, v in self._location.items():
            setattr(g, f'_location_{k}', v)

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
        decimals = times.decimals if times is not None else None

        data_layout = g.data_layout

        comps = list(g.comps)
        array = [g.variables[comp][:] for comp in comps]
        axis = 1 if times is not None else 0

        array = np.stack(array, axis=axis)

        kwargs = cls._netcdf_read_hook(g)

        if hasattr(g, '_array_backend'):
            array_backend = g._array_backend
        else:
            array_backend = 'numpy'

        attrs = {}
        location = {}
        for k in g.__dict__.keys():
            if '_attr_' in k:
                attr = k.removeprefix('_attr_')
                attrs[attr] = getattr(g, k)

            if '_location_' in k:
                attr = k.removeprefix('_location_')
                location[attr] = getattr(g, k)

        return cls._create_struct(coorddata=coords,
                                  array=array,
                                  comps=comps,
                                  data_layout=data_layout,
                                  times=times,
                                  decimals=decimals,
                                  array_backend=array_backend,
                                  attrs=attrs,
                                  location=location,
                                  **kwargs)

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
            raise TypeError("Invalid type for "
                            f"location ({type(loc).__name__})")

        if len(loc) != dim:
            raise ValueError(f"Size of loc should be {dim} not {len(loc)}")

        return loc

    def _single_line_plot(self,
                          ax: Axes,
                          line: str,
                          comp: str,
                          loc: Union[Mapping, Number],
                          time: Number,
                          transform_xdata: Callable,
                          transform_ydata: Callable,
                          fig_kw: dict,
                          **line_kw):

        coord_kw = self._get_plot_data(line, loc, 1)

        data = self.get(time=time,
                        comp=comp,
                        output_fs=False,
                        squeeze=True,
                        **coord_kw)

        if transform_ydata is not None:
            data = transform_ydata(data)

        return self.coords.plot_line(line, data,
                                     transform_xdata=transform_xdata,
                                     ax=ax,
                                     fig_kw=fig_kw,
                                     **line_kw)

    def plot_line(self,
                  comp: str,
                  line: str = None,
                  loc: Union[Iterable, Number, Mapping[str, Number]] = None,
                  time: Number = None,
                  transform_ydata: Callable = None,
                  transform_xdata: Callable = None,
                  ax: Axes = None,
                  fig_kw: Mapping = None,
                  labels=None,
                  **line_kw) -> Axes:

        if self.ndim == 1:
            if line is not None and line not in self._data_layout:
                warnings.warn(("%s is 1D using only"
                              " valid line '%s' not %s") % (type(self).__name__,
                                                            self._data_layout[0],
                                                            line),
                              stacklevel=find_stack_level())
            line = self._data_layout[0]

            if loc is not None:
                warnings.warn(("%s is 1D loc"
                               " %s cannot be used") % (type(self).__name__,
                                                        loc),
                              stacklevel=find_stack_level())
            loc = {}

        time = self.check_time(time)

        if line_kw is None:
            line_kw = {}

        if isinstance(loc, (Mapping, Number)):
            if labels is not None:
                warnings.warn("labels ignored: only one location",
                              category=UserWarning,
                              stacklevel=find_stack_level())

            return self._single_line_plot(ax,
                                          line,
                                          comp,
                                          loc,
                                          time,
                                          transform_xdata,
                                          transform_ydata,
                                          fig_kw,
                                          **line_kw)
        else:
            if not hasattr(loc, '__iter__'):
                raise TypeError("Argument 'loc' must be a Number, Mapping "
                                "or an iterable of them")

            if labels is not None:
                if 'label' in line_kw:
                    raise ValueError("'label' and 'labels' cannot "
                                     "both be supplied")

                if len(labels) != len(loc):
                    raise ValueError("'labels' must be the same size as 'loc'")

            for i, loc in enumerate(loc):
                if labels is not None:
                    line_kw['label'] = labels[i]

                ax = self._single_line_plot(ax,
                                            line,
                                            comp,
                                            loc,
                                            time,
                                            transform_xdata,
                                            transform_ydata,
                                            fig_kw,
                                            **line_kw)
            return ax

    def check_time(self, time):

        if time is None:
            if self._times is None:
                return None

            elif len(self._times) == 1:
                return self._times[0]

            else:
                raise ValueError("Time must be specified if there is more "
                                 f"than one time in {type(self).__name__}")

        else:
            return time

    def _base_contour(self,
                      comp: str,
                      plane: str,
                      loc: Union[Number, Mapping[str, Number]],
                      time: Number = None,
                      transform_cdata: Callable = None) -> ArrayLike:

        if self.ndim == 2:
            if loc is not None:
                warnings.warn(("%s is 2D loc"
                               " %s cannot be used") % (type(self).__name__,
                                                        loc),
                              stacklevel=find_stack_level())
            loc = {}

        elif self.ndim < 2:
            raise ValueError(f"{type(self).__name__}.ndim must be at "
                             "least 2 to create contour plot")

        coord_kw = self._get_plot_data(plane, loc, 2)

        time = self.check_time(time)

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
                   comp: str,
                   plane: Iterable[str],
                   loc: Union[Number, Mapping[str, Number]] = None,
                   time: Number = None,
                   transform_ydata: Callable = None,
                   transform_xdata: Callable = None,
                   transform_cdata: Callable = None,
                   ax: Axes = None,
                   fig_kw: Mapping = None,
                   **contour_kw) -> Axes:

        data = self._base_contour(comp, plane, loc, time, transform_cdata)

        if contour_kw is None:
            contour_kw = {}

        return self.coords.pcolormesh(plane, data, ax=ax,
                                      transform_xdata=transform_xdata,
                                      transform_ydata=transform_ydata,
                                      fig_kw=fig_kw,
                                      **contour_kw)

    def contourf(self,
                 comp: str,
                 plane: Iterable[str],
                 loc: Union[Number, Mapping[str, Number]] = None,
                 time: Number = None,
                 transform_ydata: Callable = None,
                 transform_xdata: Callable = None,
                 transform_cdata: Callable = None,
                 ax: Axes = None,
                 fig_kw: Mapping = None,
                 **contour_kw) -> Axes:

        data = self._base_contour(comp, plane, loc, time, transform_cdata)

        if contour_kw is None:
            contour_kw = {}

        return self.coords.contourf(plane, data, ax=ax,
                                    transform_xdata=transform_xdata,
                                    transform_ydata=transform_ydata,
                                    fig_kw=fig_kw,
                                    **contour_kw)

    def contour(self,
                comp: str,
                plane: Iterable[str],
                loc: Union[Number, Mapping[str, Number]] = None,
                time: Number = None,
                transform_ydata: Callable = None,
                transform_xdata: Callable = None,
                transform_cdata: Callable = None,
                ax: Axes = None,
                fig_kw: Mapping = None,
                **contour_kw) -> Axes:

        data = self._base_contour(comp, plane, loc, time, transform_cdata)

        if contour_kw is None:
            contour_kw = {}

        return self.coords.contour(plane, data, ax=ax,
                                   transform_xdata=transform_xdata,
                                   transform_ydata=transform_ydata,
                                   fig_kw=fig_kw,
                                   **contour_kw)

    def quiver(self,
               comps: Iterable[str],
               plane: Iterable[str],
               loc: Union[Number, Mapping[str, Number]] = None,
               time: Number = None,
               spacing: Tuple[int] = (1, 1),
               transform_ydata: Callable = None,
               transform_xdata: Callable = None,
               transform_cdata: Callable = None,
               ax: Axes = None,
               fig_kw: Mapping = None,
               **quiver_kw):

        u_data = self._base_contour(
            comps[0], plane, loc, time, transform_cdata)
        v_data = self._base_contour(
            comps[1], plane, loc, time, transform_cdata)

        if quiver_kw is None:
            quiver_kw = {}

        return self.coords.quiver(plane,
                                  u_data,
                                  v_data,
                                  spacing=spacing,
                                  transform_xdata=transform_xdata,
                                  transform_ydata=transform_ydata,
                                  fig_kw=fig_kw,
                                  ax=ax,
                                  **quiver_kw)

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

    def to_vtk(self, time=None, comps=None):

        grid = self.coords.to_vtk(layout=self._data_layout)
        if not (self.times is None or len(self.times) == 1) and time is None:
            raise ValueError("There are multiple times, time must be present")

        if comps is None:
            comps = self.comps

        for comp in comps:
            grid.point_data[comp] = np.ravel(
                self.get(time=time, comp=comp), order='F')

        return grid

    def window(self, method, *args, inplace=False, **kwargs):
        if method == 'uniform':
            data, times = self._window_uniform(*args, **kwargs)

        elif method == 'avg_time':
            data, times = self._window_avg_time(*args, **kwargs)
        else:
            raise NotImplementedError("Window method not implemented")

        if inplace:
            self._times = TimeIndexer(times)
            self._set_array(data, self._array_backend,
                            dtype=self.dtype, copy=False)
            return self
        else:
            kwargs = self._init_args_from_kwargs(array=data,
                                                 times=times)

            return self._create_struct(**kwargs)

    def _window_uniform(self, hwidth, *args, **kwargs):
        times = self.times
        if hasattr(hwidth, '__len__'):
            if len(hwidth) != len(times):
                raise ValueError("Invalid length of hwidths")
        else:
            if not all(np.diff(times) < hwidth):
                warnings.warn("All times are not spaced "
                              f"smaller than hwidth ({hwidth})",
                              stacklevel=find_stack_level())

        data = np.zeros_like(self._array)
        for j, time in enumerate(times):
            hwidth_val = hwidth[j] if hasattr(hwidth, '__len__') else hwidth
            val_indices = np.logical_and(
                times > time-hwidth, times < time+hwidth_val)
            valid_times = times[val_indices]
            if len(valid_times) == 1:
                warnings.warn(f"Window average with hwidth {hwidth}"
                              f" has one time: {valid_times[0]}",
                              stacklevel=find_stack_level())

            ltime = valid_times[0]
            utime = valid_times[-1]

            data[j] = self.get(time=slice(ltime, utime),
                               squeeze=False, output_fs=False).mean(axis=0)

        return data, times

    def _window_avg_time(self, hwidth, avg_start):
        times = self.times
        t0 = times[0] + hwidth
        t1 = times[-1] - hwidth
        times = times[np.logical_and(times > t0, times < t1)]

        shape = list(self._array.shape)
        shape[0] = len(times)

        data = np.zeros(shape, dtype=self.dtype)

        for j, time in enumerate(times):

            i1 = np.argmin(abs(self.times - (time + hwidth)))
            i0 = np.argmin(abs(self.times - (time - hwidth)))

            T1 = self.times[i1] - avg_start
            T0 = self.times[i0] - avg_start

            data[j] = (T1*self._array[i1] - T0*self._array[i0])/(T1 - T0)

        return data, times

    def time_to_ND(self) -> FlowStructND:
        if self._times is None:
            raise ValueError("This function can only be called "
                             "if times are present")

        if 't' in self._data_layout:
            return self

        new_array = np.moveaxis(self._array, 0, -1).copy()
        data_layout = self._data_layout + ('t',)

        new_flow_type = self.flow_type.get_time_type().name

        coords = self.coords.copy()
        coords.flow_type = new_flow_type
        coords.concat(coords.__class__(new_flow_type,
                                       {'t': self.times}))
        logger.debug(f"Shape: {new_array.shape} {len(coords)}")
        kwargs = self._init_args_from_kwargs(coorddata=coords,
                                             array=new_array,
                                             data_layout=data_layout,
                                             times=None,
                                             time_decimals=None)
        return self._create_struct(**kwargs)

    def equals(self, other):
        try:
            self._check_fstruct_compat(other, True, True)
        except ValueError:
            return False

        return np.array_equal(self._array, other._array)

    def __eq__(self, other_datastruct):
        return self.equals(other_datastruct)

    def __ne__(self, other_datastruct):
        return not self.equals(other_datastruct)

    def __str__(self):
        return "%s(%s, comps=%s, times=%s, shape=%s)" % (type(self).__name__,
                                                         self.flow_type.name,
                                                         list(self.comps),
                                                         self.times,
                                                         self.shape)

    def _array_function_check_meta(self, fstruct: FlowStructND):
        try:
            self._check_fstruct_compat(fstruct, True, True)
        except ValueError:
            return False

        return True


@FlowStructND.implements(np.array_equal)
def array_equal(fstruct1: FlowStructND, fstruct2: FlowStructND, *args, **kwargs):
    try:
        fstruct1._check_fstruct_compat(fstruct2, True, True)
    except ValueError:
        return False

    return np.array_equal(fstruct1._array, fstruct2._array)


@FlowStructND.implements(np.allclose)
def allclose(fstruct1: FlowStructND, fstruct2: FlowStructND, *args, **kwargs):
    return fstruct1.equals(fstruct2)


class _fstruct_slicer:
    def __init__(self, fstruct: FlowStructND):
        self._fstruct = weakref.ref(fstruct)

    def __getitem__(self, args) -> FlowStructND:
        if not isinstance(args, tuple):
            args = (args,)

        coords = self._fstruct()._data_layout

        coord_slicer = {k: val for k, val in zip(coords, args)}

        return self._fstruct().get(**coord_slicer)
