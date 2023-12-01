import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Sequence, Union, Callable, Mapping
from numbers import Number

from .datastruct import DataStruct
from .flow_type import FlowType
from .gradient import (first_derivative,
                       second_derivative)

logger = logging.getLogger(__name__)


class CoordStruct(DataStruct):
    def __init__(self, flow_type: FlowType, *args, **kwargs):
        if not isinstance(flow_type, FlowType):
            raise TypeError(f"flow_type ust be an instance of FlowTypeBase")

        self._flow_type = flow_type

        super().__init__(*args, **kwargs)
        self._flow_type.validate_keys(self.index)
        self._validate_coords()

    def _validate_coords(self):
        for d in self._data:
            diff = np.diff(d)
            if any(diff < 0):
                raise ValueError("Coordinates must be in ascending order")

    def _validate_inputs(self, inputs):
        super()._validate_inputs(inputs)
        for x in inputs:
            if isinstance(x, self.__class__):
                if self.flow_type is not x.flow_type:
                    raise ValueError("Flow type should be the same")

    @property
    def flow_type(self):
        return self._flow_type

    def rescale(self, key: str, val: Number):
        self._data[key] /= val

    def _create_axis(self, fig_kw):
        logger.debug("Create axis in plot_line")

        if fig_kw is None:
            fig_kw = {}
        fig_kw.update({'projection': self.flow_type.projection})
        return plt.subplots(**fig_kw)

    def plot_line(self, comp: str, data: Sequence, ax=None,
                  transform_xdata: Callable = None,
                  fig_kw=None, **kwargs):
        if ax is None:
            fig, ax = self._create_axis(fig_kw)

        coords = self.get(comp)

        if transform_xdata is not None:
            if not callable(transform_xdata):
                raise TypeError("transform_xdata must be callable")
            coords = transform_xdata(coords)

        if len(data) != len(coords):
            raise ValueError("Coordinate and data have different "
                             f"shapes ({len(data)}) vs ({len(coords)}).")

        ax.plot(coords, data, **kwargs)

        return ax

    def _get_coords_contour(self, plane, transform_xdata: Callable = None, transform_ydata: Callable = None):
        if len(plane) != 2:
            raise ValueError("Length of plane must be 2")

        xcoords = self.get(plane[0])
        ycoords = self.get(plane[1])

        if transform_xdata is not None:
            if not callable(transform_xdata):
                raise TypeError("transform_xdata must be callable")
            xcoords = transform_xdata(xcoords)

        if transform_ydata is not None:
            if not callable(transform_ydata):
                raise TypeError("transform_ydata must be callable")
            ycoords = transform_ydata(ycoords)

        return xcoords, ycoords

    def contour(self, plane: Sequence[str], data: np.ndarray, ax=None,
                transform_xdata=None,
                transform_ydata=None,
                fig_kw=None, **kwargs):

        if ax is None:
            fig, ax = self._create_axis(fig_kw)

        xcoords, ycoords = self._get_coords_contour(plane, transform_xdata,
                                                    transform_ydata)

        return ax.contour(ycoords, xcoords, data, **kwargs)

    def contourf(self, plane: Sequence[str], data: np.ndarray, ax=None,
                 transform_xdata=None,
                 transform_ydata=None,
                 fig_kw=None, **kwargs):
        if ax is None:
            fig, ax = self._create_axis(fig_kw)

        xcoords, ycoords = self._get_coords_contour(plane, transform_xdata,
                                                    transform_ydata)

        return ax.contourf(ycoords, xcoords, data, **kwargs)

    def pcolormesh(self, plane: Sequence[str], data: np.ndarray, ax=None,
                   transform_xdata=None,
                   transform_ydata=None,
                   fig_kw: Mapping = None, **kwargs):
        if ax is None:
            fig, ax = self._create_axis(fig_kw)

        xcoords, ycoords = self._get_coords_contour(plane, transform_xdata,
                                                    transform_ydata)

        return ax.pcolormesh(ycoords, xcoords, data, **kwargs)

    def coord_index(self, comp: str, loc: Union[Number, list, slice]):
        if isinstance(loc, Number):
            logger.debug("Input loc is a number")
            coords = self.get(comp)

            max_coord = np.amax(coords)
            if loc > max_coord:
                if loc < max_coord + 0.5*(coords[-1] - coords[-2]):
                    return coords.size - 1
                raise ValueError("loc outside of bounds")

            min_coord = np.amin(coords)
            if loc < min_coord:
                if loc > min_coord - 0.5*(coords[1] - coords[0]):
                    return 0
                raise ValueError("loc outside of bounds")

            return np.argmin(np.abs(coords - loc))

        elif isinstance(loc, (list, tuple, np.ndarray)):
            logger.debug("Input loc is a is list, tuple or array")
            return [self.coord_index(comp, l) for l in loc]

        elif isinstance(loc, slice):
            logger.debug("Input loc is a slice")

            if loc.step is not None:
                raise NotImplementedError("slice step not allowed for now")

            if loc.stop is None:
                stop = self.get(comp).size
            else:
                stop = self.coord_index(comp, loc.stop) + 1

            if loc.start is None:
                start = 0
            else:
                start = self.coord_index(comp, loc.start)

            if start >= stop:
                raise ValueError("slice stop must be greater than start")

            return slice(start, stop)

        else:
            raise TypeError("CoordStruct cannot be indexed with "
                            f"type {type(loc).__name__}")

    def first_derivative(self, comp, data, axis=0, method=None):
        coords = self.get(comp)
        return first_derivative(data, coords, axis=axis, methof=method)

    def second_derivative(self, comp, data, axis=0, method=None):
        coords = self.get(comp)
        return second_derivative(data, coords, axis=axis, methof=method)

    def integrate(self, comp, data, axis=0, method=None):
        pass

    def cumulative_integrate(self, comp, data, axis=0, method=None):
        pass

    def interpolate(self, other_cstruct, data):
        pass

    def copy(self):
        return self.__class__(self._flow_type, self.to_dict(), copy=True)


@CoordStruct.implements(np.allclose)
def allclose(dstruct1: CoordStruct, dstruct2: CoordStruct, *args, **kwargs):
    if dstruct1.index != dstruct2.index:
        return False

    if dstruct1.flow_type != dstruct2.flow_type:
        return False

    for d1, d2 in zip(dstruct1._data, dstruct2._data):
        if not np.allclose(d1, d2, *args, **kwargs):
            return False

    return True


@CoordStruct.implements(np.array_equal)
def array_equal(dstruct1: CoordStruct, dstruct2: CoordStruct, *args, **kwargs):
    if dstruct1.index != dstruct2.index:
        return False

    if dstruct1.flow_type != dstruct2.flow_type:
        return False

    for d1, d2 in zip(dstruct1._data, dstruct2._data):
        if not np.array_equal(d1, d2, *args, **kwargs):
            return False

    return True
