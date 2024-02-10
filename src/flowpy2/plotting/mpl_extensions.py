"""
## plot
This is a module extending matplotlib functionality
for simpler high-level use in this application
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cycler import cycler

import itertools
import warnings
import copy
from shutil import which

import matplotlib.tri as mtri
from scipy.interpolate import interp1d
from ..utils import find_stack_level

from matplotlib.axes._base import _process_plot_var_args
import matplotlib.lines as mlines
from matplotlib import cbook


class FlowAxes(mpl.axes.Axes):
    name = 'FlowAxes'

    def plot(self, *args, scalex=True, scaley=True, data=None, **kwargs):
        lines = super().plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)

        twinned = [twin for twin in self._twinned_axes.get_siblings(
            self) if twin != self]

        for twin in twinned:
            _kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
            if mpl._version.__version__ >= '3.8':
                _lines = [*twin._get_lines(twin, *args, data=data, **_kwargs)]
            else:
                _lines = [*twin._get_lines(*args, data=data, **_kwargs)]

        return lines

    cplot = plot

    def apply_func_contours(self, comp, func):
        quadmesh_list = [x for x in self.get_children()
                         if isinstance(x, mpl.collections.QuadMesh)]
        contour_list = [x for x in self.get_children()
                        if isinstance(x, mpl.collections.PathCollection)]

        if comp == 'x':
            indexer_quad = (slice(None), slice(None), 0)
            indexer_contour = (slice(None), 0)
        else:
            indexer_quad = (slice(None), slice(None), 1)
            indexer_contour = (slice(None), 1)

        if quadmesh_list:
            for quad in quadmesh_list:
                quad._coordinates[indexer_quad] = func(
                    quad._coordinates[indexer_quad])

        if contour_list:
            for path_col in contour_list:
                paths = path_col.get_paths()
                for path in paths:
                    vertices = path._vertices
                    vertices[indexer_contour] = func(vertices[indexer_contour])
                    path._vertices = vertices

    def normalise(self, axis, val, use_gcl=False):

        lines = self.get_lines()[-1:] if use_gcl else self.get_lines()

        for i, line in enumerate(lines):
            if hasattr(val, "__iter__"):
                if len(val) != len(self.get_lines()):
                    raise ValueError(
                        "There must be as many lines as normalisation values")
                norm_val = val[i]
            else:
                if hasattr(val, "__iter__"):
                    if len(val) != len(line.get_xdata()):
                        raise RuntimeError("The length of vals must be the same as the" +
                                           "number of lines in an axis")
                norm_val = val
            xdata = 0
            ydata = 0
            xdata, ydata = line.get_data()

            if axis == 'x':
                xdata = np.array(xdata)/norm_val
            else:
                ydata = np.array(ydata)/norm_val
            line.set_data(xdata, ydata)

        lim_val = max(val) if hasattr(val, "__iter__") else val

        def _other_logscale_lim(axis, lims):
            if axis == 'x':
                scale = self.get_xscale()
            else:
                scale = self.get_yscale()

            if scale == 'log' and any([lim < 0 for lim in lims]):
                return True
            else:
                return False

        if self.get_lines():
            if axis == 'x':
                xlims = [x/lim_val for x in self.get_xlim()]
                if _other_logscale_lim(axis, xlims):
                    data = np.array([line.get_xdata()
                                    for line in self.get_lines()])
                    xlims = [np.amin(data), np.amax(data)]

                self.set_xlim(xlims)
            else:
                ylims = [y/lim_val for y in self.get_ylim()]

                if _other_logscale_lim(axis, ylims):
                    data = np.array([line.get_ydata()
                                    for line in self.get_lines()])
                    ylims = [np.amin(data), np.amax(data)]

                self.set_ylim(ylims)

    def array_normalise(self, axis, val, use_gcl=False):
        lines = self.get_lines()[-1:] if use_gcl else self.get_lines()

        for line in lines:
            xdata, ydata = line.get_data()
            if xdata.size != len(val):
                raise ValueError(
                    "The size of val must be the same as the data")
            if axis == 'x':
                xdata = np.array(xdata)/val
            else:
                ydata = np.array(ydata)/val
            line.set_data(xdata, ydata)

        self.relim()
        self.autoscale_view(True, True, True)

    def apply_func(self, axis, func, use_gcl=False):

        lines = self.get_lines()[-1:] if use_gcl else self.get_lines()
        for line in lines:
            xdata, ydata = line.get_data()
            if axis == 'y':
                ydata = func(ydata)
            elif axis == 'x':
                xdata = func(xdata)
            else:
                raise KeyError
            line.set_data(xdata, ydata)

        self.relim()
        self.autoscale_view(True, True, True)


mpl.projections.register_projection(FlowAxes)

_default_projection = ['FlowAxes']


def set_default_projection(projection):
    _default_projection[0] = projection


figure = plt.figure


def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, *args, **fig_kw):
    fig = plt.figure(*args, **fig_kw)
    if subplot_kw is None:
        subplot_kw = {'projection': _default_projection[0]}

    ax = fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=squeeze,
                      subplot_kw=subplot_kw, gridspec_kw=gridspec_kw)
    return fig, ax


def show(*args, **kwargs):
    plt.show(*args, **kwargs)


def close(*args, **kwargs):
    plt.close(*args, **kwargs)
