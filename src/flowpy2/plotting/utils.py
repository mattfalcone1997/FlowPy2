import matplotlib as mpl
import numpy as np

from cycler import cycler

from matplotlib.collections import RegularPolyCollection
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.transforms import Transform

from typing import Iterable
from numbers import Number

def _linekw_alias(**kwargs):
    alias_dict = {'aa': 'antialiased',
                  'c': 'color',
                  'ds': 'drawstyle',
                  'ls': 'linestyle',
                  'lw': 'linewidth',
                  'mec': 'markeredgecolor',
                  'mew': 'markeredgewidth',
                  'mfc': 'markerfacecolor',
                  'mfcalt': 'markerfacecoloralt',
                  'ms': 'markersize'}

    new_dict = {}
    for key, val in kwargs.items():
        if key in alias_dict.keys():
            key = alias_dict[key]
        new_dict[key] = val
    return new_dict


def update_prop_cycle(**kwargs):
    avail_keys = [x[4:]
                  for x in mpl.lines.Line2D.__dict__.keys() if x[0:3] == 'get']

    if not all([key in avail_keys for key in kwargs.keys()]):
        msg = "The key is invalid for the matplotlib property cycler"
        raise ValueError(msg)

    kwargs = _linekw_alias(**kwargs)

    cycler_dict = mpl.rcParams['axes.prop_cycle'].by_key()
    for key, item in kwargs.items():
        if not hasattr(item, "__iter__"):
            item = [item]
        elif isinstance(item, str):
            if item == "":
                item = [item]
        cycler_dict[key] = item

    item_length = [len(item) for _, item in cycler_dict.items()]
    cycle_length = np.lcm.reduce(item_length)

    for key, val in cycler_dict.items():
        cycler_dict[key] = list(val)*int(cycle_length/len(val))
    mpl.rcParams['axes.prop_cycle'] = cycler(**cycler_dict)


_default_prop_dict = dict(linestyle=['-', '--', '-.', ':'],
                         marker=['x', '.', 'v', '^', '+'],
                         color='bgrcmyk')

def set_default_prop_dict(mapping: dict):
    _default_prop_dict.clear()
    _default_prop_dict.update(mapping)

def reset_prop_cycle(**kwargs):
    update_prop_cycle(**_default_prop_dict)
    update_prop_cycle(**kwargs)


def create_colorbar(ax: Axes,
                    qm: RegularPolyCollection,
                    patch: Iterable,
                    h_pad: Number=0.,
                    w_pad: Number=0.,
                    transform: Transform=None,
                    background=True,
                    in_layout=True,
                    **kwds):

    if transform is None:
        transform = ax.transAxes
        
    inverted_transform = transform.inverted()

    cax = ax.inset_axes(patch, transform=transform,in_layout=in_layout)
    cbar = ax.figure.colorbar(
        qm, cax=cax, **kwds)
    
    if background:
        bbox = cax.get_tightbbox(ax.figure.canvas.get_renderer())

        x0, y0, width, height = bbox.bounds
        width, height = inverted_transform.transform_point([x0+width, y0+height])
        x0, y0 = inverted_transform.transform_point([x0, y0])

        width -= x0
        height -= y0
        ax.add_patch(Rectangle((x0-w_pad, y0-h_pad), width+2*w_pad,
                            height+2*h_pad, transform=transform, fc='w', zorder=2))
    return cax

reset_prop_cycle()
