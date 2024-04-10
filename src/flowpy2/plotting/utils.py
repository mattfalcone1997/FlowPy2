import matplotlib as mpl
import numpy as np

from cycler import cycler


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


reset_prop_cycle()
