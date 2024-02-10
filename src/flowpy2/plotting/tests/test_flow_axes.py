import matplotlib.pyplot as plt
import matplotlib as mpl
import pytest
import numpy as np

from matplotlib.testing.decorators import check_figures_equal


from flowpy2.plotting import update_prop_cycle


@check_figures_equal()
def test_basic_plots(fig_test: mpl.figure.Figure, fig_ref: mpl.figure.Figure):

    ax_test = fig_test.subplots(subplot_kw={'projection': 'FlowAxes'})
    ax_ref = fig_ref.subplots()

    x = np.arange(100)
    y = np.arange(100)

    ax_test.plot(x, y)
    ax_ref.plot(x, y)


@check_figures_equal()
def test_twinx(fig_test: mpl.figure.Figure, fig_ref: mpl.figure.Figure):
    ax_test = fig_test.subplots(subplot_kw={'projection': 'FlowAxes'})
    ax_ref = fig_ref.subplots()

    x = np.arange(100)
    y = np.arange(100)

    update_prop_cycle(c='br')

    ax_test1 = ax_test.twinx()
    ax_test.plot(x, y, marker='')
    ax_test.set_ylim([0, 200])
    ax_test1.plot(x, 2.*y, marker='')

    ax_ref.plot(x, y, c='b', marker='')
    ax_ref.set_ylim([0, 200])
    ax_ref1 = ax_ref.twinx()
    ax_ref1.plot(x, 2.*y, c='r', marker='', ls='--')
