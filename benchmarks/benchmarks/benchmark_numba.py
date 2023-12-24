import numpy as np
from .common import BenchMark
import flowpy.gradient as gr
import flowpy.numba.gradient as nbgr


class _GradientsTestSuite(BenchMark):
    _array_size = None
    _grad_size = 500
    _dx = 0.01
    _x = np.linspace(0, 1, _grad_size)

    def setup(self):
        self._array = np.random.randn(self._array_size, self._grad_size)

    def time_gradient1_order2_dx(self):
        nbgr.gradient1_order2_dx(self._array, self._dx)

    def time_gradient2_order2_dx(self):
        nbgr.gradient2_order2_dx(self._array, self._dx)

    def time_numpy_gradient1_dx(self):
        np.gradient(self._array, self._dx)

    def time_gradient1_order2_var_x(self):
        nbgr.gradient1_order2_var_x(self._array, self._x)

    def time_gradient2_order2_var_x(self):
        nbgr.gradient2_order2_var_x(self._array, self._x)

    def time_numpy_gradient1_var_x(self):
        np.gradient(self._array, self._x)

    def time_gradient1_order6_dx(self):
        nbgr.gradient1_order6_dx(self._array, self._dx)

    def time_gradient2_order6_dx(self):
        nbgr.gradient2_order6_dx(self._array, self._dx)

    def time_gradient2_order6_dx(self):
        nbgr.gradient2_order6_dx(self._array, self._dx)

    def time_gradient1_order6_var_x(self):
        nbgr.gradient1_order6_var_x(self._array, self._x)

    def time_gradient2_order6_var_x(self):
        nbgr.gradient2_order6_var_x(self._array, self._x)

    def time_gradient2_order6_var_x(self):
        nbgr.gradient2_order6_var_x(self._array, self._x)


class smallGradientsTestSuite(BenchMark):
    _array_size = 1


class medGradientsTestSuite(BenchMark):
    _array_size = 500


class largeGradientsTestSuite(BenchMark):
    _array_size = 5000
