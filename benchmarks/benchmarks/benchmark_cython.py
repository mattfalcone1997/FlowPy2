import numpy as np
from .common import (BenchMark,
                     get_params)
import flowpy.cython.gradient as cgr
import os


class suite_gradients_cython(BenchMark):

    def setup(self, n, nthread):
        os.environ['OMP_NUM_THREADS'] = f"{nthread}"

        self.grad_size = 500
        self.dx = 0.01
        self.array = np.random.randn(n, self.grad_size)

        self.array_t = self.array.T
        self.x = np.linspace(0, 1, self.grad_size)

    def time_gradient1_order2_dx(self, n, nthread):
        cgr.gradient1_order2_dx(self.array, self.dx)

    def time_gradient2_order2_dx(self, n, nthread):
        cgr.gradient2_order2_dx(self.array, self.dx)

    def time_numpy_gradient1_dx(self, n, nthread):
        np.gradient(self.array, self.dx, axis=-1)

    def time_numpy_gradient1_dx_outer(self, n, nthread):
        np.gradient(self.array_t, self.dx, axis=0)

    def time_gradient1_order2_var_x(self, n, nthread):
        cgr.gradient1_order2_var_x(self.array, self.x)

    def time_gradient2_order2_var_x(self, n, nthread):
        cgr.gradient2_order2_var_x(self.array, self.x)

    def time_numpy_gradient1_var_x(self, n, nthread):
        np.gradient(self.array, self.x.squeeze(), axis=-1)

    def time_numpy_gradient1_var_x_outer(self, n, nthread):
        np.gradient(self.array_t, self.x.squeeze(), axis=0)

    # def time_gradient1_order6_dx(self, n, nthread):
    #     cgr.gradient1_order6_dx(self.array, self.dx)

    # def time_gradient2_order6_dx(self, n, nthread):
    #     cgr.gradient2_order6_dx(self.array, self.dx)

    # def time_gradient2_order6_dx(self, n, nthread):
    #     cgr.gradient2_order6_dx(self.array, self.dx)


suite_gradients_cython.params = get_params()
