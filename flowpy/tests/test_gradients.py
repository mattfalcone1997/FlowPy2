import pytest
import numpy as np
from flowpy import gradient


def test_register_gradient():

    gradient.register_gradient('numpy2', np.gradient)


def test_default_setter():

    gradient.set_default_gradient('numpy')

    with pytest.raises(ValueError):
        gradient.set_default_gradient('numpy3')

    gradient.register_gradient('numpy3', np.gradient)
    gradient.set_default_gradient('numpy3')


def test_first_derivative():
    array = np.linspace(0, 100)
    dx = 1.

    gradient.set_default_gradient('numpy')
    gradient.first_derivative(array, dx)

    gradient.first_derivative(array, dx, method='numpy', axis=0)
    gradient.first_derivative(array, dx, method='numba2', axis=0)

    gradient.first_derivative(array, dx, method='numba6', axis=0)


def test_second_derivative():
    array = np.linspace(0, 100)
    dx = 1.

    gradient.set_default_gradient('numpy')
    gradient.second_derivative(array, dx)

    gradient.second_derivative(array, dx, method='numpy', axis=0)
    gradient.second_derivative(array, dx, method='numba2', axis=0)

    gradient.second_derivative(array, dx, method='numba6', axis=0)
    gradient.second_derivative(array, dx, method='numba6', axis=0)
