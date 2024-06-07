import pytest
import numpy as np

from scipy.integrate import (simpson,
                             cumulative_trapezoid)
from flowpy2 import integrate
from numbers import Number


def test_register_integrations():

    integrate.register_integration('scipy_simps2', simpson)

    assert 'scipy_simps2' in integrate._integrations

    integrate.register_cumulat_integration('scipy_cumtrapz2',
                                           cumulative_trapezoid)

    assert 'scipy_cumtrapz2' in integrate._cumulative_integrations


def test_default_setter():
    integrate.set_default_integration('scipy_simps')

    with pytest.raises(ValueError):
        integrate.set_default_integration('scipy_simps3')

    integrate.register_integration('scipy_simps3', simpson)
    integrate.set_default_integration('scipy_simps3')

    integrate.set_default_cumulat_integration('scipy_cumtrapz')

    with pytest.raises(ValueError):
        integrate.set_default_cumulat_integration('scipy_cumtrapz3')

    integrate.register_cumulat_integration('scipy_cumtrapz3',
                                           cumulative_trapezoid)
    integrate.set_default_cumulat_integration('scipy_cumtrapz3')


def test_integrate():
    array = np.linspace(0, 100)
    dx = 1.

    integrate.set_default_integration('scipy_simps')

    x1 = integrate.integrate(array, dx=dx)
    x2 = integrate.integrate(array, x=array)

    assert isinstance(x1, Number)
    assert isinstance(x2, Number)

    array1 = np.random.randn(10, 100, 20)
    x3 = integrate.integrate(array1, dx=dx, axis=1)

    assert x3.shape == (array1.shape[0], array1.shape[2])


def test_cumulative_integrate():
    array = np.linspace(0, 100)
    dx = 1.

    integrate.set_default_cumulat_integration('scipy_cumtrapz')

    x1 = integrate.cumulative_integrate(array, dx=dx, initial=0)
    x2 = integrate.cumulative_integrate(array, x=array, initial=0)

    assert x1.shape == array.shape
    assert x2.shape == array.shape

    array1 = np.random.randn(10, 100, 20)
    x3 = integrate.cumulative_integrate(array1, dx=dx, axis=1, initial=0)

    assert x3.shape == array1.shape


def test_integrate_validity():
    array = np.linspace(0, 100, 101)
    dx = 1.

    x1 = integrate.integrate(array, dx=dx)
    x1_correct = 0.5*100*100

    assert np.isclose(x1, x1_correct, atol=0, rtol=1e-12)

    x1_array = integrate.cumulative_integrate(array, dx=dx, initial=0)
    x1_array_correct = 0.5*array*array

    assert np.allclose(x1_array, x1_array_correct, atol=0, rtol=1e-12)
