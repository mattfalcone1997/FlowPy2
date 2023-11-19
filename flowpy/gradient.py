import numpy as np
import logging

from typing import Callable, Union
from numba import njit

logger = logging.getLogger(__name__)


_first_derivatives = dict()
_second_derivatives = dict()

_default_method = None


def set_default_gradient(name: str):
    if name not in _first_derivatives:
        raise ValueError("Gradient method is invalid")

    _default_method = name


def register_gradient(name: str, first_deriv: Callable,
                      second_deriv: Union[Callable, None] = None):
    if name in _first_derivatives:
        raise ValueError(f"Method {name} already registered")

    _first_derivatives[name] = first_deriv

    if second_deriv is None:
        logger.debug(f"Setting second derivative of {name} "
                     "with first derivative")

        def second_deriv(data, *varargs, axis=None):
            return first_deriv(first_deriv(data, *varargs, axis=axis),
                               *varargs, axis=axis)

    _second_derivatives[name] = second_deriv


def first_derivative(*args, method=None, **kwargs):
    if method is None:
        method = _default_method

    if method not in _first_derivatives:
        raise ValueError("Invalid derivative method")

    return _first_derivatives[method](*args, **kwargs)


def second_derivative(*args, method=None, **kwargs):
    if method is None:
        method = _default_method

    if method not in _second_derivatives:
        raise ValueError("Invalid derivative method")

    return _second_derivatives[method](*args, **kwargs)


register_gradient('numpy', np.gradient)
set_default_gradient('numpy')


@njit(parallel=True)
def numba_gradient1_order2(*args, **kwargs):
    raise NotImplementedError("Not implemented yet")


@njit(parallel=True)
def numba_gradient2_order2(*args, **kwargs):
    raise NotImplementedError("Not implemented yet")


register_gradient('numba2', numba_gradient1_order2,
                  numba_gradient2_order2)


@njit(parallel=True)
def numba_gradient1_order6(*args, **kwargs):
    raise NotImplementedError("Not implemented yet")


@njit(parallel=True)
def numba_gradient2_order6(*args, **kwargs):
    raise NotImplementedError("Not implemented yet")


register_gradient('numba6', numba_gradient1_order6,
                  numba_gradient2_order6)
