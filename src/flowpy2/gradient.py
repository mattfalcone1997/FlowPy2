import numpy as np
import logging
import sympy
from typing import Callable, Union
from numbers import Number
from .numba import gradient

from matplotlib.rcsetup import validate_any
from math import ceil, floor
logger = logging.getLogger(__name__)


_first_derivatives = dict()
_second_derivatives = dict()

_default_method = [None]

def _validate_gradient(val):
    set_default_gradient(val)
    return val

_rc_params = {'default_method': 'numba2'}
_rc_validators = {'default_method': _validate_gradient}

def set_default_gradient(name: str):
    if name not in _first_derivatives:
        raise ValueError("Gradient method is invalid")

    _default_method[0] = name


def register_gradient(name: str, first_deriv: Callable,
                      second_deriv: Union[Callable, None] = None,
                      force: bool = False):
    if name in _first_derivatives and not force:
        raise ValueError(f"Method {name} already registered")

    _first_derivatives[name] = first_deriv

    if second_deriv is None:
        logger.debug(f"Setting second derivative of {name} "
                     "with first derivative")

        def second_deriv(data, *varargs, axis=None):
            return first_deriv(first_deriv(data, *varargs, axis=axis),
                               *varargs, axis=axis)

    _second_derivatives[name] = second_deriv




def return_gradients():
    gradients = list(_first_derivatives.keys())
    i = gradients.index(_default_method[0])
    gradients[i] += " (default)"

    return gradients


def first_derivative(*args, method=None, **kwargs):
    if method is None:
        method = _default_method[0]

    if method not in _first_derivatives:
        raise ValueError(f"Invalid derivative method {method}")

    return _first_derivatives[method](*args, **kwargs)


def second_derivative(*args, method=None, **kwargs):
    if method is None:
        method = _default_method[0]

    if method not in _second_derivatives:
        raise ValueError(f"Invalid derivative method {method}")

    return _second_derivatives[method](*args, **kwargs)


register_gradient('numpy', np.gradient)


def _prepare_parallel(array, axis):
    if array.ndim == 1:
        return array[None]
    else:
        arr = np.moveaxis(array, axis, -1)
        s = np.prod(arr.shape[:-1])
        return arr.reshape((s, arr.shape[-1]))


def _prepare_return(gradient, array_org, axis):
    if array_org.ndim == 1:
        return gradient.squeeze()
    else:
        shape1 = list(array_org.shape)
        del shape1[axis]
        shape1.append(gradient.shape[-1])
        gradient = gradient.reshape(shape1)
        return np.moveaxis(gradient, -1, axis)


def numba_gradient1_order2(array, *varargs, axis=None):

    if len(varargs) == 1:
        arr1 = _prepare_parallel(array, axis=axis)
        if isinstance(varargs[0], Number):
            grad = gradient.gradient1_order2_dx(arr1, varargs[0])
        else:
            grad = gradient.gradient1_order2_var_x(arr1, varargs[0])

        return _prepare_return(grad, array, axis)
    else:
        raise NotImplementedError("Cannot use multiple arrays yet")


def numba_gradient2_order2(array, *varargs, axis=None):
    if len(varargs) == 1:
        arr1 = _prepare_parallel(array, axis=axis)
        if isinstance(varargs[0], Number):
            grad = gradient.gradient2_order2_dx(arr1, varargs[0])
        else:
            grad = gradient.gradient2_order2_var_x(arr1, varargs[0])

        return _prepare_return(grad, array, axis)
    else:
        raise NotImplementedError("Cannot use multiple arrays yet")


register_gradient('numba2', numba_gradient1_order2,
                  numba_gradient2_order2)


def numba_gradient1_order6(array, *varargs, axis=None):
    if len(varargs) == 1:
        arr1 = _prepare_parallel(array, axis=axis)
        if isinstance(varargs[0], Number):
            grad = gradient.gradient1_order6_dx(arr1, varargs[0])
        else:
            grad = gradient.gradient1_order6_var_x(arr1, varargs[0])

        return _prepare_return(grad, array, axis)
    else:
        raise NotImplementedError("Cannot use multiple arrays yet")


def numba_gradient2_order6(array, *varargs, axis=None):
    if len(varargs) == 1:
        arr1 = _prepare_parallel(array, axis=axis)
        if isinstance(varargs[0], Number):
            grad = gradient.gradient2_order6_dx(arr1, varargs[0])
        else:
            grad = gradient.gradient2_order6_var_x(arr1, varargs[0])

        return _prepare_return(grad, array, axis)
    else:
        raise NotImplementedError("Cannot use multiple arrays yet")


register_gradient('numba6', numba_gradient1_order6,
                  numba_gradient2_order6)


def compute_FD_stencil(derivative, rhs_stencil, lhs_stencil=None, subs=None, use_rational=True, simplify=True, method='LU'):
    rhs_stencil = np.array(rhs_stencil)
    if lhs_stencil is None:
        lhs_stencil = [0]
    lhs_stencil = np.array(lhs_stencil)

    if 0 not in lhs_stencil:
        raise ValueError("0 must be in the lhs_stencil if present")

    point_max = max(ceil(lhs_stencil.max()),
                    ceil(rhs_stencil.max()))
    point_min = min(floor(lhs_stencil.min()),
                    floor(rhs_stencil.min()))

    dx_list = [sympy.Symbol("dx%d" % i)
               for i in range(1, 1+point_max-point_min)]

    rhs_h = []
    lhs_h = []
    for rhs in rhs_stencil:
        if rhs < 0:
            start = rhs - point_min
            end = -point_min
            if start != round(start):
                start_ceil = ceil(start)
                if use_rational:
                    val = sympy.Rational(
                        str(start - start_ceil))*dx_list[start_ceil-1]
                else:
                    val = (start - start_ceil)*dx_list[start_ceil-1]
                rhs_h.append(-sum(dx_list[start_ceil:end]) + val)
            else:
                rhs_h.append(-sum(dx_list[start:end]))
        elif rhs == 0:
            rhs_h.append(0)
        else:
            start = -point_min
            end = rhs - point_min
            if end != round(end):
                end_floor = floor(end)
                if use_rational:
                    val = sympy.Rational(
                        str(end - end_floor))*dx_list[end_floor]
                else:
                    val = (end - end_floor)*dx_list[end_floor]
                rhs_h.append(sum(dx_list[start:end_floor]) + val)
            else:
                rhs_h.append(sum(dx_list[start:end]))

    for lhs in lhs_stencil:
        if lhs < 0:
            start = lhs - point_min
            end = -point_min
            lhs_h.append(-sum(dx_list[start:end]))
        elif lhs > 0:
            start = -point_min
            end = lhs - point_min
            lhs_h.append(sum(dx_list[start:end]))

    nrows = len(rhs_stencil) + len(lhs_stencil) - 1
    rows = []

    for i in range(0, nrows):
        cols = []
        for h in rhs_h:
            cols.append(h**(i)/sympy.factorial(i))

        if i >= derivative:
            for h in lhs_h:
                cols.append(-(h**(i-derivative))/sympy.factorial(i-derivative))
        else:
            cols.extend([0]*len(lhs_h))
        rows.append(cols)

    mat = sympy.Matrix(rows)
    rhs_vec = sympy.Matrix([0]*(nrows))
    rhs_vec[derivative] = 1
    sol = mat.solve(rhs_vec, method=method)

    if subs is not None:
        dx_names = [dx.name for dx in dx_list]
        if not all(dx in subs for dx in dx_names):
            raise ValueError(
                "All values must be substituted in subs is present")
        sol = sol.subs(subs)

    return sympy.simplify(sol) if simplify else sol