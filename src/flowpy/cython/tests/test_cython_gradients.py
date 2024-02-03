import numpy as np
import pytest
import flowpy.cython.gradient as gr


def test_gradient_order2_dx():
    """
    Tests the numba second order approximations of the derivatives 
    with constant spacing by testing its exactness to approximately
     machine precision for a second order polynomial
    """
    x = np.linspace(0., 6., 3)
    f = 2.*x**2 + 3.*x + 1.

    f_prime = gr.gradient1_order2_dx(f[None], x[1]-x[0])

    f_prime_sol = 4.*x + 3.

    assert np.allclose(f_prime.squeeze(), f_prime_sol, atol=0, rtol=1e-12)

    f_prime2 = gr.gradient2_order2_dx(f[None], x[1]-x[0])

    f_prime2_sol = np.full_like(x, 4.)
    assert np.allclose(f_prime2.squeeze(), f_prime2_sol, atol=0, rtol=1e-12)

    # check with many values in outer dimension
    f = np.ones((100, 3))*(2.*x**2 + 3.*x + 1.)

    f_prime = gr.gradient1_order2_dx(f, x[1]-x[0])

    f_prime_sol = 4.*x + 3.

    assert np.allclose(f_prime, f_prime_sol, atol=0, rtol=1e-12)

    f_prime2 = gr.gradient2_order2_dx(f, x[1]-x[0])

    f_prime2_sol = np.full_like(x, 4.)
    assert np.allclose(f_prime2, f_prime2_sol, atol=0, rtol=1e-12)


def test_gradient_order2_var_x():
    """
    Tests the numba second order approximations of the derivatives 
    with variable spacing by testing its exactness to approximately
    machine precision for a second order polynomial
    """
    x = np.linspace(0., 6., 3)
    f = 2.*x**2 + 3.*x + 1.

    f_prime = gr.gradient1_order2_var_x(f[None], x)

    f_prime_sol = 4.*x + 3.

    assert np.allclose(f_prime.squeeze(), f_prime_sol, atol=0, rtol=1e-12)

    f_prime2 = gr.gradient2_order2_var_x(f[None], x)

    f_prime2_sol = np.full_like(x, 4.)
    assert np.allclose(f_prime2.squeeze(), f_prime2_sol, atol=0, rtol=1e-12)

    x = np.array([0., 4., 6.])
    f = 2.*x**2 + 3.*x + 1.

    f_prime = gr.gradient1_order2_var_x(f[None], x)

    f_prime_sol = 4.*x + 3.

    assert np.allclose(f_prime.squeeze(), f_prime_sol, atol=0, rtol=1e-12)

    f_prime2 = gr.gradient2_order2_var_x(f[None], x)

    f_prime2_sol = np.full_like(x, 4.)
    assert np.allclose(f_prime2.squeeze(), f_prime2_sol, atol=0, rtol=1e-12)

    # check parallel
    f = np.ones((100, 3))*(2.*x**2 + 3.*x + 1.)

    f_prime = gr.gradient1_order2_var_x(f, x)

    f_prime_sol = 4.*x + 3.

    assert np.allclose(f_prime, f_prime_sol, atol=0, rtol=1e-12)

    f_prime2 = gr.gradient2_order2_var_x(f, x)

    f_prime2_sol = np.full_like(x, 4.)
    assert np.allclose(f_prime2, f_prime2_sol, atol=0, rtol=1e-12)
