import numpy as np
import pytest
import flowpy2.numba.operations as op
import logging

_log = logging.getLogger(__name__)


def test_tdma_validity():
    n = 1000
    ldiag = np.ones(n, dtype='f8')
    ldiag[-1] = 0

    rdiag = ldiag.copy()
    rdiag[0] = 0

    cdiag = np.full(n, -2.)
    cdiag[0] = 1
    cdiag[-1] = 1

    rhs = np.zeros(n)
    rhs[0] = 100
    rhs[-1] = 300

    out = op.tdma_solve(ldiag, cdiag, rdiag, rhs.reshape((1, n)))

    sol = np.linspace(100, 300, n)

    assert np.allclose(
        sol, out, atol=0, rtol=1e-12), \
        "Check simple solution to heat conduction equation order 2"

    large_sol = np.ones((100, n))*sol
    large_rhs = np.ones((100, n))*rhs

    out = op.tdma_solve(ldiag, cdiag, rdiag, large_rhs)
    assert np.allclose(
        large_sol, out, atol=0, rtol=1e-12), \
        "Check parallel solution to heat conduction equation order 2"


def test_tdma_exceptions():
    n = 1000
    ldiag = np.ones(n-1)
    rdiag = np.ones(n)
    cdiag = np.ones(n)

    rhs = np.zeros(n)

    with pytest.raises(ValueError):
        op.tdma_solve(ldiag, cdiag, rdiag, rhs.reshape((1, n)))

    rdiag = np.ones(n-1)
    with pytest.raises(ValueError):
        op.tdma_solve(ldiag, cdiag, rdiag, rhs.reshape((1, n)))

    cdiag = np.ones(n-1)
    with pytest.raises(ValueError):
        op.tdma_solve(ldiag, cdiag, rdiag, rhs.reshape((1, n)))

    rhs = np.zeros(n-1)
    with pytest.raises(ValueError):
        op.tdma_solve(ldiag, cdiag, rdiag, rhs)
