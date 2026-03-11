import numpy as np
import pytest
from kalman.optimal_control import lqr_continuous, lqr_discrete
from kalman.feedback_observers import place_poles
from kalman.filtering import kalman_filter

def test_lqr_continuous():
    A = np.array([[0, 1], [-2, -3]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.eye(1)

    K, P = lqr_continuous(A, B, Q, R)

    # Check dimensions
    assert K.shape == (1, 2)
    assert P.shape == (2, 2)

    # Check positive definiteness of P
    assert np.all(np.linalg.eigvals(P) > 0)

    # Check that A - BK is stable
    A_cl = A - B @ K
    assert np.all(np.real(np.linalg.eigvals(A_cl)) < 0)

def test_lqr_discrete():
    A = np.array([[0.5, 1], [0, 0.5]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.eye(1)

    K, P = lqr_discrete(A, B, Q, R)

    # Check dimensions
    assert K.shape == (1, 2)
    assert P.shape == (2, 2)

    # Check positive definiteness of P
    assert np.all(np.linalg.eigvals(P) > 0)

    # Check that A - BK is stable (eigenvalues strictly inside unit circle)
    A_cl = A - B @ K
    assert np.all(np.abs(np.linalg.eigvals(A_cl)) < 1.0)

def test_place_poles():
    A = np.array([[0, 1], [-2, -3]])
    B = np.array([[0], [1]])

    poles = [-1, -2]
    K = place_poles(A, B, poles)

    assert K.shape == (1, 2)

    A_cl = A - B @ K
    eig_cl = np.linalg.eigvals(A_cl)

    # Sort for comparison
    eig_cl.sort()
    poles.sort()

    np.testing.assert_almost_equal(eig_cl, poles)

def test_kalman_filter():
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [0.1]])
    C = np.array([[1, 0]])

    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1

    x0 = np.array([0, 0])
    P0 = np.eye(2)

    u = np.zeros(10)
    y = np.ones((10, 1))

    x_est, P_est = kalman_filter(A, B, C, Q, R, x0, P0, u, y)

    assert x_est.shape == (10, 2)
    assert P_est.shape == (10, 2, 2)
