import numpy as np
import pytest
from kalman.linear_systems import parse_lti_system, check_controllability, check_observability, check_asymptotic_stability

def test_parse_lti_system():
    A = [[1, 2], [3, 4]]
    B = [1, 1]
    C = [1, 0]
    D = 0

    Ap, Bp, Cp, Dp = parse_lti_system(A, B, C, D)

    assert Ap.shape == (2, 2)
    assert Bp.shape == (2, 1)
    assert Cp.shape == (1, 2)
    assert Dp.shape == (1, 1)

def test_controllability():
    # Controllable system
    A = np.array([[0, 1], [-2, -3]])
    B = np.array([[0], [1]])

    controllable, rank, C_mat = check_controllability(A, B)
    assert controllable == True
    assert rank == 2

    # Uncontrollable system
    A_un = np.array([[1, 0], [0, 2]])
    B_un = np.array([[1], [0]])

    controllable, rank, _ = check_controllability(A_un, B_un)
    assert controllable == False
    assert rank == 1

def test_observability():
    # Observable system
    A = np.array([[0, 1], [-2, -3]])
    C = np.array([[1, 0]])

    observable, rank, O_mat = check_observability(A, C)
    assert observable == True
    assert rank == 2

    # Unobservable system
    A_un = np.array([[1, 0], [0, 2]])
    C_un = np.array([[0, 1]])

    observable, rank, _ = check_observability(A_un, C_un)
    assert observable == False
    assert rank == 1

def test_asymptotic_stability():
    # Stable continuous
    A = np.array([[-1, 0], [0, -2]])
    stable, eig = check_asymptotic_stability(A, discrete=False)
    assert stable == True

    # Unstable continuous
    A_un = np.array([[1, 0], [0, -2]])
    stable, eig = check_asymptotic_stability(A_un, discrete=False)
    assert stable == False

    # Stable discrete
    A_d = np.array([[0.5, 0], [0, 0.8]])
    stable, eig = check_asymptotic_stability(A_d, discrete=True)
    assert stable == True

    # Unstable discrete
    A_d_un = np.array([[1.5, 0], [0, 0.8]])
    stable, eig = check_asymptotic_stability(A_d_un, discrete=True)
    assert stable == False
