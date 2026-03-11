import numpy as np
import scipy.signal

def place_poles(A, B, poles):
    """
    Computes the state-feedback gain matrix K such that the closed-loop system
    matrix (A - BK) has the specified eigenvalues (poles).
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    poles = np.array(poles, dtype=complex)

    # Check controllability first
    # Using scipy.signal.place_poles which implements the robust pole placement algorithm
    try:
        res = scipy.signal.place_poles(A, B, poles)
        return res.gain_matrix
    except Exception as e:
        raise ValueError(f"Pole placement failed: {str(e)}")

def luenberger_observer(A, C, poles):
    """
    Computes the observer gain matrix L such that the observer error dynamics
    matrix (A - LC) has the specified eigenvalues (poles).
    Since the eigenvalues of (A - LC) are the same as (A^T - C^T L^T), we can use
    pole placement on (A^T, C^T) to find L^T, then transpose the result.
    """
    A = np.array(A, dtype=float)
    C = np.array(C, dtype=float)
    poles = np.array(poles, dtype=complex)

    # We want eigenvalues of A - LC to be `poles`
    # (A - LC)^T = A^T - C^T L^T
    # So we place poles for system (A^T, C^T)
    try:
        res = scipy.signal.place_poles(A.T, C.T, poles)
        return res.gain_matrix.T
    except Exception as e:
        raise ValueError(f"Observer pole placement failed: {str(e)}")

def simulate_system_feedback(A, B, C, D, K, x0, u_ref, time_steps, dt=0.1, discrete=False):
    """
    Simulates a linear system with state feedback u = -Kx + u_ref.
    If continuous, simulates using Euler integration.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    D = np.array(D, dtype=float)
    K = np.array(K, dtype=float)
    x = np.array(x0, dtype=float)
    u_ref = np.array(u_ref, dtype=float)

    n_states = A.shape[0]
    n_outputs = C.shape[0]

    states = np.zeros((time_steps, n_states))
    outputs = np.zeros((time_steps, n_outputs))

    for t in range(time_steps):
        u = -K @ x + u_ref

        states[t, :] = x
        outputs[t, :] = C @ x + D @ u

        if discrete:
            x = A @ x + B @ u
        else:
            dx = A @ x + B @ u
            x = x + dx * dt

    return states, outputs
