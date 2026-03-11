import numpy as np
import scipy.linalg

def lqr_continuous(A, B, Q, R):
    """
    Solves the continuous-time linear quadratic regulator (LQR) problem.
    Finds the optimal state-feedback matrix K such that the control law
    u(t) = -Kx(t) minimizes the cost function J = integral(x^T Q x + u^T R u) dt.

    This is done by solving the Continuous Algebraic Riccati Equation (CARE):
    A^T P + P A - P B R^-1 B^T P + Q = 0
    Then computing K = R^-1 B^T P.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    Q = np.array(Q, dtype=float)
    R = np.array(R, dtype=float)

    # Solve Continuous Algebraic Riccati Equation
    try:
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)

        # Compute the LQR gain matrix K
        # K = R^-1 B^T P
        K = np.linalg.inv(R) @ B.T @ P

        return K, P
    except Exception as e:
        raise ValueError(f"LQR solution failed: {str(e)}")

def lqr_discrete(A, B, Q, R):
    """
    Solves the discrete-time linear quadratic regulator (LQR) problem.
    Finds the optimal state-feedback matrix K such that the control law
    u[k] = -Kx[k] minimizes the cost function J = sum(x[k]^T Q x[k] + u[k]^T R u[k]).

    This is done by solving the Discrete Algebraic Riccati Equation (DARE):
    P = A^T P A - (A^T P B) (R + B^T P B)^-1 (B^T P A) + Q
    Then computing K = (R + B^T P B)^-1 (B^T P A).
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    Q = np.array(Q, dtype=float)
    R = np.array(R, dtype=float)

    # Solve Discrete Algebraic Riccati Equation
    try:
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)

        # Compute the LQR gain matrix K
        # K = (R + B^T P B)^-1 (B^T P A)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        return K, P
    except Exception as e:
        raise ValueError(f"Discrete LQR solution failed: {str(e)}")
