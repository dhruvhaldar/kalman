import numpy as np

def kalman_filter(A, B, C, Q, R, x0, P0, u, y):
    """
    Implements a discrete-time Kalman filter.

    Inputs:
    A: State transition matrix (n x n)
    B: Control input matrix (n x m)
    C: Observation matrix (p x n)
    Q: Process noise covariance (n x n)
    R: Measurement noise covariance (p x p)
    x0: Initial state estimate (n x 1)
    P0: Initial error covariance (n x n)
    u: Control inputs (T x m)
    y: Measurements (T x p)

    Returns:
    x_est: Estimated states (T x n)
    P_est: Estimated error covariances (T x n x n)
    """
    T = y.shape[0]
    n = A.shape[0]

    x_est = np.zeros((T, n))
    P_est = np.zeros((T, n, n))

    x = x0.reshape(-1, 1) if x0.ndim == 1 else x0
    P = P0

    for t in range(T):
        # 1. Predict (Time Update)
        u_t = u[t].reshape(-1, 1) if u.ndim > 1 else np.array([[u[t]]])
        x_pred = A @ x + B @ u_t
        P_pred = A @ P @ A.T + Q

        # 2. Update (Measurement Update)
        y_t = y[t].reshape(-1, 1) if y.ndim > 1 else np.array([[y[t]]])

        # Kalman gain: K = P_pred * C^T * (C * P_pred * C^T + R)^-1
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)

        # Update estimate with measurement
        y_tilde = y_t - C @ x_pred  # Innovation
        x = x_pred + K @ y_tilde

        # Update covariance: P = (I - K * C) * P_pred
        I = np.eye(n)
        P = (I - K @ C) @ P_pred

        # Store estimates
        x_est[t, :] = x.flatten()
        P_est[t, :, :] = P

    return x_est, P_est

def simulate_system_with_noise(A, B, C, D, x0, u, time_steps, Q, R, seed=None):
    """
    Simulates a discrete-time linear system with process and measurement noise.

    x[k+1] = A x[k] + B u[k] + w[k],   w ~ N(0, Q)
    y[k]   = C x[k] + D u[k] + v[k],   v ~ N(0, R)
    """
    if seed is not None:
        np.random.seed(seed)

    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    D = np.array(D, dtype=float)
    Q = np.array(Q, dtype=float)
    R = np.array(R, dtype=float)

    n_states = A.shape[0]
    n_outputs = C.shape[0]
    n_inputs = B.shape[1] if len(B.shape) > 1 else 1

    x = np.array(x0, dtype=float).flatten()

    # Process inputs
    if u is None:
        u = np.zeros((time_steps, n_inputs))
    elif len(u.shape) == 1:
        u = u.reshape(-1, 1)

    x_true = np.zeros((time_steps, n_states))
    y_meas = np.zeros((time_steps, n_outputs))

    for t in range(time_steps):
        u_t = u[t]

        # Store true state
        x_true[t, :] = x

        # Generate measurement with noise
        v = np.random.multivariate_normal(np.zeros(n_outputs), R)
        y_meas[t, :] = C @ x + D @ u_t + v

        # Advance state with process noise
        w = np.random.multivariate_normal(np.zeros(n_states), Q)
        x = A @ x + B @ u_t + w

    return x_true, y_meas
