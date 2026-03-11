import numpy as np

def parse_lti_system(A, B, C, D):
    """
    Parses LTI system matrices and converts them to numpy arrays.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    D = np.array(D, dtype=float)

    # Check dimensions
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("Matrix A must be square.")

    if len(B.shape) == 1:
        B = B.reshape(-1, 1)
    if B.shape[0] != n:
        raise ValueError("Matrix B must have the same number of rows as A.")

    if len(C.shape) == 1:
        C = C.reshape(1, -1)
    if C.shape[1] != n:
        raise ValueError("Matrix C must have the same number of columns as A.")

    if len(D.shape) == 1:
        if D.size == 1:
            D = D.reshape(1, 1)
        else:
            raise ValueError("Matrix D dimension mismatch.")
    elif len(D.shape) == 0:
        D = np.zeros((C.shape[0], B.shape[1]))
    if D.shape != (C.shape[0], B.shape[1]):
        raise ValueError("Matrix D dimensions must match C rows and B columns.")

    return A, B, C, D

def controllability_matrix(A, B):
    """
    Computes the controllability matrix C = [B, AB, ..., A^(n-1)B].
    """
    n = A.shape[0]
    C_mat = B
    temp = B
    for _ in range(1, n):
        temp = A @ temp
        C_mat = np.hstack((C_mat, temp))
    return C_mat

def observability_matrix(A, C):
    """
    Computes the observability matrix O = [C; CA; ...; CA^(n-1)].
    """
    n = A.shape[0]
    O_mat = C
    temp = C
    for _ in range(1, n):
        temp = temp @ A
        O_mat = np.vstack((O_mat, temp))
    return O_mat

def check_controllability(A, B):
    """
    Checks if the system (A, B) is controllable.
    """
    n = A.shape[0]
    C_mat = controllability_matrix(A, B)
    rank = np.linalg.matrix_rank(C_mat)
    return rank == n, rank, C_mat

def check_observability(A, C):
    """
    Checks if the system (A, C) is observable.
    """
    n = A.shape[0]
    O_mat = observability_matrix(A, C)
    rank = np.linalg.matrix_rank(O_mat)
    return rank == n, rank, O_mat

def check_asymptotic_stability(A, discrete=False):
    """
    Checks asymptotic stability based on the eigenvalues of A.
    If discrete=False, real parts must be strictly negative.
    If discrete=True, magnitudes must be strictly less than 1.
    """
    eigenvalues = np.linalg.eigvals(A)
    if discrete:
        stable = np.all(np.abs(eigenvalues) < 1.0)
    else:
        stable = np.all(np.real(eigenvalues) < 0.0)
    return stable, eigenvalues.tolist()
