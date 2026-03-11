from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List, Optional, Tuple

from kalman.linear_systems import parse_lti_system, check_controllability, check_observability, check_asymptotic_stability
from kalman.feedback_observers import place_poles
from kalman.optimal_control import lqr_continuous, lqr_discrete
from kalman.filtering import kalman_filter, simulate_system_with_noise

app = FastAPI(title="Kalman API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    A: List[List[float]]
    B: List[List[float]]
    C: List[List[float]]
    D: List[List[float]]
    discrete: bool = False

class LQRRequest(BaseModel):
    A: List[List[float]]
    B: List[List[float]]
    Q: List[List[float]]
    R: List[List[float]]
    discrete: bool = False

from typing import Any

class FeedbackRequest(BaseModel):
    A: List[List[float]]
    B: List[List[float]]
    poles: List[Any] # Support real or complex poles parsed in logic
    discrete: bool = False

class KalmanRequest(BaseModel):
    A: List[List[float]]
    B: List[List[float]]
    C: List[List[float]]
    D: List[List[float]]
    Q: List[List[float]]
    R: List[List[float]]
    x0: List[float]
    P0: List[List[float]]
    time_steps: int
    seed: Optional[int] = None

@app.post("/api/analyze")
def analyze_system(req: AnalyzeRequest):
    try:
        A, B, C, D = parse_lti_system(req.A, req.B, req.C, req.D)

        controllable, c_rank, _ = check_controllability(A, B)
        observable, o_rank, _ = check_observability(A, C)
        stable, eigenvalues = check_asymptotic_stability(A, req.discrete)

        minimal = controllable and observable

        return {
            "controllable": bool(controllable),
            "controllability_rank": int(c_rank),
            "observable": bool(observable),
            "observability_rank": int(o_rank),
            "stable": bool(stable),
            "eigenvalues": [[e.real, e.imag] for e in eigenvalues],
            "minimal": bool(minimal)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/feedback")
def solve_feedback(req: FeedbackRequest):
    try:
        A = np.array(req.A, dtype=float)
        B = np.array(req.B, dtype=float)

        # Scipy place_poles can take real or complex pairs
        # Parse inputs appropriately
        poles = []
        for p in req.poles:
            if isinstance(p, dict): # if passed as {real: x, imag: y} from JS
                poles.append(complex(p.get("real", 0), p.get("imag", 0)))
            elif isinstance(p, list) and len(p) == 2:
                poles.append(complex(p[0], p[1]))
            else:
                poles.append(complex(p, 0))

        K = place_poles(A, B, poles)

        # Compute eigenvalues before and after
        eig_open = np.linalg.eigvals(A)
        eig_closed = np.linalg.eigvals(A - B @ K)

        return {
            "K": K.tolist(),
            "eig_open": [[e.real, e.imag] for e in eig_open],
            "eig_closed": [[e.real, e.imag] for e in eig_closed]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/lqr")
def solve_lqr(req: LQRRequest):
    try:
        if req.discrete:
            K, P = lqr_discrete(req.A, req.B, req.Q, req.R)
        else:
            K, P = lqr_continuous(req.A, req.B, req.Q, req.R)

        return {
            "K": K.tolist(),
            "P": P.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/kalman")
def run_kalman(req: KalmanRequest):
    try:
        A, B, C, D = parse_lti_system(req.A, req.B, req.C, req.D)
        Q = np.array(req.Q, dtype=float)
        R = np.array(req.R, dtype=float)
        x0 = np.array(req.x0, dtype=float)
        P0 = np.array(req.P0, dtype=float)

        # Simulation
        u = np.zeros((req.time_steps, B.shape[1] if len(B.shape)>1 else 1)) # zero input

        x_true, y_meas = simulate_system_with_noise(
            A, B, C, D, x0, u, req.time_steps, Q, R, req.seed
        )

        # Estimation
        x_est, P_est = kalman_filter(A, B, C, Q, R, np.zeros_like(x0), P0, u, y_meas)

        return {
            "x_true": x_true.tolist(),
            "y_meas": y_meas.tolist(),
            "x_est": x_est.tolist(),
            "P_est": [p.tolist() for p in P_est]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
