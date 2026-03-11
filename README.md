# Kalman

Kalman is a web-based computational workbench built for SF2832 Mathematical Systems Theory. It provides numerical solvers and interactive visualizations for analyzing linear dynamical systems, synthesizing state feedback controllers, and implementing optimal observers (Kalman Filters).

The tool features a Dark Space Mission Control design, built entirely with Tailwind CSS and the Preline UI component library, ensuring a responsive, accessible, and highly technical aesthetic.

## 📚 Syllabus Mapping (SF2832)

This project strictly adheres to the course learning outcomes:

| Module | Syllabus Topic | Implemented Features |
| :--- | :--- | :--- |
| **Analysis** | Controllability and observability | Automated matrix rank verification for reachability ($\mathcal{C}$) and observability ($\mathcal{O}$) Gramians. |
| **Stability** | Systems of linear differential eq. | Eigenvalue decomposition to check for asymptotic and marginal stability in continuous/discrete time. |
| **Control** | Feedback, pole placement | Synthesizes state-feedback gain matrices ($K$) to arbitrarily assign closed-loop poles using Ackermann's formula. |
| **Optimal** | Linear quadratic control | Continuous Algebraic Riccati Equation (CARE) solver for LQR optimization. |
| **Estimation** | Observers and Kalman filters | Luenberger observer design and Discrete-Time Kalman Filter simulation for state estimation under noise. |

## 🚀 Deployment (Vercel)

Kalman is designed to run as a serverless mathematical engine.

1. Fork this repository.
2. Deploy to Vercel (Python runtime is auto-detected).
3. Access the Systems Dashboard at `https://your-kalman.vercel.app`.

## 📊 Visualizations & Artifacts

### 1. State Estimation (The Kalman Filter)
Simulates a dynamical system subjected to process and measurement noise, comparing the true hidden state $x$ against the observer's estimated state $\hat{x}$.

*Figure 1: Kalman Filter Tracking. The Plotly.js graph displays the noisy measurement scatter alongside the smoothed, optimal state estimation. The convergence of the error covariance matrix $P$ demonstrates the filter's steady-state performance.*

### 2. Closed-Loop Stability (Pole Placement)
Visualizing the eigenvalues of the system matrix $A$ before and after applying state feedback $u = -Kx$.

*Figure 2: Pole-Zero Map. The visualization shows the migration of system poles. For continuous systems, poles must be moved to the left half-plane ($Re < 0$); for discrete systems, they must be placed inside the unit circle ($\|z\| < 1$) to guarantee asymptotic stability.*
