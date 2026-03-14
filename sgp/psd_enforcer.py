"""
Higham (2002) alternating projections to the nearest positive semi-definite matrix.

Reference: Higham, N.J. (2002). "Computing the nearest correlation matrix — a
problem from finance." IMA Journal of Numerical Analysis, 22, 329-343.

The algorithm alternates between two projections:
  1. Project onto the set of symmetric matrices with unit diagonal (S)
  2. Project onto the cone of positive semi-definite matrices (P+)

Convergence is guaranteed and is linear (each step reduces the Frobenius
distance to the feasible set).
"""
from __future__ import annotations

import numpy as np
import structlog

from engines.errors import DartsEngineError

logger = structlog.get_logger(__name__)

_DEFAULT_MAX_ITER = 100
_DEFAULT_TOL = 1e-8


def higham_psd(
    rho: np.ndarray,
    max_iter: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
    eigenvalue_floor: float = 0.0,
) -> np.ndarray:
    """
    Project a symmetric matrix onto the nearest positive semi-definite matrix.

    Uses Higham (2002) alternating projections.

    Parameters
    ----------
    rho:
        Symmetric matrix to project.  Need not be positive semi-definite.
        Must be square with values in [-1, 1] (correlation matrix).
    max_iter:
        Maximum number of alternating projection iterations.
    tol:
        Convergence tolerance on the Frobenius norm of the iterative update.
    eigenvalue_floor:
        Minimum eigenvalue to enforce (default 0 for PSD, use small positive
        value like 1e-8 to get a strictly positive definite result).

    Returns
    -------
    np.ndarray
        Nearest PSD correlation matrix (unit diagonal, symmetric).

    Raises
    ------
    DartsEngineError
        If the input is not square or not symmetric, or if the algorithm
        fails to converge.
    """
    n = rho.shape[0]
    if rho.ndim != 2 or rho.shape[1] != n:
        raise DartsEngineError(
            f"Input must be a square 2D array, got shape {rho.shape}"
        )

    # Check approximate symmetry
    if not np.allclose(rho, rho.T, atol=1e-6):
        logger.warning("higham_psd_asymmetric_input", max_asymmetry=float(np.max(np.abs(rho - rho.T))))
        rho = (rho + rho.T) / 2.0

    # Check if already PSD (fast path)
    if _is_psd(rho, floor=eigenvalue_floor):
        logger.debug("higham_psd_already_psd", n=n)
        return rho.copy()

    Y = rho.copy().astype(np.float64)
    dS = np.zeros_like(Y)  # Dykstra correction

    for iteration in range(max_iter):
        R = Y - dS

        # Projection 1: onto PSD cone (via eigendecomposition)
        X = _project_psd(R, floor=eigenvalue_floor)

        # Update Dykstra correction
        dS = X - R

        # Projection 2: onto correlation matrix space (unit diagonal)
        Y_new = _project_unit_diagonal(X)

        # Check convergence
        change = float(np.linalg.norm(Y_new - Y, ord="fro"))
        Y = Y_new

        if change < tol:
            logger.debug(
                "higham_psd_converged",
                iterations=iteration + 1,
                change=round(change, 10),
            )
            break
    else:
        # Did not converge strictly — check if result is acceptable
        final_min_eig = float(np.min(np.linalg.eigvalsh(Y)))
        if final_min_eig < -1e-5:
            raise DartsEngineError(
                f"Higham PSD projection did not converge in {max_iter} iterations. "
                f"Minimum eigenvalue: {final_min_eig:.6f}"
            )
        logger.warning(
            "higham_psd_max_iter_reached",
            max_iter=max_iter,
            final_min_eig=round(final_min_eig, 8),
        )

    # Final cleanup: enforce exact unit diagonal and symmetry
    Y = _project_unit_diagonal(Y)
    Y = (Y + Y.T) / 2.0

    return Y


def _project_psd(matrix: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """
    Project a symmetric matrix onto the PSD cone.

    Replaces negative eigenvalues with `floor`.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues_clipped = np.maximum(eigenvalues, floor)
    return eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T


def _project_unit_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Set diagonal to 1 and clip off-diagonal to [-1, 1]."""
    result = matrix.copy()
    n = result.shape[0]
    np.fill_diagonal(result, 1.0)
    # Clip off-diagonal values to correlation range
    for i in range(n):
        for j in range(n):
            if i != j:
                result[i, j] = max(-1.0, min(1.0, result[i, j]))
    return result


def _is_psd(matrix: np.ndarray, floor: float = 0.0, tol: float = 1e-8) -> bool:
    """Return True if the matrix is positive semi-definite."""
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvalues >= floor - tol))
    except np.linalg.LinAlgError:
        return False


def nearest_psd_correlation(
    matrix: np.ndarray,
    eigenvalue_floor: float = 1e-8,
) -> np.ndarray:
    """
    Convenience wrapper: return the nearest PSD correlation matrix.

    Enforces strict positive definiteness by using a small eigenvalue floor.
    """
    return higham_psd(matrix, eigenvalue_floor=eigenvalue_floor)
