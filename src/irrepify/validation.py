"""Numerical validation of symmetry analysis.

Validates that wavefunctions transform correctly under symmetry operations
by comparing the direct transformation R|psi> against the projected
reconstruction D(R)|psi>.  If the wavefunction truly spans a representation
of the little group, both methods agree to machine precision.

Works with any :class:`Projectable` subclass (plane-wave, PAW, polynomial).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class OperationResult:
    """Validation result for a single symmetry operation.

    Attributes
    ----------
    D_nn : np.array
        Representation matrix D(R), shape (nbands, nbands).
    error_n : np.array
        Per-band relative errors, shape (nbands,).
    """

    D_nn: np.array
    error_n: np.array

    @property
    def max_error(self):
        return float(np.max(self.error_n))

    def is_valid(self, tol=1e-6):
        return self.max_error < tol


@dataclass
class ValidationResult:
    """Validation result for an entire little group.

    Attributes
    ----------
    results_s : list[OperationResult]
        Per-operation results, indexed by operation.
    """

    results_s: list[OperationResult]

    @property
    def D_snn(self):
        """Representation matrices, shape (nsym, nbands, nbands)."""
        return np.array([r.D_nn for r in self.results_s])

    @property
    def worst_error(self):
        return max(r.max_error for r in self.results_s)

    def is_valid(self, tol=1e-6):
        return all(r.is_valid(tol) for r in self.results_s)

    def __repr__(self):
        n_ops = len(self.results_s)
        return (f"ValidationResult({n_ops} ops, "
                f"worst_error={self.worst_error:.2e})")


def representation_matrix(projectables_n, W_cc, w_c):
    """Build the representation matrix D(R) for a degenerate subspace.

    D_nn = S_nn^{-1} O_nn, where:
        S_mn = <psi_m | psi_n>           (overlap)
        O_mn = <psi_m | R | psi_n>       (transformed overlap)

    Parameters
    ----------
    projectables_n : list of Projectable
        Wavefunctions spanning the degenerate subspace.
    W_cc : ndarray (3, 3)
        Rotation matrix in fractional coordinates.
    w_c : ndarray (3,)
        Translation vector in fractional coordinates.

    Returns
    -------
    D_nn : ndarray, shape (n, n)
        Representation matrix.
    """
    n = len(projectables_n)
    S_nn = np.zeros((n, n), dtype=complex)
    O_nn = np.zeros((n, n), dtype=complex)

    transformed_n = [p.operation(W_cc, w_c=w_c) for p in projectables_n]

    for m in range(n):
        for k in range(n):
            S_nn[m, k] = projectables_n[m].dot(projectables_n[k])
            O_nn[m, k] = projectables_n[m].dot(transformed_n[k])

    try:
        return np.linalg.solve(S_nn, O_nn)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(S_nn) @ O_nn


def check_operation(projectables_n, W_cc, w_c):
    """Validate a single symmetry operation on a degenerate subspace.

    Compares <psi_m | R | psi_k> (direct) against (S D)_mk (projected).

    Parameters
    ----------
    projectables_n : list of Projectable
        Wavefunctions spanning the degenerate subspace.
    W_cc : ndarray (3, 3)
        Rotation matrix in fractional coordinates.
    w_c : ndarray (3,)
        Translation vector in fractional coordinates.

    Returns
    -------
    OperationResult
    """
    n = len(projectables_n)
    D_nn = representation_matrix(projectables_n, W_cc, w_c)

    transformed_n = [p.operation(W_cc, w_c=w_c) for p in projectables_n]

    error_n = np.zeros(n)
    for k in range(n):
        direct_m = np.array([
            projectables_n[m].dot(transformed_n[k]) for m in range(n)
        ])

        S_mn = np.array([
            [projectables_n[m].dot(projectables_n[j]) for j in range(n)]
            for m in range(n)
        ])
        projected_m = S_mn @ D_nn[:, k]

        diff_m = direct_m - projected_m
        norm = np.linalg.norm(direct_m)
        error_n[k] = np.linalg.norm(diff_m) / norm if norm > 0 else 0.0

    return OperationResult(D_nn=D_nn, error_n=error_n)


def validate_little_group(projectables_n, spg_ops, tol=1e-6, verbose=False):
    """Validate all operations in a little group.

    Parameters
    ----------
    projectables_n : list of Projectable
        Wavefunctions spanning the degenerate subspace.
    spg_ops : SPGOperations
        The little group operations.
    tol : float
        Relative error threshold per operation.
    verbose : bool
        Print per-operation diagnostics.

    Returns
    -------
    ValidationResult
    """
    results_s = []
    for s, (W_cc, w_c) in enumerate(zip(spg_ops.W_scc, spg_ops.w_sc)):
        result = check_operation(projectables_n, W_cc, w_c)
        results_s.append(result)

        if verbose:
            status = "PASS" if result.is_valid(tol) else "FAIL"
            print(f"  Op {s}: max_error={result.max_error:.2e} [{status}]")

    validation = ValidationResult(results_s=results_s)

    if verbose:
        n_valid = sum(r.is_valid(tol) for r in results_s)
        print(f"\n  {n_valid}/{len(results_s)} operations passed "
              f"(worst: {validation.worst_error:.2e})")

    return validation
