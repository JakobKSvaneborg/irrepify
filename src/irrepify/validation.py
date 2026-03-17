"""Numerical validation of symmetry analysis.

Validates that wavefunctions transform correctly under symmetry operations
by comparing the direct transformation R|psi> against the projected
reconstruction D(R)|psi>.  If the wavefunction truly spans a representation
of the little group, both methods agree to machine precision.

Works with any Projectable subclass (plane-wave, PAW, polynomial).
"""

import numpy as np


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

    Compares <psi_m | R | psi_k> (direct) against (S D)_mk (projected)
    and returns the per-band relative error.

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
    error_n : ndarray, shape (n,)
        Per-band relative error.
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

    return D_nn, error_n


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
    D_snn : ndarray, shape (nsym, n, n)
        Representation matrices for each operation.
    error_sn : ndarray, shape (nsym, n)
        Per-band relative errors for each operation.
    """
    D_list = []
    error_list = []

    for s, (W_cc, w_c) in enumerate(zip(spg_ops.W_scc, spg_ops.w_sc)):
        D_nn, error_n = check_operation(projectables_n, W_cc, w_c)
        D_list.append(D_nn)
        error_list.append(error_n)

        if verbose:
            max_err = np.max(error_n)
            status = "PASS" if max_err < tol else "FAIL"
            print(f"  Op {s}: max_error={max_err:.2e} [{status}]")

    D_snn = np.array(D_list)
    error_sn = np.array(error_list)

    if verbose:
        n_valid = np.sum(np.max(error_sn, axis=1) < tol)
        worst = np.max(error_sn)
        print(f"\n  {n_valid}/{len(D_list)} operations passed "
              f"(worst: {worst:.2e})")

    return D_snn, error_sn
