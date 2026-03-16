"""Numerical validation of symmetry analysis.

Validates that wavefunctions transform correctly under symmetry operations
by comparing two independent methods:

Method 1 (Direct): Apply the symmetry operation R to the wavefunction,
    producing R|psi>.
Method 2 (Projected): Build the representation matrix D(R) from overlaps
    and reconstruct R|psi> as D(R) @ |psi>.

If the wavefunction truly forms a representation of the little group,
both methods should agree to machine precision.

This module works with irrepify's Projectable interface, so it supports
both plane-wave and PAW projector methods.
"""

import numpy as np


def representation_matrix(projectables, op_cc, w_c):
    """Build the representation matrix D(R) for a degenerate subspace.

    D_mn(R) = <psi_m | R | psi_n>  (with metric correction)

    For an orthonormal basis, D = O. For a non-orthonormal basis
    (e.g. pseudo-wavefunctions), D = S^{-1} O where S is the overlap.

    Parameters
    ----------
    projectables : list of Projectable
        Wavefunctions spanning the degenerate subspace.
    op_cc : ndarray (3, 3)
        Rotation matrix in fractional coordinates.
    w_c : ndarray (3,)
        Translation vector in fractional coordinates.

    Returns
    -------
    ndarray, shape (n, n)
        Representation matrix D(R).
    """
    n = len(projectables)
    S = np.zeros((n, n), dtype=complex)
    O = np.zeros((n, n), dtype=complex)

    transformed = [p.operation(op_cc, w_c=w_c) for p in projectables]

    for m in range(n):
        for k in range(n):
            S[m, k] = projectables[m].dot(projectables[k])
            O[m, k] = projectables[m].dot(transformed[k])

    try:
        return np.linalg.solve(S, O)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(S) @ O


def check_symmetry_precision(projectables, op_cc, w_c, tolerance=1e-6,
                              verbose=False):
    """Validate that wavefunctions transform correctly under a symmetry op.

    Compares the direct transformation R|psi> against the projected
    transformation D(R)|psi> = sum_j D_mj |psi_j>. If R is truly in the
    little group, R|psi> lies entirely within the degenerate subspace and
    both methods agree.

    Parameters
    ----------
    projectables : list of Projectable
        Wavefunctions spanning the degenerate subspace.
    op_cc : ndarray (3, 3)
        Rotation matrix in fractional coordinates.
    w_c : ndarray (3,)
        Translation vector in fractional coordinates.
    tolerance : float
        Relative error threshold for validity.
    verbose : bool
        If True, print detailed diagnostics.

    Returns
    -------
    dict
        Contains 'is_valid', 'relative_error', 'per_band_errors', and the
        representation matrix 'D_matrix'.
    """
    n = len(projectables)
    D = representation_matrix(projectables, op_cc, w_c)

    # For each band, compare <psi_m | R | psi_n> (direct)
    # against sum_j D_mj <psi_m | psi_j> (projected)
    transformed = [p.operation(op_cc, w_c=w_c) for p in projectables]

    per_band_errors = []
    for band_n in range(n):
        # Direct overlaps: <psi_m | R | psi_n> for all m
        direct = np.array([
            projectables[m].dot(transformed[band_n]) for m in range(n)
        ])

        # Projected: sum_j D_mj <psi_m | psi_j>
        S_row = np.array([
            [projectables[m].dot(projectables[j]) for j in range(n)]
            for m in range(n)
        ])
        projected = S_row @ D[:, band_n]

        diff = direct - projected
        norm = np.linalg.norm(direct)
        abs_error = np.linalg.norm(diff)
        rel_error = abs_error / norm if norm > 0 else abs_error

        per_band_errors.append({
            'band': band_n,
            'abs_error': abs_error,
            'rel_error': rel_error,
        })

        if verbose:
            print(f"  Band {band_n}: abs_error={abs_error:.2e}, "
                  f"rel_error={rel_error:.2e}")

    max_rel_error = max(e['rel_error'] for e in per_band_errors)
    is_valid = max_rel_error < tolerance

    if verbose:
        status = "PASS" if is_valid else "FAIL"
        print(f"  Overall: max_rel_error={max_rel_error:.2e} [{status}]")

    return {
        'is_valid': is_valid,
        'relative_error': max_rel_error,
        'per_band_errors': per_band_errors,
        'D_matrix': D,
    }


def validate_little_group(projectables, spg_ops, tolerance=1e-6,
                           verbose=False):
    """Validate all operations in a little group against a degenerate subspace.

    Parameters
    ----------
    projectables : list of Projectable
        Wavefunctions spanning the degenerate subspace.
    spg_ops : SPGOperations
        The little group operations.
    tolerance : float
        Relative error threshold for each operation.
    verbose : bool
        If True, print per-operation diagnostics.

    Returns
    -------
    dict
        Contains 'all_valid' (bool), 'results' (list of per-op dicts),
        'worst_error' (float), and 'D_matrices' (list of representation
        matrices).
    """
    results = []
    D_matrices = []

    for s, (W_cc, w_c) in enumerate(zip(spg_ops.W_scc, spg_ops.w_sc)):
        if verbose:
            print(f"Operation {s}:")
        result = check_symmetry_precision(
            projectables, W_cc, w_c, tolerance=tolerance, verbose=verbose
        )
        results.append(result)
        D_matrices.append(result['D_matrix'])

    worst_error = max(r['relative_error'] for r in results)
    all_valid = all(r['is_valid'] for r in results)

    if verbose:
        n_valid = sum(r['is_valid'] for r in results)
        print(f"\nValidation: {n_valid}/{len(results)} operations passed "
              f"(worst error: {worst_error:.2e})")

    return {
        'all_valid': all_valid,
        'results': results,
        'worst_error': worst_error,
        'D_matrices': D_matrices,
    }
