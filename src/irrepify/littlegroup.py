"""Little group utilities for k-point symmetry analysis.

Given an SPGOperations object and a k-point, this module filters for the
little group G_k — the subset of space group operations {R|t} that leave
the k-point invariant (modulo a reciprocal lattice vector):

    R^{-T} k = k + G,   G integer

For supercells, the module can also compute the factor group G_k / T_inner,
where T_inner are the inner translations (primitive lattice translations
that are not supercell lattice translations).
"""

import numpy as np

from .pointgroup import SPGOperations


def get_little_group(spg_ops, k_point, tolerance=1e-5):
    """Filter SPGOperations for the little group of a k-point.

    Parameters
    ----------
    spg_ops : SPGOperations
        Full space group operations.
    k_point : array-like
        The k-point in fractional reciprocal coordinates.
    tolerance : float
        Tolerance for k-point equivalence (mod reciprocal lattice).

    Returns
    -------
    SPGOperations
        New SPGOperations containing only operations in the little group.
    """
    k = np.asarray(k_point, dtype=float)
    keep = []

    for s, (W_cc, w_c) in enumerate(zip(spg_ops.W_scc, spg_ops.w_sc)):
        # In fractional coordinates, the reciprocal-space rotation is R^{-T}.
        # For integer rotation matrices, R^{-T} = (R^{-1})^T.
        R_inv = np.round(np.linalg.inv(W_cc)).astype(int)
        k_rotated = R_inv.T @ k
        diff = k_rotated - k
        if np.allclose(diff, np.round(diff), atol=tolerance):
            keep.append(s)

    if not keep:
        # At minimum, identity should always be in the little group
        raise ValueError(
            f"No operations found in little group of k={k_point}. "
            "Check that the k-point is in fractional coordinates."
        )

    W_scc = np.array([spg_ops.W_scc[s] for s in keep])
    w_sc = np.array([spg_ops.w_sc[s] for s in keep])

    return SPGOperations(
        atoms=spg_ops.atoms,
        W_scc=W_scc,
        w_sc=w_sc,
        origin_shift_c=spg_ops.origin_shift_c,
        cell_cv=spg_ops.cell_cv,
        pointgroup=spg_ops.pointgroup,
        allow_translations=True,
    )


def get_little_group_factor(spg_ops, k_point, tolerance=1e-5):
    """Compute the factor group G_k / T_inner for supercell analysis.

    For supercells, multiple operations may share the same rotation matrix
    but differ by inner translations. This function keeps one representative
    per unique rotation (the one with the smallest translation norm),
    effectively computing the point group of the little group.

    Parameters
    ----------
    spg_ops : SPGOperations
        Full space group operations (typically from a supercell).
    k_point : array-like
        The k-point in fractional reciprocal coordinates.
    tolerance : float
        Tolerance for k-point equivalence.

    Returns
    -------
    SPGOperations
        Factor group with one representative per unique rotation.
    """
    little = get_little_group(spg_ops, k_point, tolerance)

    # Group by rotation matrix, keep representative with smallest translation
    unique_ops = {}
    for W_cc, w_c in zip(little.W_scc, little.w_sc):
        key = tuple(W_cc.flatten())
        w_norm = np.linalg.norm(w_c % 1.0)
        if key not in unique_ops or w_norm < unique_ops[key][1]:
            unique_ops[key] = ((W_cc, w_c), w_norm)

    W_list = [op for (op, _) in unique_ops.values()]
    W_scc = np.array([W for W, w in W_list])
    w_sc = np.array([w for W, w in W_list])

    return SPGOperations(
        atoms=spg_ops.atoms,
        W_scc=W_scc,
        w_sc=w_sc,
        origin_shift_c=spg_ops.origin_shift_c,
        cell_cv=spg_ops.cell_cv,
        pointgroup=spg_ops.pointgroup,
        allow_translations=True,
    )


def find_kpoint_in_ibz(ibz_kpoints, k_target, rotations=None, tolerance=1e-5):
    """Find a k-point in the irreducible Brillouin zone.

    First checks for a direct match. If rotations are provided, also checks
    whether any symmetry-equivalent k-point is in the IBZ.

    Parameters
    ----------
    ibz_kpoints : ndarray, shape (n_kpts, 3)
        IBZ k-points in fractional coordinates.
    k_target : array-like
        Target k-point in fractional coordinates.
    rotations : ndarray, shape (n_sym, 3, 3), optional
        Rotation matrices (fractional coordinates) to check equivalences.
    tolerance : float
        Tolerance for k-point matching.

    Returns
    -------
    int
        Index into ibz_kpoints, or -1 if not found.
    """
    k = np.asarray(k_target, dtype=float)

    # Direct match
    for i, k_ibz in enumerate(ibz_kpoints):
        if np.allclose(k, k_ibz, atol=tolerance):
            return i

    # Check symmetry-equivalent points
    if rotations is not None:
        for i, k_ibz in enumerate(ibz_kpoints):
            for R in rotations:
                R_inv = np.round(np.linalg.inv(R)).astype(int)
                k_rot = R_inv.T @ k_ibz
                diff = k_rot - k
                if np.allclose(diff, np.round(diff), atol=tolerance):
                    return i

    return -1
