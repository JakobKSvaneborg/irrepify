"""Little group of a k-point.

The little group G_k is the subset of space group operations {R|t} that
leave a k-point invariant modulo a reciprocal lattice vector:

    R^{-T} k = k + G,   G integer

This module provides :class:`LittleGroup`, which wraps an
:class:`SPGOperations` filtered to G_k. For supercells it can also
compute the factor group G_k / T_inner.
"""

from dataclasses import dataclass

import numpy as np

from .pointgroup import SPGOperations


@dataclass
class LittleGroup:
    """The little group of a k-point.

    Attributes
    ----------
    kpt_c : np.array
        The k-point in fractional reciprocal coordinates.
    spg_ops : SPGOperations
        Space group operations restricted to G_k.
    keep_s : np.array
        Indices into the parent SPGOperations that survive the filter.
    """

    kpt_c: np.array
    spg_ops: SPGOperations
    keep_s: np.array

    @classmethod
    def from_spg_ops(cls, spg_ops, kpt_c, tol=1e-5):
        """Filter SPGOperations for the little group of kpt_c.

        Parameters
        ----------
        spg_ops : SPGOperations
            Full space group operations.
        kpt_c : array-like
            k-point in fractional reciprocal coordinates, shape (3,).
        tol : float
            Tolerance for k-point equivalence (mod reciprocal lattice).
        """
        kpt_c = np.asarray(kpt_c, dtype=float)
        keep = []

        for s, W_cc in enumerate(spg_ops.W_scc):
            # Reciprocal-space rotation is W^{-T}.
            # For integer W_cc: W^{-T} = (W^{-1})^T.
            W_inv_cc = np.round(np.linalg.inv(W_cc)).astype(int)
            kpt_rot_c = W_inv_cc.T @ kpt_c
            diff_c = kpt_rot_c - kpt_c
            if np.allclose(diff_c, np.round(diff_c), atol=tol):
                keep.append(s)

        if not keep:
            raise ValueError(
                f"No operations found in little group of k={kpt_c}. "
                "Check that the k-point is in fractional coordinates."
            )

        keep_s = np.array(keep)
        W_scc = np.array([spg_ops.W_scc[s] for s in keep_s])
        w_sc = np.array([spg_ops.w_sc[s] for s in keep_s])

        little_ops = SPGOperations(
            atoms=spg_ops.atoms,
            W_scc=W_scc,
            w_sc=w_sc,
            origin_shift_c=spg_ops.origin_shift_c,
            cell_cv=spg_ops.cell_cv,
            pointgroup=spg_ops.pointgroup,
            allow_translations=True,
        )
        return cls(kpt_c=kpt_c, spg_ops=little_ops, keep_s=keep_s)

    @property
    def n_ops(self):
        """Number of operations in G_k."""
        return len(self.keep_s)

    def factor_group(self):
        """Compute the factor group G_k / T_inner.

        For supercells, multiple operations may share the same rotation
        but differ by inner translations.  This keeps one representative
        per unique rotation (smallest translation norm), giving the
        point group of the little group.

        Returns
        -------
        LittleGroup
            Factor group with one representative per unique rotation.
        """
        ops = self.spg_ops
        unique = {}  # key: flattened W_cc -> (index, ||w||)
        for s, (W_cc, w_c) in enumerate(zip(ops.W_scc, ops.w_sc)):
            key = tuple(W_cc.flatten())
            w_norm = np.linalg.norm(w_c % 1.0)
            if key not in unique or w_norm < unique[key][1]:
                unique[key] = (s, w_norm)

        keep = sorted(idx for idx, _ in unique.values())
        W_scc = np.array([ops.W_scc[s] for s in keep])
        w_sc = np.array([ops.w_sc[s] for s in keep])

        factor_ops = SPGOperations(
            atoms=ops.atoms,
            W_scc=W_scc,
            w_sc=w_sc,
            origin_shift_c=ops.origin_shift_c,
            cell_cv=ops.cell_cv,
            pointgroup=ops.pointgroup,
            allow_translations=True,
        )
        return LittleGroup(
            kpt_c=self.kpt_c,
            spg_ops=factor_ops,
            keep_s=np.array(keep),
        )


def find_kpoint_in_ibz(kpts_kc, kpt_c, W_scc=None, tol=1e-5):
    """Find a k-point in the irreducible Brillouin zone.

    Checks for a direct match first.  If rotation matrices are given,
    also checks symmetry-equivalent k-points.

    Parameters
    ----------
    kpts_kc : ndarray, shape (nkpts, 3)
        IBZ k-points in fractional coordinates.
    kpt_c : array-like, shape (3,)
        Target k-point in fractional coordinates.
    W_scc : ndarray, shape (nsym, 3, 3), optional
        Rotation matrices (fractional) to check equivalences.
    tol : float
        Tolerance for k-point matching.

    Returns
    -------
    int
        Index into kpts_kc, or -1 if not found.
    """
    kpt_c = np.asarray(kpt_c, dtype=float)

    for k, kpt_ibz_c in enumerate(kpts_kc):
        if np.allclose(kpt_c, kpt_ibz_c, atol=tol):
            return k

    if W_scc is not None:
        for k, kpt_ibz_c in enumerate(kpts_kc):
            for W_cc in W_scc:
                W_inv_cc = np.round(np.linalg.inv(W_cc)).astype(int)
                kpt_rot_c = W_inv_cc.T @ kpt_ibz_c
                diff_c = kpt_rot_c - kpt_c
                if np.allclose(diff_c, np.round(diff_c), atol=tol):
                    return k

    return -1
