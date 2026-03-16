"""Phase 1 tests for GSA-to-irrepify migration.

Tests the new character tables, little group module, and validation module
without requiring GPAW (pure group-theory and unit tests).
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from irrepify.data import character_tables, spglib_to_schoenflies
from irrepify.pointgroup import CharacterTable, SPGOperations
from irrepify.littlegroup import (
    get_little_group,
    get_little_group_factor,
    find_kpoint_in_ibz,
)
from irrepify.validation import representation_matrix
from irrepify.projectables import PolynomialProjectable


# =============================================================================
# Character table completeness and consistency
# =============================================================================

class TestCharacterTables:
    """Verify all 32 crystallographic point groups have character tables."""

    def test_all_32_point_groups_present(self):
        """Every Schoenflies name in spglib_to_schoenflies must have a table."""
        for intl, schoenflies in spglib_to_schoenflies.items():
            assert schoenflies in character_tables, (
                f"Missing character table for {schoenflies} "
                f"(international: {intl})"
            )

    def test_character_table_count(self):
        """We should have at least 32 character tables."""
        assert len(character_tables) >= 32

    @pytest.mark.parametrize("name", list(character_tables.keys()))
    def test_table_structure(self, name):
        """Each table must have matching irreps, classes, and table dims."""
        data = character_tables[name]
        assert "irreps" in data
        assert "classes" in data
        assert "table" in data
        table = np.array(data["table"])
        assert table.shape[0] == len(data["irreps"]), (
            f"{name}: n_irreps mismatch"
        )
        assert table.shape[1] == len(data["classes"]), (
            f"{name}: n_classes mismatch"
        )

    @pytest.mark.parametrize("name", list(character_tables.keys()))
    def test_identity_column(self, name):
        """The identity class (E) should be first, with chi(E) = dim."""
        data = character_tables[name]
        assert data["classes"][0] == "E", f"{name}: first class should be E"
        table = np.array(data["table"])
        # chi(E) should be real and positive (the irrep dimension)
        for i, irrep in enumerate(data["irreps"]):
            dim = table[i, 0]
            assert np.isclose(dim.imag, 0), (
                f"{name}/{irrep}: chi(E) has imaginary part"
            )
            assert dim.real > 0, f"{name}/{irrep}: chi(E) should be positive"

    @pytest.mark.parametrize("name", list(character_tables.keys()))
    def test_orthogonality_rows(self, name):
        """Row orthogonality: sum_g |C_g| chi_i(g)* chi_j(g) = |G| delta_ij.

        This is the Grand Orthogonality Theorem for characters.
        We extract class sizes from the class name prefix (e.g., '2C3' → 2).
        """
        data = character_tables[name]
        table = np.array(data["table"], dtype=complex)
        classes = data["classes"]

        # Parse class sizes from names
        sizes = []
        for cls in classes:
            if cls in ("E", "i"):
                sizes.append(1)
            else:
                # Extract leading number, e.g., "2C3" → 2, "1C2" → 1
                digits = ""
                for ch in cls:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                sizes.append(int(digits) if digits else 1)
        sizes = np.array(sizes)

        # Group order
        order = np.sum(sizes)

        # Check orthogonality
        n_irreps = table.shape[0]
        for i in range(n_irreps):
            for j in range(n_irreps):
                inner = np.sum(sizes * np.conj(table[i]) * table[j])
                expected = order if i == j else 0
                assert np.isclose(inner, expected, atol=1e-10), (
                    f"{name}: row orthogonality failed for "
                    f"irreps ({data['irreps'][i]}, {data['irreps'][j]}): "
                    f"got {inner}, expected {expected}"
                )


class TestNewCharacterTables:
    """Specifically test the 4 newly added character tables."""

    def test_D6_structure(self):
        data = character_tables["D6"]
        assert len(data["irreps"]) == 6
        assert len(data["classes"]) == 6
        # Order should be 12
        table = np.array(data["table"])
        dims = table[:, 0].real
        assert np.isclose(np.sum(dims**2), 12)

    def test_T_structure(self):
        data = character_tables["T"]
        assert len(data["irreps"]) == 4  # A, E(1), E(2), T
        assert len(data["classes"]) == 4
        # Order should be 12
        table = np.array(data["table"])
        dims = table[:, 0].real
        assert np.isclose(np.sum(dims**2), 12)

    def test_Th_structure(self):
        data = character_tables["Th"]
        assert len(data["irreps"]) == 8
        assert len(data["classes"]) == 8
        # Order should be 24
        table = np.array(data["table"])
        dims = table[:, 0].real
        assert np.isclose(np.sum(dims**2), 24)

    def test_O_structure(self):
        data = character_tables["O"]
        assert len(data["irreps"]) == 5
        assert len(data["classes"]) == 5
        # Order should be 24
        table = np.array(data["table"])
        dims = table[:, 0].real
        assert np.isclose(np.sum(dims**2), 24)


# =============================================================================
# Little group tests
# =============================================================================

class TestLittleGroup:
    """Test little group filtering."""

    @pytest.fixture
    def si_spg_ops(self):
        """Silicon primitive cell SPGOperations."""
        atoms = bulk("Si")
        return SPGOperations.from_atoms(atoms, layergroup=False)

    def test_gamma_point_is_full_group(self, si_spg_ops):
        """At Gamma, the little group equals the full point group."""
        little = get_little_group(si_spg_ops, [0, 0, 0])
        assert len(little.W_scc) == len(si_spg_ops.W_scc)

    def test_general_k_reduces_group(self, si_spg_ops):
        """A general k-point should have fewer operations."""
        little = get_little_group(si_spg_ops, [0.123, 0.234, 0.345])
        assert len(little.W_scc) < len(si_spg_ops.W_scc)
        # At minimum, identity should remain
        assert len(little.W_scc) >= 1

    def test_identity_always_present(self, si_spg_ops):
        """Identity should be in the little group for any k-point."""
        for k in [[0, 0, 0], [0.5, 0, 0], [0.1, 0.2, 0.3]]:
            little = get_little_group(si_spg_ops, k)
            eye = np.eye(3, dtype=int)
            has_identity = any(
                np.array_equal(np.round(W).astype(int), eye)
                for W in little.W_scc
            )
            assert has_identity, f"Identity missing for k={k}"

    def test_high_symmetry_point(self, si_spg_ops):
        """X point (0.5, 0, 0.5) in diamond should have a non-trivial group."""
        little = get_little_group(si_spg_ops, [0.5, 0, 0.5])
        # Should have more than just identity
        assert len(little.W_scc) > 1

    def test_factor_group_at_gamma(self, si_spg_ops):
        """Factor group at Gamma should equal the little group (no supercell)."""
        factor = get_little_group_factor(si_spg_ops, [0, 0, 0])
        little = get_little_group(si_spg_ops, [0, 0, 0])
        assert len(factor.W_scc) == len(little.W_scc)


class TestFindKpointInIBZ:
    """Test k-point lookup in IBZ."""

    def test_direct_match(self):
        ibz = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]])
        assert find_kpoint_in_ibz(ibz, [0.5, 0, 0]) == 1

    def test_no_match(self):
        ibz = np.array([[0, 0, 0], [0.5, 0, 0]])
        assert find_kpoint_in_ibz(ibz, [0.3, 0.3, 0.3]) == -1


# =============================================================================
# Validation module tests (using PolynomialProjectable)
# =============================================================================

class TestValidation:
    """Test the validation module using polynomial projectables.

    PolynomialProjectable lets us test the validation machinery without
    GPAW, by checking how coordinate axes transform under known point
    group operations.
    """

    def test_representation_matrix_identity(self):
        """Identity operation should give D = I."""
        cell = np.eye(3) * 5.0
        px = PolynomialProjectable(cell, weights_v=np.array([1, 0, 0]))
        py = PolynomialProjectable(cell, weights_v=np.array([0, 1, 0]))
        pz = PolynomialProjectable(cell, weights_v=np.array([0, 0, 1]))

        D = representation_matrix(
            [px, py, pz],
            op_cc=np.eye(3, dtype=int),
            w_c=np.zeros(3),
        )
        np.testing.assert_allclose(D, np.eye(3), atol=1e-12)

    def test_representation_matrix_inversion(self):
        """Inversion should give D = -I for (x, y, z) basis."""
        cell = np.eye(3) * 5.0
        px = PolynomialProjectable(cell, weights_v=np.array([1, 0, 0]))
        py = PolynomialProjectable(cell, weights_v=np.array([0, 1, 0]))
        pz = PolynomialProjectable(cell, weights_v=np.array([0, 0, 1]))

        D = representation_matrix(
            [px, py, pz],
            op_cc=-np.eye(3, dtype=int),
            w_c=np.zeros(3),
        )
        np.testing.assert_allclose(D, -np.eye(3), atol=1e-12)

    def test_representation_matrix_c4z(self):
        """C4 rotation about z: x→y, y→-x, z→z."""
        cell = np.eye(3) * 5.0
        px = PolynomialProjectable(cell, weights_v=np.array([1, 0, 0]))
        py = PolynomialProjectable(cell, weights_v=np.array([0, 1, 0]))
        pz = PolynomialProjectable(cell, weights_v=np.array([0, 0, 1]))

        # C4 about z in Cartesian: [[0,-1,0],[1,0,0],[0,0,1]]
        # In fractional with cubic cell: same matrix
        C4z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        D = representation_matrix(
            [px, py, pz],
            op_cc=C4z,
            w_c=np.zeros(3),
        )
        # D_mn = <psi_m | R | psi_n>
        # R maps x→-y, y→x (pushing forward: (Rf)(r) = f(R^{-1}r))
        # So R|x> has overlap with y: <y|R|x> = 1, <x|R|x> = 0
        # and R|y> has overlap with x: <x|R|y> = -1
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        np.testing.assert_allclose(D, expected, atol=1e-12)
