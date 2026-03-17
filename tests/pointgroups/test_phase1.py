"""Phase 1 tests for GSA-to-irrepify migration.

Tests the new character tables, LittleGroup, and validation module
without requiring GPAW (pure group-theory and unit tests).
"""

import numpy as np
import pytest
from ase.build import bulk

from irrepify.data import character_tables, spglib_to_schoenflies
from irrepify.pointgroup import SPGOperations
from irrepify.littlegroup import LittleGroup, find_kpoint_in_ibz
from irrepify.validation import representation_matrix, check_operation
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
        """chi(E) should be real and positive (the irrep dimension)."""
        data = character_tables[name]
        assert data["classes"][0] == "E"
        table = np.array(data["table"])
        for i, irrep in enumerate(data["irreps"]):
            dim = table[i, 0]
            assert np.isclose(dim.imag, 0), (
                f"{name}/{irrep}: chi(E) has imaginary part"
            )
            assert dim.real > 0, f"{name}/{irrep}: chi(E) should be positive"

    @pytest.mark.parametrize("name", list(character_tables.keys()))
    def test_orthogonality_rows(self, name):
        """Row orthogonality (GOT): sum_g |C_g| chi_i(g)* chi_j(g) = |G| delta_ij."""
        data = character_tables[name]
        table_ig = np.array(data["table"], dtype=complex)
        classes = data["classes"]

        # Parse class sizes from names: "2C3" -> 2, "E" -> 1, "i" -> 1
        sizes_g = []
        for cls in classes:
            if cls in ("E", "i"):
                sizes_g.append(1)
            else:
                digits = ""
                for ch in cls:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                sizes_g.append(int(digits) if digits else 1)
        sizes_g = np.array(sizes_g)
        order = np.sum(sizes_g)

        n_irreps = table_ig.shape[0]
        for i in range(n_irreps):
            for j in range(n_irreps):
                inner = np.sum(sizes_g * np.conj(table_ig[i]) * table_ig[j])
                expected = order if i == j else 0
                assert np.isclose(inner, expected, atol=1e-10), (
                    f"{name}: GOT failed for "
                    f"({data['irreps'][i]}, {data['irreps'][j]}): "
                    f"got {inner}, expected {expected}"
                )


class TestNewCharacterTables:
    """Specifically test the 4 newly added character tables."""

    @pytest.mark.parametrize("name,n_irreps,order", [
        ("D6", 6, 12),
        ("T", 4, 12),
        ("Th", 8, 24),
        ("O", 5, 24),
    ])
    def test_structure_and_order(self, name, n_irreps, order):
        data = character_tables[name]
        assert len(data["irreps"]) == n_irreps
        assert len(data["classes"]) == len(data["irreps"])
        table_ig = np.array(data["table"])
        dims_i = table_ig[:, 0].real
        assert np.isclose(np.sum(dims_i**2), order)


# =============================================================================
# LittleGroup tests
# =============================================================================

class TestLittleGroup:

    @pytest.fixture
    def si_spg_ops(self):
        """Silicon primitive cell SPGOperations."""
        return SPGOperations.from_atoms(bulk("Si"), layergroup=False)

    def test_gamma_is_full_group(self, si_spg_ops):
        """At Gamma, the little group equals the full group."""
        lg = LittleGroup.from_spg_ops(si_spg_ops, kpt_c=[0, 0, 0])
        assert lg.n_ops == len(si_spg_ops.W_scc)

    def test_general_k_reduces(self, si_spg_ops):
        """A general k-point should have fewer operations."""
        lg = LittleGroup.from_spg_ops(si_spg_ops, kpt_c=[0.123, 0.234, 0.345])
        assert lg.n_ops < len(si_spg_ops.W_scc)
        assert lg.n_ops >= 1

    def test_identity_always_present(self, si_spg_ops):
        """Identity should be in G_k for any k-point."""
        eye_cc = np.eye(3, dtype=int)
        for kpt_c in [[0, 0, 0], [0.5, 0, 0], [0.1, 0.2, 0.3]]:
            lg = LittleGroup.from_spg_ops(si_spg_ops, kpt_c=kpt_c)
            has_identity = any(
                np.array_equal(np.round(W_cc).astype(int), eye_cc)
                for W_cc in lg.spg_ops.W_scc
            )
            assert has_identity, f"Identity missing for k={kpt_c}"

    def test_high_symmetry_point(self, si_spg_ops):
        """X point in diamond should have a non-trivial little group."""
        lg = LittleGroup.from_spg_ops(si_spg_ops, kpt_c=[0.5, 0, 0.5])
        assert lg.n_ops > 1

    def test_factor_group_at_gamma(self, si_spg_ops):
        """Factor group at Gamma equals the little group (no supercell)."""
        lg = LittleGroup.from_spg_ops(si_spg_ops, kpt_c=[0, 0, 0])
        fg = lg.factor_group()
        assert fg.n_ops == lg.n_ops

    def test_kpt_c_stored(self, si_spg_ops):
        kpt_c = [0.25, 0.0, 0.25]
        lg = LittleGroup.from_spg_ops(si_spg_ops, kpt_c=kpt_c)
        np.testing.assert_allclose(lg.kpt_c, kpt_c)


class TestFindKpointInIBZ:

    def test_direct_match(self):
        kpts_kc = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]])
        assert find_kpoint_in_ibz(kpts_kc, kpt_c=[0.5, 0, 0]) == 1

    def test_no_match(self):
        kpts_kc = np.array([[0, 0, 0], [0.5, 0, 0]])
        assert find_kpoint_in_ibz(kpts_kc, kpt_c=[0.3, 0.3, 0.3]) == -1


# =============================================================================
# Validation module tests (using PolynomialProjectable)
# =============================================================================

class TestValidation:
    """Test the validation module using polynomial projectables."""

    @pytest.fixture
    def xyz_projectables(self):
        """Orthonormal (x, y, z) basis in a cubic cell."""
        cell_cv = np.eye(3) * 5.0
        px = PolynomialProjectable(cell_cv, weights_v=np.array([1, 0, 0]))
        py = PolynomialProjectable(cell_cv, weights_v=np.array([0, 1, 0]))
        pz = PolynomialProjectable(cell_cv, weights_v=np.array([0, 0, 1]))
        return [px, py, pz]

    def test_identity_gives_unit_matrix(self, xyz_projectables):
        D_nn = representation_matrix(
            xyz_projectables,
            W_cc=np.eye(3, dtype=int),
            w_c=np.zeros(3),
        )
        np.testing.assert_allclose(D_nn, np.eye(3), atol=1e-12)

    def test_inversion_gives_minus_identity(self, xyz_projectables):
        D_nn = representation_matrix(
            xyz_projectables,
            W_cc=-np.eye(3, dtype=int),
            w_c=np.zeros(3),
        )
        np.testing.assert_allclose(D_nn, -np.eye(3), atol=1e-12)

    def test_c4z_representation(self, xyz_projectables):
        """C4 about z: pushing-forward convention gives D = R itself."""
        C4z_cc = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        D_nn = representation_matrix(
            xyz_projectables, W_cc=C4z_cc, w_c=np.zeros(3),
        )
        # (Rf)(r) = f(R^{-1}r): D_mn = <m|R|n> = R_mn in Cartesian
        expected_nn = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        np.testing.assert_allclose(D_nn, expected_nn, atol=1e-12)

    def test_check_operation_identity(self, xyz_projectables):
        D_nn, error_n = check_operation(
            xyz_projectables,
            W_cc=np.eye(3, dtype=int),
            w_c=np.zeros(3),
        )
        assert np.all(error_n < 1e-10)
        np.testing.assert_allclose(D_nn, np.eye(3), atol=1e-12)
