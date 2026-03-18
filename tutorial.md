# irrepify Tutorial

A practical guide to analyzing the symmetry of electronic wavefunctions using
irreducible representations and point group theory.

## What irrepify Does

irrepify identifies the point group symmetry of a molecular or crystalline
structure and decomposes electronic wavefunctions into irreducible
representations (irreps). This tells you, for example, that a particular
molecular orbital of benzene transforms as E1u rather than A1g — information
essential for understanding selection rules, bonding, and spectroscopy.

The package works with:

- **Molecular structures** (via ASE's molecule database or XYZ files)
- **Crystal defects** (via supercell + cluster extraction)
- **Plane-wave DFT wavefunctions** (GPAW)
- **PAW projection coefficients** (GPAW)

## Installation

```bash
pip install -e .
```

Dependencies: `ase`, `numpy`, `spglib`. For DFT wavefunction analysis, you also
need `gpaw`.

## Concepts

### Point Groups and Symmetry Operations

A point group is the set of symmetry operations (rotations, reflections,
inversion) that leave a molecule invariant. For example, water (H2O) has C2v
symmetry: a C2 rotation axis and two mirror planes.

irrepify detects the point group automatically using spglib, then classifies
operations into conjugacy classes (groups of equivalent operations).

### Irreducible Representations

Each irrep is a pattern of how a function transforms under symmetry. A function
that is unchanged by all operations belongs to the totally symmetric irrep (A1
in C2v). The character table encodes these transformation patterns.

### Projectables

A "projectable" is anything you can project onto symmetry operations — a
Cartesian direction, a wavefunction, or PAW projection coefficients. irrepify
computes the overlap ⟨ψ|Rψ⟩ for each operation R, then solves for the irrep
decomposition.

## Quick Start: Analyzing Molecular Symmetry

### Step 1: Detect the Point Group

```python
from ase.build import molecule
from irrepify import SPGOperations, PointGroup

# Build a water molecule in a periodic box
atoms = molecule("H2O")
atoms.center(vacuum=5)
atoms.set_pbc(True)

# Detect symmetry operations
spg_ops = SPGOperations.from_atoms(atoms, layergroup=False)
print(f"Point group: {spg_ops.pointgroup}")  # -> C2v
```

The `layergroup=False` flag tells spglib to use 3D space group detection. Set
`layergroup=True` for 2D systems (slabs, monolayers).

### Step 2: Build the Point Group and Character Table

```python
pg = PointGroup(spg_ops)

# Access the character table
ct = pg.character_table
ct.print()
# Shows the irreps (A1, A2, B1, B2) and their characters
# under each conjugacy class (E, C2, σv_xz, σv_yz)
```

### Step 3: Test Which Irrep a Direction Belongs To

```python
from irrepify import PolynomialProjectable

# The z-axis direction
z = PolynomialProjectable(atoms.cell, weights_v=[0, 0, 1])
signature = pg.signature(z)
weights = pg.detect_irrep(signature)

for irrep, w in zip(ct.irreps, weights):
    if w.real > 0.01:
        print(f"{irrep}: {w.real:.2f}")
# -> A1: 1.00  (z transforms as A1 in C2v)
```

The x and y directions transform as B1 and B2 respectively.

### Step 4: Complex Irreps with Linear Combinations

For groups with complex irreps (C3, C4, C6, etc.), use
`LinearCombinationProjectable`:

```python
from irrepify import LinearCombinationProjectable, PolynomialProjectable

# For a C3 molecule:
x = PolynomialProjectable(atoms.cell, weights_v=[1, 0, 0])
y = PolynomialProjectable(atoms.cell, weights_v=[0, 1, 0])

# x+iy transforms as E(1), x-iy transforms as E(2)
xpiy = LinearCombinationProjectable([1, 1j], [x, y])
xmiy = LinearCombinationProjectable([1, -1j], [x, y])
```

To merge complex conjugate pairs (E(1)+E(2) → E), use:

```python
merged = pg.detect_irrep_merged(signature)
for irrep, weight in merged:
    if weight.real > 0.01:
        print(f"{irrep}: {weight.real:.2f}")
```

## Analyzing DFT Wavefunctions (GPAW)

### Plane-Wave Analysis

```python
from gpaw.new.ase_interface import GPAW
from irrepify import SPGOperations, PointGroup, Projectable
from irrepify.projections import SymmetryEigenvalues

# Run a DFT calculation
calc = GPAW(mode={"name": "pw", "force_complex_dtype": True})
atoms.calc = calc
atoms.get_potential_energy()

# Analyze all bands at once
se = SymmetryEigenvalues.from_calc(calc, layergroup=False, pani=False)
print(se)
# Shows each state's irrep, eigenvalue, and occupation
```

### PAW Projection Analysis

PAW projections (P_ani) provide a cheaper alternative to full plane-wave
overlap. This is especially useful for large systems:

```python
se = SymmetryEigenvalues.from_calc(calc, layergroup=False, pani=True)
print(se.occupied_states)
```

### Manual Band-by-Band Analysis

```python
spg_ops = SPGOperations.from_atoms(calc.atoms, layergroup=False)
pg = PointGroup(spg_ops)

for band in range(6):
    proj = Projectable.from_calc(calc, band)
    sig = pg.signature(proj)
    norm = sig[0].real
    if norm > 1e-10:
        sig = sig / norm
    for irrep, w in pg.detect_irrep_merged(sig):
        if w.real > 0.01:
            print(f"Band {band}: {irrep} (weight={w.real:.2f})")
```

## Working with Crystal Defects

For defect structures in supercells:

1. Create the defect supercell
2. Extract a cluster around the defect site
3. Embed in a large cell matching the local symmetry
4. Analyze using `PaniProjectable` with mapped projections

```python
from irrepify import PaniProjectable

# After DFT calculation on supercell...
kpt = calc.wfs.kpt_u[0]

# Map projections from supercell atoms to cluster atoms
P_ai_cluster = {}
for a in range(len(cluster)):
    supercell_idx = cluster_indices[a]
    if supercell_idx in kpt.P_ani:
        P_ai_cluster[a] = kpt.P_ani[supercell_idx][band]

proj = PaniProjectable(P_ai_cluster, cluster, cluster_setups)
signature = pg.signature(proj)
```

## Cell Setup for Different Symmetries

The unit cell must be compatible with the point group. irrepify handles this via
spglib, but the cell choice matters:

```python
import numpy as np

# For 3-fold or 6-fold symmetry, use a hexagonal cell:
L = 10
angle = 2 * np.pi / 3
c, s = np.cos(angle), np.sin(angle)
cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
atoms.set_cell(cell)
atoms.set_pbc(True)
atoms.center()

# For cubic or orthorhombic, a simple box works:
atoms.center(vacuum=5)
atoms.set_pbc(True)
```

## Saving and Loading Results

```python
from irrepify.projections import SymmetryEigenvalues

# Save to JSON
se.save("symmetry_analysis.json")

# Load back
se_loaded = SymmetryEigenvalues.load("symmetry_analysis.json")
print(se_loaded.occupied_states)
```

## Supported Point Groups

irrepify includes character tables for 28 point groups:

| Family | Groups |
|--------|--------|
| Trivial | C1, Ci, Cs |
| Cyclic | C2, C3, C4, C6 |
| Dihedral | D2, D3, D4 |
| With horizontal mirror | C2h, C3h, C4h, C6h, D2h, D3h, D4h, D6h |
| With vertical mirror | C2v, C3v, C4v, C6v |
| With dihedral mirror | D2d, D3d |
| Improper rotation | S4, S6 |
| Cubic | Td, Oh |

## API Reference

### Key Classes

- **`SPGOperations`**: Detects symmetry from an ASE Atoms object using spglib.
  Use `SPGOperations.from_atoms(atoms, layergroup=bool)`.

- **`PointGroup`**: The main analysis object. Builds conjugacy classes and
  character table. Use `pg.signature(projectable)` and
  `pg.detect_irrep(signature)`.

- **`CharacterTable`**: Stores irrep names, class names, and the character
  matrix. Access via `pg.character_table`.

- **`PolynomialProjectable`**: For testing symmetry of Cartesian directions.
  Takes the cell and a weight vector.

- **`LinearCombinationProjectable`**: For complex linear combinations like x+iy.

- **`Projectable`**: Wraps a GPAW pseudo-wavefunction for plane-wave overlap.

- **`PaniProjectable`**: Wraps PAW projection coefficients P_ani for cheaper
  symmetry analysis.

- **`SymmetryEigenvalues`**: High-level interface that analyzes all bands from a
  GPAW calculation. Use `SymmetryEigenvalues.from_calc(calc, pani=bool)`.

- **`State`**: Represents a single electronic state with its irrep, eigenvalue,
  occupation, and degeneracy.

### Key Parameters

- `layergroup`: Set `True` for 2D systems (monolayers, slabs), `False` for 3D.
- `symprec`: Symmetry detection tolerance (default 0.1 Angstrom). Increase for
  noisy geometries.
- `pani`: Use PAW projections (`True`) or plane-wave overlaps (`False`).

## Troubleshooting

**Wrong point group detected**: Try adjusting `symprec` or check that your cell
is compatible with the expected symmetry. Hexagonal cells are needed for 3-fold
and 6-fold groups.

**Irrep weights not integer**: Small deviations (e.g., 0.98 instead of 1.0) are
normal for pseudo-wavefunctions. Large deviations suggest the wavefunction is
not well-converged or the structure has lower symmetry than expected.

**"Cannot detect all conjugacy classes"**: The detected conjugacy class labels
don't match the character table. This usually means the molecule orientation
doesn't match the expected convention. Try rotating the molecule or using a
different cell.

**Complex irrep pairs**: Groups like C3 have complex conjugate irrep pairs
E(1)/E(2). Use `detect_irrep_merged()` to combine them into a single "E" label
with summed weight.
