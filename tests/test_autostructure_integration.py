"""Integration tests for the AutoStructureBuilder."""

import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk
from pymatgen.core import Structure, Lattice
from unittest.mock import MagicMock, patch

from src.autostructure.builder import AutoStructureBuilder
from src.autostructure.interface import InterfaceBuilder
from src.autostructure.molecular import MolecularGenerator
from src.autostructure.ionic import IonicGenerator
from src.autostructure.preopt import PreOptimizer

@pytest.fixture
def dummy_structure():
    """Creates a dummy MgO structure."""
    return Structure(
        Lattice.cubic(4.2),
        ["Mg", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )

@pytest.fixture
def dummy_atoms():
    """Creates dummy ASE atoms."""
    return bulk("Cu", "fcc", a=3.6)

def test_builder_initialization():
    """Test AutoStructureBuilder initialization."""
    # MPRester is imported inside a method in intelligence.py, so we patch the class where it ends up
    # or just ensure we don't trigger it.
    # We just want to check if builder initializes.
    builder = AutoStructureBuilder(api_key="dummy_key")
    assert builder.intelligence is not None
    assert builder.intelligence.api_key == "dummy_key"

def test_interface_builder_strain_limit():
    """Test that InterfaceBuilder skips structures with high strain."""
    # Create two incompatible structures
    # A: Lattice 3.0
    atoms_a = bulk("Fe", "bcc", a=3.0)
    # B: Lattice 5.0 (Huge mismatch > 15%)
    atoms_b = bulk("Fe", "bcc", a=5.0)

    lj_params = {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}
    builder = InterfaceBuilder(atoms_a, atoms_b, lj_params=lj_params)

    # Run strain scan
    builder.epitaxial_strain_scan()

    # Should be empty because strain > 15%
    assert len(builder.generated_structures) == 0

def test_interface_builder_success():
    """Test that InterfaceBuilder works for compatible structures."""
    # Compatible
    atoms_a = bulk("Fe", "bcc", a=2.87)
    atoms_b = bulk("Fe", "bcc", a=2.90) # Small mismatch

    lj_params = {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}
    builder = InterfaceBuilder(atoms_a, atoms_b, lj_params=lj_params)
    builder.epitaxial_strain_scan()

    assert len(builder.generated_structures) > 0
    # Check if meta is correct
    assert builder.generated_structures[0].info["type"] == "interface_epitaxial"

def test_preoptimizer_calculator():
    """Test PreOptimizer calculator factory."""
    lj_params = {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}
    opt = PreOptimizer(lj_params=lj_params)

    # Test EMT elements
    atoms_emt = bulk("Cu")
    calc = opt.get_calculator(atoms_emt)
    from ase.calculators.emt import EMT
    assert isinstance(calc, EMT)

    # Test LJ fallback
    atoms_lj = bulk("Si") # Si not in EMT set
    calc = opt.get_calculator(atoms_lj)
    from ase.calculators.lj import LennardJones
    assert isinstance(calc, LennardJones)

def test_ionic_generator_md_snapshots():
    """Test IonicGenerator MD snapshots with explicit calculator."""
    # Create a simple structure that supports EMT (e.g. Cu, mimicking ionic for test)
    # Using Cu just to pass EMT check, though IonicGenerator expects charged things.
    # The generator logic doesn't strictly enforce charge for MD, just structure.
    struct = Structure(Lattice.cubic(3.6), ["Cu"], [[0,0,0]])

    lj_params = {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}
    gen = IonicGenerator(struct, lj_params=lj_params)

    # Patch run_pre_optimization to avoid actual expensive relaxation
    # but ensure it returns atoms with calculator if asked.
    # Actually, let's just run it, it's fast for single atom.

    # We need to mock MD run to save time?
    # Langevin running 50 steps is fast.

    structures = gen.generate_all()

    # Check if MD snapshots are generated
    md_snaps = [s for s in gen.generated_structures if s.info.get("type") == "md_snapshot"]
    assert len(md_snaps) > 0
    # Check if calculator was attached (though it might not be serialized in info)
    # The test ensures the code didn't crash.

def test_molecular_high_pressure():
    """Test MolecularGenerator high pressure packing note/behavior."""
    # Create dummy molecular crystal (H2)
    # H2 in a box
    struct = Structure(Lattice.cubic(5.0), ["H", "H"], [[0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])

    lj_params = {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}
    gen = MolecularGenerator(struct, lj_params=lj_params)
    # LOWER mic_distance because H-H bond is ~0.74A, default limit is 0.8A
    gen.pre_optimizer.mic_distance = 0.5

    # Patch _get_molecules to return the single molecule
    with patch.object(gen, '_get_molecules', return_value=[[0, 1]]):
        gen.generate_high_pressure_packing()

    # Check results
    hp_structs = [s for s in gen.generated_structures if s.info.get("type") == "high_pressure_packing"]
    assert len(hp_structs) == 3
    # Check volumes are smaller
    vol_orig = struct.volume
    for s in hp_structs:
        assert s.get_volume() < vol_orig

def test_ionic_generator_charged_defects(dummy_structure):
    """Test charged defect generation."""
    lj_params = {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}
    gen = IonicGenerator(dummy_structure, lj_params=lj_params)
    gen.generate_charged_defects()

    defects = [s for s in gen.generated_structures if s.info.get("type") == "charged_defect"]
    assert len(defects) > 0
    # Check charge info
    assert "charge" in defects[0].info
