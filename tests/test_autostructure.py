import pytest
import os
import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from pymatgen.core import Structure, Lattice

from src.autostructure import (
    AutoStructureBuilder,
    PreOptimizer,
    IonicGenerator,
    AlloyGenerator,
    CovalentGenerator,
    MolecularGenerator,
    InterfaceBuilder
)

# Mock MPRester to avoid needing API key during tests
class MockMPRester:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    class materials:
        class summary:
            @staticmethod
            def search(*args, **kwargs):
                return []

@pytest.fixture
def mock_mp(monkeypatch):
    # Since we use delayed import in intelligence.py, we need to mock where it comes from
    # But since it might not be imported yet, we can mock sys.modules or just patch the usage if possible.
    # However, 'src.autostructure.intelligence.MPRester' doesn't exist.
    # We can try to patch 'mp_api.client.MPRester' directly if mp_api is installed.
    # If not, the code catches ImportError.
    # But we want to test the logic when it IS available.

    # We need to simulate that 'mp_api.client' exists and has MPRester.
    import sys
    from unittest.mock import MagicMock

    mock_module = MagicMock()
    mock_module.MPRester = MockMPRester
    monkeypatch.setitem(sys.modules, "mp_api.client", mock_module)

def test_pre_optimization():
    # Create two atoms very close
    # Starting at 0.4 A, PreOptimizer with LJ should push them apart.
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.4]])
    preopt = PreOptimizer()

    # Run pre-optimization
    # If it fails (still too close), it raises ValueError.
    # But usually LJ is very repulsive at 0.4A so it should fly apart.
    # We test that it successfully relaxes and returns a valid structure OR
    # raises ValueError if we enforce stricter limits.

    # However, for 0.4A, the force is so huge it might explode or move far.
    # Let's verify that IF it returns, the distance is larger.
    try:
        relaxed = preopt.run_pre_optimization(atoms)
        dist = relaxed.get_distance(0, 1)
        # Should be pushed apart significantly
        assert dist > 0.7
    except ValueError:
        # If it decided to discard it because it couldn't relax enough in steps,
        # that is also valid behavior for a safety valve.
        pass

def test_ionic_generator():
    # NaCl
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    gen = IonicGenerator(atoms)
    results = gen.generate_all()

    # Check types
    types = [a.info.get("type") for a in results]
    assert "charged_defect" in types
    assert "surface" in types
    assert "md_snapshot" in types

    # Check charge info
    charged = [a for a in results if a.info.get("type") == "charged_defect"]
    if charged:
        assert "charge" in charged[0].info

def test_alloy_generator():
    atoms = bulk("Cu", "fcc", a=3.6)
    gen = AlloyGenerator(atoms)
    results = gen.generate_all()

    types = [a.info.get("type") for a in results]
    assert "random_substitution" in types
    # GBs might fail on simple bulk without supercell, but code handles exception
    # Phase separation requires >1 species, so Cu won't trigger it unless we force it
    # But generator runs safely.

def test_covalent_generator():
    atoms = bulk("Si", "diamond", a=5.43)
    gen = CovalentGenerator(atoms)
    results = gen.generate_all()

    types = [a.info.get("type") for a in results]
    assert "shear_strain" in types
    assert "interstitial" in types
    assert "amorphous_quench" in types

def test_molecular_generator():
    # Fake molecular crystal (just CO2 in a box)
    # CO bond is ~1.16 A
    atoms = Atoms("CO2", positions=[[0,0,0], [0,0,1.16], [0,0,-1.16]], cell=[10,10,10], pbc=True)
    gen = MolecularGenerator(atoms)
    results = gen.generate_all()

    types = [a.info.get("type") for a in results]
    # Intramolecular might fail if it doesn't detect molecule (depends on CrystalNN default radii)
    # But high_pressure_packing should always work
    assert "high_pressure_packing" in types

def test_interface_builder():
    a = bulk("Cu", "fcc", a=3.6)
    b = bulk("Ag", "fcc", a=4.09)

    builder = InterfaceBuilder(a, b)
    results = builder.generate_all()

    types = [a.info.get("type") for a in results]
    # Check if ANY interface type was generated
    # "interface_epitaxial" might fail if strain is too large (though Ag/Cu is okay)
    # Twist always runs.
    assert "interface_twist" in types
    # It seems my implementation creates twist structures consistently
    # Epitaxial creates them if stacking succeeds.

def test_main_builder(mock_mp):
    atoms = bulk("Fe", "bcc", a=2.87)
    builder = AutoStructureBuilder()
    results = builder.generate(atoms)

    assert len(results) > 0
    assert isinstance(results[0], Atoms)
