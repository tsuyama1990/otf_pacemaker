"""Tests for Phase 4: Scenario-Driven Generation and MACE Pre-optimization."""

import pytest
import numpy as np
from ase import Atoms
from unittest.mock import MagicMock, patch

from src.generation.optimizer import FoundationOptimizer
from src.generation.scenarios import (
    ScenarioFactory,
    InterfaceGenerator,
    SurfaceGenerator,
    DefectGenerator,
)


@pytest.fixture
def mock_mace_calc():
    """Mock MACE calculator."""
    calc = MagicMock()
    calc.get_potential_energy.return_value = -10.0
    calc.get_forces.return_value = np.zeros((2, 3))
    return calc


@pytest.fixture
def mock_lbfgs():
    """Mock LBFGS optimizer."""
    with patch("src.generation.optimizer.LBFGS") as mock:
        yield mock


def test_foundation_optimizer_init():
    """Test FoundationOptimizer initialization."""
    with patch("src.generation.optimizer.mace_mp"):
        opt = FoundationOptimizer(model="small")
        assert opt.fmax == 0.1
        assert opt.steps == 50


def test_foundation_optimizer_relax(mock_mace_calc, mock_lbfgs):
    """Test relax method."""
    with patch("src.generation.optimizer.mace_mp", return_value=mock_mace_calc):
        opt = FoundationOptimizer()

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.5]])
        relaxed = opt.relax([atoms])

        assert len(relaxed) == 1
        # Check if LBFGS was called
        mock_lbfgs.assert_called()
        mock_lbfgs.return_value.run.assert_called_with(fmax=0.1, steps=50)


def test_interface_generator():
    """Test InterfaceGenerator."""
    config = {
        "type": "interface",
        "substrate": {"symbol": "MgO", "structure": "rocksalt", "lattice": 4.21},
        "layer": {"symbol": "Fe", "structure": "bcc", "lattice": 2.87},
        "vacuum": 5.0
    }

    gen = InterfaceGenerator(config)
    structures = gen.generate()

    assert len(structures) == 1
    atoms = structures[0]
    assert isinstance(atoms, Atoms)
    # Check for presence of both elements
    symbols = atoms.get_chemical_symbols()
    assert "Mg" in symbols
    assert "Fe" in symbols
    # Check vacuum
    assert atoms.cell[2, 2] > 10.0  # Vacuum added


def test_surface_generator():
    """Test SurfaceGenerator."""
    config = {
        "type": "surface",
        "element": "Pt",
        "indices": [[1, 1, 1]],
        "layers": 3,
        "vacuum": 5.0
    }

    gen = SurfaceGenerator(config)
    structures = gen.generate()

    assert len(structures) == 1
    atoms = structures[0]
    assert "Pt" in atoms.get_chemical_symbols()
    # Check vacuum
    assert atoms.cell[2, 2] > 5.0


def test_defect_generator_vacancy():
    """Test DefectGenerator with vacancy."""
    config = {
        "type": "defect",
        "base": {"symbol": "MgO", "structure": "rocksalt", "lattice": 4.21},
        "defect_type": "vacancy",
        "species": "Mg",
        "concentration": 0.1 # High enough to ensure removal
    }

    gen = DefectGenerator(config)
    structures = gen.generate()

    assert len(structures) == 1
    atoms = structures[0]
    # Check if atom count is reduced
    # Base is 2 atoms * 3x3x3 = 54 atoms
    # 10% vacancy -> remove ~5 atoms
    assert len(atoms) < 54


def test_scenario_factory():
    """Test ScenarioFactory."""
    config = {"type": "interface"}
    gen = ScenarioFactory.create(config)
    assert isinstance(gen, InterfaceGenerator)

    with pytest.raises(ValueError):
        ScenarioFactory.create({"type": "unknown"})
