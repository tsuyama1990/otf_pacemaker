import pytest
from ase import Atoms
from src.engines.dft.configurator import DFTConfigurator
from src.workflows.seed_generation import SeedGenerator
from src.scenario_generation.strategies.small_cell import SmallCellGenerator
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_dft_params():
    """Fixture for DFTParams."""
    params = MagicMock()
    params.auto_physics = True
    return params

@pytest.fixture
def mock_meta_config():
    """Fixture for MetaConfig."""
    meta = MagicMock()
    meta.sssp_json_path = "path/to/sssp.json"
    meta.pseudo_dir.resolve.return_value = "/abs/path/to/pseudo"
    meta.dft_command = "mpirun -np 4 pw.x"
    return meta

@patch("src.engines.dft.configurator.load_sssp_database")
@patch("src.engines.dft.configurator.validate_pseudopotentials")
@patch("src.engines.dft.configurator.get_pseudopotentials_dict")
@patch("src.engines.dft.configurator.calculate_cutoffs")
@patch("src.engines.dft.configurator.Espresso")
def test_charged_defect_configuration(mock_espresso, mock_cutoffs, mock_pseudos, mock_validate, mock_sssp, mock_dft_params, mock_meta_config):
    """Test that atoms.info['charge'] is propagated to the QE calculator."""
    # Arrange
    configurator = DFTConfigurator(params=mock_dft_params, meta=mock_meta_config)
    atoms = Atoms('NaCl', info={'charge': -2})

    # Mock the return values of the patched functions
    mock_sssp.return_value = {}
    mock_validate.return_value = None
    mock_pseudos.return_value = {}
    mock_cutoffs.return_value = (60, 240)

    # Act
    calculator, _ = configurator.build(atoms, ['Na', 'Cl'])

    # Assert
    mock_espresso.assert_called_once()
    _, kwargs = mock_espresso.call_args
    input_data = kwargs.get("input_data", {})
    system_data = input_data.get("system", {})
    assert system_data.get("tot_charge") == -2

@patch("src.workflows.seed_generation.AtomicEnergyManager")
@patch("src.workflows.seed_generation.PymatgenHeuristics")
@patch("src.workflows.seed_generation.Espresso")
def test_magnetic_e0_calculation(mock_espresso, mock_heuristics, mock_atomic_manager):
    """Test that magnetic elements get correct spin polarization for E0."""
    # Arrange
    mock_config = MagicMock()
    mock_config.md_params.elements = ['Fe']
    mock_config.dft_params.sssp_json_path = "dummy.json"
    mock_config.dft_params.pseudo_dir = "pseudo"
    mock_config.dft_params.command = "pw.x"

    mock_manager_instance = mock_atomic_manager.return_value
    mock_manager_instance.get_e0.return_value = {"Fe": -100.0}

    mock_heuristics.get_recommended_params.return_value = {
        "magnetism": {
            "nspin": 2,
            "moments": {"Fe": 5.0}
        },
        "system": {}
    }

    # Act & Assert
    with patch('src.workflows.seed_generation.load_sssp_database', return_value={}), \
         patch('src.workflows.seed_generation.get_pseudopotentials_dict', return_value={}), \
         patch('src.workflows.seed_generation.calculate_cutoffs', return_value=(60, 240)), \
         patch('pathlib.Path.exists', return_value=True):

        generator = SeedGenerator(mock_config)
        generator._get_e0_dict()

        mock_manager_instance.get_e0.assert_called_once()
        args, kwargs = mock_manager_instance.get_e0.call_args
        dft_factory = args[1] if len(args) > 1 else kwargs['calculator_factory']

        calculator = dft_factory('Fe')

    mock_espresso.assert_called_once()
    _, kwargs_espresso = mock_espresso.call_args
    input_data = kwargs_espresso.get("input_data", {})
    system_data = input_data.get("system", {})
    assert system_data.get("nspin") == 2
    assert system_data.get("starting_magnetization") == {"Fe": 5.0}

def test_hydrogen_overlap_preservation():
    """Test that H2 molecule is preserved with the new overlap logic."""
    # Arrange
    h2_molecule = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    bond_thresholds = {"H-H": 0.7, "default": 1.5}
    generator = SmallCellGenerator(r_core=1.0, box_size=10.0, stoichiometric_ratio={}, bond_thresholds=bond_thresholds)

    # Act
    generator._remove_overlaps(h2_molecule, center_pos=[0, 0, 0])

    # Assert
    assert len(h2_molecule) == 2

def test_hydrogen_overlap_removal():
    """Test that two Fe atoms at H2 distance are removed."""
    # Arrange
    fe2_molecule = Atoms('Fe2', positions=[[0, 0, 0], [0, 0, 0.74]])
    bond_thresholds = {"H-H": 0.7, "default": 1.5}
    generator = SmallCellGenerator(r_core=1.0, box_size=10.0, stoichiometric_ratio={}, bond_thresholds=bond_thresholds)

    # Act
    generator._remove_overlaps(fe2_molecule, center_pos=[0, 0, 0])

    # Assert
    assert len(fe2_molecule) == 1
