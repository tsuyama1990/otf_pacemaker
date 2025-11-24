
import pytest
from src.core.config import Config, LJParams, MDParams
from ase.data import covalent_radii, atomic_numbers

def test_config_from_dict_explicit_lj_params():
    config_dict = {
        "md_params": {
            "timestep": 1.0,
            "temperature": 300,
            "pressure": 0,
            "n_steps": 100,
            "elements": ["Cu"],
            "initial_structure": "structure.xyz",
            "masses": {"Cu": 63.55}
        },
        "al_params": {
            "gamma_threshold": 0.1,
            "n_clusters": 5,
            "r_core": 4.0,
            "box_size": 12.0,
            "initial_potential": "pot.ace",
            "potential_yaml_path": "pot.yaml"
        },
        "dft_params": {
            "sssp_json_path": "sssp.json",
            "ecutwfc": 50,
            "kpts": [1, 1, 1],
            "pseudo_dir": ".",
            "command": "pw.x"
        },
        "lj_params": {
            "epsilon": 0.5,
            "sigma": 2.0,
            "cutoff": 5.0
        },
        "seed": 123
    }

    config = Config.from_dict(config_dict)
    assert config.lj_params.epsilon == 0.5
    assert config.lj_params.sigma == 2.0
    assert config.lj_params.cutoff == 5.0
    assert config.seed == 123
    # Check that extra keys were filtered (no error raised)
    assert config.dft_params.sssp_json_path == "sssp.json"

def test_config_from_dict_generated_lj_params():
    elements = ["Cu", "Ni"]
    config_dict = {
        "md_params": {
            "timestep": 1.0,
            "temperature": 300,
            "pressure": 0,
            "n_steps": 100,
            "elements": elements,
            "initial_structure": "structure.xyz",
            "masses": {"Cu": 63.55, "Ni": 58.69}
        },
        "al_params": {
            "gamma_threshold": 0.1,
            "n_clusters": 5,
            "r_core": 4.0,
            "box_size": 12.0,
            "initial_potential": "pot.ace",
            "potential_yaml_path": "pot.yaml"
        },
        "dft_params": {
            "sssp_json_path": "sssp.json",
            "ecutwfc": 50,
            "kpts": [1, 1, 1],
            "pseudo_dir": ".",
            "command": "pw.x"
        }
        # lj_params MISSING
    }

    config = Config.from_dict(config_dict)

    # Calculate expected sigma
    # r_avg = (r_Cu + r_Ni) / 2
    r_cu = covalent_radii[atomic_numbers["Cu"]]
    r_ni = covalent_radii[atomic_numbers["Ni"]]
    r_avg = (r_cu + r_ni) / 2.0

    # User's implementation uses hardcoded 0.8909 and rounding to 3 decimals
    expected_sigma = round(2.0 * r_avg * 0.8909, 3)
    expected_epsilon = 1.0
    expected_cutoff = round(2.5 * expected_sigma, 3)

    assert config.lj_params.epsilon == expected_epsilon
    # Relax tolerance slightly to account for floating point rounding differences
    # on .5 boundaries (Banker's rounding vs standard rounding)
    assert abs(config.lj_params.sigma - expected_sigma) < 1e-3
    assert abs(config.lj_params.cutoff - expected_cutoff) < 2e-3
    assert config.seed == 42 # Default

def test_config_fails_if_no_elements_and_no_lj_params():
    config_dict = {
        "md_params": {
            "timestep": 1.0,
            "temperature": 300,
            "pressure": 0,
            "n_steps": 100,
            # elements MISSING
            "initial_structure": "structure.xyz",
            "masses": {"Cu": 63.55}
        },
        "al_params": {
            "gamma_threshold": 0.1,
            "n_clusters": 5,
            "r_core": 4.0,
            "box_size": 12.0,
            "initial_potential": "pot.ace",
            "potential_yaml_path": "pot.yaml"
        },
        "dft_params": {
            "sssp_json_path": "sssp.json",
            "ecutwfc": 50,
            "kpts": [1, 1, 1],
            "pseudo_dir": ".",
            "command": "pw.x"
        }
        # lj_params MISSING
    }

    # This should fail.
    # Either MDParams raises TypeError (missing elements) or LJParams raises TypeError (missing args)
    # We just ensure it raises TypeError
    with pytest.raises(TypeError):
        Config.from_dict(config_dict)
