import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from src.engines.lammps.runner import LAMMPSRunner
from src.engines.lammps.input_generator import LAMMPSInputGenerator
from src.core.enums import SimulationState

def test_lammps_input_generation(tmp_path):
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        cmd = "echo"
        lj_params = {'epsilon': 1.0, 'sigma': 1.0, 'cutoff': 2.5}
        md_params = {
            'elements': ['Ag', 'Pd'],
            'timestep': 0.001,
            'temperature': 300,
            'pressure': 1.0,
            'restart_freq': 1000,
            'masses': {'Ag': 107.87, 'Pd': 106.42},
            'dump_freq': 1000
        }

        input_generator = LAMMPSInputGenerator(lj_params=lj_params, md_params=md_params)
        runner = LAMMPSRunner(cmd=cmd, input_generator=input_generator)

        potential_path = "test_pot.yace"
        steps = 100
        gamma_threshold = 0.15
        input_structure = "data.in"

        Path(input_structure).touch()
        Path("log.lammps").touch()

        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0

            runner.run(
                potential_path=potential_path,
                steps=steps,
                gamma_threshold=gamma_threshold,
                input_structure=input_structure,
                is_restart=False
            )

        input_file = tmp_path / "in.lammps"
        assert input_file.exists()

        content = input_file.read_text()

        assert "pair_style hybrid/overlay pace/extrapolation lj/cut 2.5" in content
        assert f"pair_coeff * * pace/extrapolation {potential_path} Ag Pd" in content
        assert "pair_coeff * * lj/cut 1.0 1.0" in content
        assert "fix f_gamma all pair 10 pace/extrapolation gamma 1" in content
        assert "compute c_max_gamma all reduce max f_f_gamma" in content
        assert "fix halt_sim all halt 10 v_max_gamma > 0.15 error continue" in content
        assert "mass 1 107.87" in content
        assert "mass 2 106.42" in content

    finally:
        os.chdir(original_cwd)
