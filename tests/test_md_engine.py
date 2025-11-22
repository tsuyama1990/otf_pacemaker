import os
from pathlib import Path
from unittest.mock import patch
import pytest
from src.md_engine import LAMMPSRunner

def test_lammps_input_generation(tmp_path):
    # 1. Setup
    # We want to run in tmp_path so files are created there.
    # But LAMMPSRunner writes to "in.lammps" in the current directory.
    # So we should change cwd to tmp_path or verify if LAMMPSRunner can take a path.
    # Looking at code: input_file_path = "in.lammps" (hardcoded).
    # So we must change CWD.

    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        cmd = "echo" # Dummy command
        lj_params = {'epsilon': 1.0, 'sigma': 1.0, 'cutoff': 2.5}
        md_params = {
            'elements': ['Ag', 'Pd'],
            'timestep': 0.001,
            'temperature': 300,
            'pressure': 1.0,
            'restart_freq': 1000
        }

        runner = LAMMPSRunner(cmd=cmd, lj_params=lj_params, md_params=md_params)

        potential_path = "test_pot.yace"
        steps = 100
        gamma_threshold = 0.15
        input_structure = "data.in"

        # Create dummy input structure file to avoid file not found if checked
        Path(input_structure).touch()
        # Create dummy log.lammps
        Path("log.lammps").touch()

        # 2. Run (or Mock Run)
        # We mock subprocess.run to avoid actually running "echo" if we want,
        # but running "echo" is safe. However, the code opens stdout.log.

        runner.run_md(
            potential_path=potential_path,
            steps=steps,
            gamma_threshold=gamma_threshold,
            input_structure=input_structure,
            is_restart=False
        )

        # 3. Verify 'in.lammps'
        input_file = tmp_path / "in.lammps"
        assert input_file.exists()

        content = input_file.read_text()

        # Check for required commands
        assert "pair_style hybrid/overlay pace/extrapolation lj/cut 2.5" in content
        assert f"pair_coeff * * pace/extrapolation {potential_path} Ag Pd" in content
        assert "pair_coeff * * lj/cut 1.0 1.0" in content

        # Check for fix halt and uncertainty monitoring
        # "fix f_gamma all pair 10 pace/extrapolation gamma 1"
        assert "fix f_gamma all pair 10 pace/extrapolation gamma 1" in content
        # "compute c_max_gamma all reduce max f_f_gamma"
        assert "compute c_max_gamma all reduce max f_f_gamma" in content
        # "fix halt_sim all halt 10 v_max_gamma > 0.15 error continue"
        assert "fix halt_sim all halt 10 v_max_gamma > 0.15 error continue" in content

    finally:
        os.chdir(original_cwd)
