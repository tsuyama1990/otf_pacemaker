"""LAMMPS Runner.

Handles execution of LAMMPS simulations.
"""

import subprocess
from pathlib import Path

from src.core.interfaces import MDEngine
from src.core.enums import SimulationState
from .input_generator import LAMMPSInputGenerator

class LAMMPSRunner(MDEngine):
    """Handles execution of LAMMPS simulations.

    Attributes:
        cmd: The command to run LAMMPS.
        input_generator: Component to generate input files.
    """

    def __init__(self, cmd: str, input_generator: LAMMPSInputGenerator):
        """Initialize the LAMMPSRunner.

        Args:
            cmd: The command string to execute LAMMPS.
            input_generator: Instance of LAMMPSInputGenerator.
        """
        self.cmd = cmd
        self.input_generator = input_generator

    def run(
        self,
        potential_path: str,
        steps: int,
        gamma_threshold: float,
        input_structure: str,
        is_restart: bool = False,
    ) -> SimulationState:
        """Run an MD simulation using the specified potential and parameters."""
        input_file_path = "in.lammps"

        self.input_generator.generate(
            input_file_path,
            potential_path,
            steps,
            gamma_threshold,
            input_structure,
            is_restart
        )

        return self._execute_lammps(input_file_path)

    def _execute_lammps(self, input_file_path: str) -> SimulationState:
        """Executes the LAMMPS command and checks the result."""
        cmd_list = self.cmd.split() + ["-in", input_file_path]

        try:
            with open("stdout.log", "w") as out, open("stderr.log", "w") as err:
                result = subprocess.run(cmd_list, stdout=out, stderr=err)
        except Exception as e:
            print(f"LAMMPS execution failed: {e}")
            return SimulationState.FAILED

        # Check log for halt condition
        log_path = "log.lammps"
        if not Path(log_path).exists():
             if result.returncode != 0:
                 return SimulationState.FAILED
             return SimulationState.COMPLETED

        with open(log_path, "r") as f:
            log_content = f.read()

        if "Fix halt" in log_content:
            return SimulationState.UNCERTAIN
        elif "ERROR" in log_content or result.returncode != 0:
             return SimulationState.FAILED
        else:
            return SimulationState.COMPLETED
