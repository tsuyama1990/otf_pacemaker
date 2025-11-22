"""MD Engine module for running LAMMPS simulations.

This module handles the execution of Molecular Dynamics simulations using LAMMPS,
including the generation of input files and management of simulation states.
"""

from enum import Enum
from pathlib import Path


class SimulationState(Enum):
    """Enum representing the state of a simulation after execution."""

    COMPLETED = "completed"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


class LAMMPSRunner:
    """Handles execution of LAMMPS simulations.

    Attributes:
        cmd: The command to run LAMMPS (e.g., 'lmp_serial' or 'mpirun -np 4 lmp_mpi').
        lj_params: Dictionary containing Lennard-Jones parameters.
    """

    def __init__(self, cmd: str, lj_params: dict[str, float]):
        """Initialize the LAMMPSRunner.

        Args:
            cmd: The command string to execute LAMMPS.
            lj_params: A dictionary containing LJ parameters (epsilon, sigma, cutoff).
        """
        self.cmd = cmd
        self.lj_params = lj_params

    def run_md(
        self, potential_path: str, steps: int, gamma_threshold: float
    ) -> SimulationState:
        """Run an MD simulation using the specified potential and parameters.

        This method generates the necessary LAMMPS input files, including the
        `pair_style hybrid/overlay` configuration for using both LJ and ACE potentials.

        Args:
            potential_path: Path to the ACE potential file (.yace or similar).
            steps: Number of MD steps to run.
            gamma_threshold: Uncertainty threshold to stop the simulation.

        Returns:
            SimulationState: The final state of the simulation (COMPLETED, UNCERTAIN, or FAILED).
        """
        pass
