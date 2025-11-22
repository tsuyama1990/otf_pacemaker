"""MD Engine module for running LAMMPS simulations.

This module handles the execution of Molecular Dynamics simulations using LAMMPS,
including the generation of input files and management of simulation states.
"""

import subprocess
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
        md_params: Dictionary containing MD parameters.
    """

    def __init__(self, cmd: str, lj_params: dict, md_params: dict):
        """Initialize the LAMMPSRunner.

        Args:
            cmd: The command string to execute LAMMPS.
            lj_params: A dictionary containing LJ parameters (epsilon, sigma, cutoff).
            md_params: A dictionary containing MD parameters (elements, timestep, temp, press, etc.).
        """
        self.cmd = cmd
        self.lj_params = lj_params
        self.md_params = md_params

    def run_md(
        self,
        potential_path: str,
        steps: int,
        gamma_threshold: float,
        input_structure: str,
        is_restart: bool = False,
    ) -> SimulationState:
        """Run an MD simulation using the specified potential and parameters.

        This method generates the necessary LAMMPS input files, including the
        `pair_style hybrid/overlay` configuration for using both LJ and ACE potentials.

        Args:
            potential_path: Path to the ACE potential file (.yace or similar).
            steps: Number of MD steps to run.
            gamma_threshold: Uncertainty threshold to stop the simulation.
            input_structure: Path to structure file (data file or restart file).
            is_restart: Whether the input_structure is a restart file.

        Returns:
            SimulationState: The final state of the simulation (COMPLETED, UNCERTAIN, or FAILED).
        """
        input_file_path = "in.lammps"
        log_file_path = "log.lammps"
        dump_file_path = "dump.lammpstrj"
        restart_file_path = "restart.chk"

        # Extract parameters
        epsilon = self.lj_params.get("epsilon", 1.0)
        sigma = self.lj_params.get("sigma", 1.0)
        rcut = self.lj_params.get("cutoff", 2.5)

        elements = self.md_params.get("elements", [])
        element_map = " ".join(elements)

        timestep = self.md_params.get("timestep", 0.001)
        temp = self.md_params.get("temperature", 300.0)
        press = self.md_params.get("pressure", 1.0)
        restart_freq = self.md_params.get("restart_freq", 1000)

        # Build LAMMPS input script
        lines = []
        lines.append("# ACE Active Carver MD Simulation")
        lines.append("units metal")
        lines.append("atom_style atomic")
        lines.append("boundary p p p")

        if is_restart:
            lines.append(f"read_restart {input_structure}")
        else:
            lines.append(f"read_data {input_structure}")

        lines.append(f"mass * 1.0") # Simplified mass setting, usually read from data file or set per type

        # Pair style definition
        lines.append(f"pair_style hybrid/overlay pace/extrapolation lj/cut {rcut}")
        lines.append(f"pair_coeff * * pace/extrapolation {potential_path} {element_map}")
        lines.append(f"pair_coeff * * lj/cut {epsilon} {sigma}")

        lines.append(f"neighbor 1.0 bin")
        lines.append(f"neigh_modify delay 0 every 1 check yes")

        lines.append(f"timestep {timestep}")

        lines.append("thermo 10")
        lines.append("thermo_style custom step temp press pe ke etotal")

        # Fixes
        # Use npt for MD
        lines.append(f"fix 1 all npt temp {temp} {temp} 0.1 iso {press} {press} 1.0")

        # Uncertainty monitoring (fix halt)
        # We calculate gamma using pace/extrapolation
        lines.append("# Gamma calculation fix")
        lines.append("fix f_gamma all pair 10 pace/extrapolation gamma 1")
        lines.append("compute c_max_gamma all reduce max f_f_gamma")
        lines.append("variable v_max_gamma equal c_max_gamma")
        lines.append(f"fix halt_sim all halt 10 v_max_gamma > {gamma_threshold} error continue")

        # Restart
        lines.append(f"restart {restart_freq} {restart_file_path}")

        # Dump for extraction
        # We dump periodically or at least the last frame.
        # CRITICAL: We must include 'f_f_gamma' in the dump output to allow identifying
        # uncertain atoms in the active learning step.
        lines.append(f"dump 1 all custom 10 {dump_file_path} id type x y z fx fy fz f_f_gamma")

        lines.append(f"run {steps}")

        # Write input file
        with open(input_file_path, "w") as f:
            f.write("\n".join(lines))

        # Run LAMMPS
        cmd_list = self.cmd.split() + ["-in", input_file_path]

        try:
            # Redirect output to a log file to parse later if needed,
            # but LAMMPS also generates its own log.lammps.
            # We'll capture stdout/stderr to avoid cluttering the console.
            with open("stdout.log", "w") as out, open("stderr.log", "w") as err:
                result = subprocess.run(cmd_list, stdout=out, stderr=err)

            if result.returncode != 0:
                # Check if it failed due to halt or actual error.
                pass
        except Exception as e:
            print(f"LAMMPS execution failed: {e}")
            return SimulationState.FAILED

        # Check log for halt
        with open(log_file_path, "r") as f:
            log_content = f.read()

        if "Fix halt" in log_content:
            return SimulationState.UNCERTAIN
        elif "ERROR" in log_content: # Crude check
             return SimulationState.FAILED
        else:
            return SimulationState.COMPLETED
