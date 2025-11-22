"""MD Engine module for running LAMMPS simulations.

This module handles the execution of Molecular Dynamics simulations using LAMMPS,
including the generation of input files and management of simulation states.
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from src.interfaces import MDEngine
from src.enums import SimulationState


class LAMMPSRunner(MDEngine):
    """Handles execution of LAMMPS simulations.

    Attributes:
        cmd: The command to run LAMMPS.
        lj_params: Dictionary containing Lennard-Jones parameters.
        md_params: Dictionary containing MD parameters.
    """

    def __init__(self, cmd: str, lj_params: dict, md_params: dict):
        """Initialize the LAMMPSRunner.

        Args:
            cmd: The command string to execute LAMMPS.
            lj_params: LJ parameters (epsilon, sigma, cutoff).
            md_params: MD parameters (elements, timestep, temp, press, etc.).
        """
        self.cmd = cmd
        self.lj_params = lj_params
        self.md_params = md_params

    def run(
        self,
        potential_path: str,
        steps: int,
        gamma_threshold: float,
        input_structure: str,
        is_restart: bool = False,
    ) -> SimulationState:
        """Run an MD simulation using the specified potential and parameters.

        Args:
            potential_path: Path to the ACE potential file.
            steps: Number of MD steps to run.
            gamma_threshold: Uncertainty threshold to stop the simulation.
            input_structure: Path to structure file.
            is_restart: Whether the input_structure is a restart file.

        Returns:
            SimulationState: The final state of the simulation.
        """
        input_file_path = "in.lammps"

        self._write_input_file(
            input_file_path,
            potential_path,
            steps,
            gamma_threshold,
            input_structure,
            is_restart
        )

        return self._execute_lammps(input_file_path)

    def _write_input_file(
        self,
        filepath: str,
        potential_path: str,
        steps: int,
        gamma_threshold: float,
        input_structure: str,
        is_restart: bool,
    ) -> None:
        """Generates the LAMMPS input script."""

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
        dump_freq = self.md_params.get("dump_freq", 1000)
        masses = self.md_params.get("masses", {})

        lines = [
            "# ACE Active Carver MD Simulation",
            "units metal",
            "atom_style atomic",
            "boundary p p p",
        ]

        if is_restart:
            lines.append(f"read_restart {input_structure}")
        else:
            lines.append(f"read_data {input_structure}")

        # Dynamic mass setting
        for i, el in enumerate(elements):
            # Raise KeyError if mass is missing to prevent physical errors
            mass = masses[el]
            lines.append(f"mass {i+1} {mass}")

        # Pair style
        lines.append(f"pair_style hybrid/overlay pace/extrapolation lj/cut {rcut}")
        lines.append(f"pair_coeff * * pace/extrapolation {potential_path} {element_map}")
        lines.append(f"pair_coeff * * lj/cut {epsilon} {sigma}")
        lines.append("pair_modify shift yes")

        lines.append("neighbor 1.0 bin")
        lines.append("neigh_modify delay 0 every 1 check yes")
        lines.append(f"timestep {timestep}")
        lines.append("thermo 10")
        lines.append("thermo_style custom step temp press pe ke etotal")

        # NPT fix
        lines.append(f"fix 1 all npt temp {temp} {temp} 0.1 iso {press} {press} 1.0")

        # Uncertainty monitoring
        lines.append("# Gamma calculation fix")
        lines.append("fix f_gamma all pair 10 pace/extrapolation gamma 1")
        lines.append("compute c_max_gamma all reduce max f_f_gamma")
        lines.append("variable v_max_gamma equal c_max_gamma")
        lines.append(f"fix halt_sim all halt 10 v_max_gamma > {gamma_threshold} error continue")

        # Restart and Dump
        lines.append(f"restart {restart_freq} restart.chk")
        # We dump f_f_gamma to track uncertainty
        lines.append(f"dump 1 all custom {dump_freq} dump.lammpstrj id type x y z fx fy fz f_f_gamma")

        lines.append(f"run {steps}")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

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
             # If log wasn't created, check return code
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
