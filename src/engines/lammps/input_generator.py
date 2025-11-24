"""LAMMPS Input Generator.

Responsible for creating LAMMPS input scripts.
"""

from typing import Dict, List

class LAMMPSInputGenerator:
    """Responsible for generating LAMMPS input scripts."""

    def __init__(self, lj_params: dict, md_params: dict):
        self.lj_params = lj_params
        self.md_params = md_params

    def generate(
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
        shift_energy = self.lj_params.get("shift_energy", True) # Default to True for safety

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

        # Enforce consistency with Python side ShiftedLennardJones
        if shift_energy:
            lines.append("pair_modify shift yes")
        else:
            lines.append("pair_modify shift no")

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
