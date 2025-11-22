"""Configuration module for the ACE Active Carver project.

This module defines the configuration data structures used throughout the application.
It uses Python's standard dataclasses for definition and PyYAML for loading from files.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MDParams:
    """Parameters for Molecular Dynamics simulations.

    Attributes:
        timestep: Time step in femtoseconds.
        temperature: Simulation temperature in Kelvin.
        pressure: Simulation pressure in bars.
        n_steps: Number of MD steps to run.
    """

    timestep: float
    temperature: float
    pressure: float
    n_steps: int
    elements: list[str]
    initial_structure: str
    masses: dict[str, float]
    restart_freq: int = 1000
    dump_freq: int = 1000


@dataclass
class ALParams:
    """Parameters for Active Learning strategy.

    Attributes:
        gamma_threshold: Uncertainty threshold for stopping the simulation.
        n_clusters: Number of clusters to extract.
        r_core: Radius for the core region where forces are fully weighted (and fixed during relaxation).
        box_size: Size of the cubic small cell (Angstroms).
        initial_potential: Path to the initial potential file.
        stoichiometry_tolerance: Tolerance for stoichiometry check (default 0.1).
    """

    gamma_threshold: float
    n_clusters: int
    r_core: float
    box_size: float
    initial_potential: str
    stoichiometry_tolerance: float = 0.1


@dataclass
class DFTParams:
    """Parameters for Density Functional Theory calculations.

    Attributes:
        ecutwfc: Kinetic energy cutoff for wavefunctions in Ry.
        kpts: k-points mesh grid (e.g., [2, 2, 2]).
        pseudo_dir: Directory containing pseudopotentials.
        command: Command to execute the DFT code (e.g., 'mpirun -np 4 pw.x -in PREFIX.pwi > PREFIX.pwo').
    """

    ecutwfc: float
    kpts: tuple[int, int, int]
    pseudo_dir: str
    command: str


@dataclass
class LJParams:
    """Parameters for Lennard-Jones potential.

    Attributes:
        epsilon: Depth of the potential well.
        sigma: Finite distance at which the inter-particle potential is zero.
        cutoff: Cutoff distance for the potential.
    """

    epsilon: float
    sigma: float
    cutoff: float


@dataclass
class Config:
    """Main configuration class aggregating all parameter sections.

    Attributes:
        md_params: Molecular Dynamics parameters.
        al_params: Active Learning parameters.
        dft_params: DFT calculation parameters.
        lj_params: Lennard-Jones potential parameters.
    """

    md_params: MDParams
    al_params: ALParams
    dft_params: DFTParams
    lj_params: LJParams

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration data.

        Returns:
            Config: An initialized Config object.
        """
        return cls(
            md_params=MDParams(**config_dict.get("md_params", {})),
            al_params=ALParams(**config_dict.get("al_params", {})),
            dft_params=DFTParams(**config_dict.get("dft_params", {})),
            lj_params=LJParams(**config_dict.get("lj_params", {})),
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Config: An initialized Config object.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
