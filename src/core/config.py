"""Configuration module for the ACE Active Carver project.

This module defines the configuration data structures used throughout the application.
It uses Python's standard dataclasses for definition and PyYAML for loading from files.
"""

import yaml
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, List, Dict
from ase.data import atomic_numbers, covalent_radii  # REQUIRED for physics-based defaults


def generate_default_lj_params(elements: List[str]) -> Dict[str, float]:
    """
    Generates robust default Lennard-Jones parameters based on element physics.

    Physics:
        - Sigma: Derived from the sum of covalent radii to place the repulsive wall correctly.
                 sigma = (2 * r_avg) * 2^(-1/6)
        - Epsilon: Defaults to 1.0 eV (Strong Repulsion) to ensure MD stability.
        - Cutoff: Defaults to 2.5 * sigma.
    """
    if not elements:
        # Fallback safety defaults
        return {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}

    try:
        radii = []
        for el in elements:
            z = atomic_numbers.get(el)
            if z is None:
                raise ValueError(f"Unknown element symbol: {el}")
            radii.append(covalent_radii[z])

        avg_radius = np.mean(radii)

        # r_min = 2^(1/6) * sigma
        # We want r_min roughly at the sum of radii (2 * avg_radius)
        # Therefore: sigma = (2 * r_avg) / 1.122
        sigma = (2.0 * avg_radius) * 0.8909

        return {
            "epsilon": 1.0,  # Hard stability wall
            "sigma": float(round(sigma, 3)),
            "cutoff": float(round(2.5 * sigma, 3))
        }
    except Exception as e:
        print(f"Warning: Could not auto-generate LJ params ({e}). Using safe defaults.")
        return {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}


@dataclass
class MDParams:
    """Parameters for Molecular Dynamics simulations."""
    timestep: float
    temperature: float
    pressure: float
    n_steps: int
    elements: list[str]
    initial_structure: str
    masses: dict[str, float]
    restart_freq: int = 1000
    dump_freq: int = 1000
    lammps_command: str = "lmp_serial"


@dataclass
class ALParams:
    """Parameters for Active Learning strategy."""
    gamma_threshold: float
    n_clusters: int
    r_core: float
    box_size: float
    initial_potential: str
    potential_yaml_path: str
    initial_dataset_path: Optional[str] = None
    initial_active_set_path: Optional[str] = None
    stoichiometry_tolerance: float = 0.1
    min_bond_distance: float = 1.5
    num_parallel_labeling: int = 4
    query_strategy: str = "uncertainty"


@dataclass
class KMCParams:
    """Parameters for kMC simulations."""
    active: bool = False
    temperature: float = 300.0
    n_searches: int = 10
    search_radius: float = 0.1
    dimer_fmax: float = 0.05
    check_interval: int = 5
    prefactor: float = 1e12
    box_size: float = 12.0
    buffer_width: float = 2.0
    n_workers: int = 4
    active_region_mode: str = "surface_and_species"
    active_species: List[str] = field(default_factory=lambda: ["Co", "Ti", "O"])
    active_z_cutoff: float = 10.0
    move_type: str = "cluster"
    cluster_radius: float = 3.0
    selection_bias: str = "coordination"
    bias_strength: float = 2.0
    adsorbate_cn_cutoff: int = 9
    cluster_connectivity_cutoff: float = 3.0


@dataclass
class DFTParams:
    """Parameters for Density Functional Theory calculations."""
    sssp_json_path: str  # Path to SSSP JSON database
    pseudo_dir: str
    command: str
    kpoint_density: float = 60.0  # Å, for k-point grid calculation (high precision)


@dataclass
class LJParams:
    """Parameters for Lennard-Jones potential."""
    epsilon: float
    sigma: float
    cutoff: float


@dataclass
class PreOptimizationParams:
    """Parameters for MACE pre-optimization."""
    enabled: bool = False
    model: str = "medium"
    fmax: float = 0.1
    steps: int = 50
    device: str = "cuda"


@dataclass
class GenerationParams:
    """Parameters for scenario-driven generation."""
    pre_optimization: PreOptimizationParams = field(default_factory=PreOptimizationParams)
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    device: str = "cuda"  # Device for MACE filtering (cuda/cpu)


@dataclass
class Config:
    """Main configuration class aggregating all parameter sections."""
    md_params: MDParams
    al_params: ALParams
    dft_params: DFTParams
    lj_params: LJParams
    kmc_params: KMCParams = field(default_factory=KMCParams)
    generation_params: GenerationParams = field(default_factory=GenerationParams)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary with auto-LJ generation."""
        gen_dict = config_dict.get("generation", {})
        pre_opt_dict = gen_dict.get("pre_optimization", {})

        generation_params = GenerationParams(
            pre_optimization=PreOptimizationParams(**pre_opt_dict),
            scenarios=gen_dict.get("scenarios", [])
        )

        # Extract MD Params first to get elements
        md_dict = config_dict.get("md_params", {})

        # --- AUTOMATED LJ LOGIC ---
        lj_dict = config_dict.get("lj_params", {})

        if not lj_dict:
            # Retrieve elements from md_params (CRITICAL STEP)
            elements = md_dict.get("elements", [])
            # Generate defaults
            lj_dict = generate_default_lj_params(elements)
        # --------------------------

        return cls(
            md_params=MDParams(**md_dict),
            al_params=ALParams(**config_dict.get("al_params", {})),
            kmc_params=KMCParams(**config_dict.get("kmc_params", {})),
            dft_params=DFTParams(**config_dict.get("dft_params", {})),
            lj_params=LJParams(**lj_dict),
            generation_params=generation_params
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
