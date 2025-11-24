"""Configuration module for the ACE Active Carver project.

This module defines the configuration data structures used throughout the application.
It uses Python's standard dataclasses for definition and PyYAML for loading from files.
"""

import yaml
import numpy as np
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, List, Dict
from ase.data import atomic_numbers, covalent_radii

logger = logging.getLogger(__name__)

def generate_default_lj_params(elements: List[str]) -> Dict[str, float]:
    """
    Generates robust default Lennard-Jones parameters based on element physics.
    """
    if not elements:
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
        sigma = (2.0 * avg_radius) * 0.8909

        return {
            "epsilon": 1.0,
            "sigma": float(round(sigma, 3)),
            "cutoff": float(round(2.5 * sigma, 3))
        }
    except Exception as e:
        print(f"Warning: Could not auto-generate LJ params ({e}). Using safe defaults.")
        return {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}

@dataclass
class MetaConfig:
    """Environment-specific configuration."""
    dft: Dict[str, Any]
    lammps: Dict[str, Any]

    @property
    def dft_command(self) -> str:
        return self.dft.get("command", "pw.x")

    @property
    def pseudo_dir(self) -> Path:
        return Path(self.dft.get("pseudo_dir", "."))

    @property
    def sssp_json_path(self) -> Path:
        return Path(self.dft.get("sssp_json_path", "."))

    @property
    def lammps_command(self) -> str:
        return self.lammps.get("command", "lmp_serial")


@dataclass
class ExperimentConfig:
    """Experiment metadata and output settings."""
    name: str
    output_dir: Path


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
    # lammps_command removed, now in MetaConfig


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
    sampling_strategy: str = "composite"
    outlier_energy_max: float = 10.0


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
    kpoint_density: float = 60.0
    auto_physics: bool = True
    # command, pseudo_dir, sssp_json_path removed, now in MetaConfig


@dataclass
class LJParams:
    """Parameters for Lennard-Jones potential."""
    epsilon: float
    sigma: float
    cutoff: float
    shift_energy: bool = True


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
    device: str = "cuda"


@dataclass
class ACEModelParams:
    """Parameters for Pacemaker potential model."""
    pacemaker_config: Dict[str, Any] = field(default_factory=dict)
    initial_potentials: List[str] = field(default_factory=list)


@dataclass
class TrainingParams:
    """Parameters for Active Learning Training strategy."""
    replay_ratio: float = 1.0
    global_dataset_path: str = "data/global_dataset.pckl"
    # Pacemaker-specific params removed, moved to ACEModelParams.pacemaker_config


@dataclass
class Config:
    """Main configuration class aggregating all parameter sections."""
    meta: MetaConfig
    experiment: ExperimentConfig
    md_params: MDParams
    al_params: ALParams
    dft_params: DFTParams
    lj_params: LJParams
    training_params: TrainingParams
    ace_model: ACEModelParams
    seed: int = 42
    kmc_params: KMCParams = field(default_factory=KMCParams)
    generation_params: GenerationParams = field(default_factory=GenerationParams)

    @classmethod
    def load_meta(cls, path: Path) -> MetaConfig:
        """Load environment configuration from meta_config.yaml."""
        if not path.exists():
             raise FileNotFoundError(f"Meta config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            meta_dict = yaml.safe_load(f) or {}

        return MetaConfig(
            dft=meta_dict.get("dft", {}),
            lammps=meta_dict.get("lammps", {})
        )

    @classmethod
    def load_experiment(cls, config_path: Path, meta_config: MetaConfig) -> "Config":
        """Load experiment configuration and combine with meta config."""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        # Handle constants inheritance if needed (legacy support)
        constant_path = path.parent / "constant.yaml"
        if constant_path.exists():
            with constant_path.open("r", encoding="utf-8") as f:
                constant_dict = yaml.safe_load(f) or {}

            merged_dict = constant_dict.copy()
            def update_recursive(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = update_recursive(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            update_recursive(merged_dict, config_dict)
            config_dict = merged_dict

        return cls.from_dict(config_dict, meta_config)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], meta_config: MetaConfig) -> "Config":
        """Create a Config instance from a dictionary."""
        gen_dict = config_dict.get("generation", {})
        pre_opt_dict = gen_dict.get("pre_optimization", {})

        generation_params = GenerationParams(
            pre_optimization=PreOptimizationParams(**pre_opt_dict),
            scenarios=gen_dict.get("scenarios", [])
        )

        md_dict = config_dict.get("md_params", {})

        lj_dict = config_dict.get("lj_params", {})
        if not lj_dict:
            elements = md_dict.get("elements", [])
            lj_dict = generate_default_lj_params(elements)

        # DFT Params: Filter out environment paths if they still exist in dict (backwards compat)
        dft_dict = config_dict.get("dft_params", {}).copy()
        allowed_dft_keys = {"kpoint_density", "auto_physics"}
        dft_dict = {k: v for k, v in dft_dict.items() if k in allowed_dft_keys}

        exp_dict = config_dict.get("experiment", {})

        ace_dict = config_dict.get("ace_model", {})

        return cls(
            meta=meta_config,
            experiment=ExperimentConfig(
                name=exp_dict.get("name", "experiment"),
                output_dir=Path(exp_dict.get("output_dir", "output"))
            ),
            md_params=MDParams(**md_dict),
            al_params=ALParams(**config_dict.get("al_params", {})),
            kmc_params=KMCParams(**config_dict.get("kmc_params", {})),
            dft_params=DFTParams(**dft_dict),
            lj_params=LJParams(**lj_dict),
            training_params=TrainingParams(**config_dict.get("training_params", {})),
            ace_model=ACEModelParams(**ace_dict),
            generation_params=generation_params,
            seed=config_dict.get("seed", 42)
        )
