from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
import yaml
import logging

from .preopt import PreOptimizer

logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    """
    Abstract base class for all structure generators.
    """

    def __init__(
        self,
        base_structure: Union[Atoms, Structure],
        lj_params: Optional[Dict[str, float]] = None
    ):
        if isinstance(base_structure, Atoms):
            self.structure = AseAtomsAdaptor.get_structure(base_structure)
        else:
            self.structure = base_structure

        self.generated_structures: List[Atoms] = []
        self.config: Dict[str, Any] = {}

        # Use provided LJ params or safe defaults if not provided (legacy support)
        if lj_params is None:
            # Fallback (Should be avoided by proper injection)
            lj_params = {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}

        self.pre_optimizer = PreOptimizer(lj_params=lj_params)

    def _add_structure(self, struct: Union[Structure, Atoms], meta: Dict[str, Any] = None):
        """Helper to convert, relax, and store structure."""
        if isinstance(struct, Structure):
            atoms = AseAtomsAdaptor.get_atoms(struct)
        else:
            atoms = struct

        # Copy metadata
        if meta:
            if not atoms.info:
                atoms.info = {}
            atoms.info.update(meta)

        # Safety Valve: Pre-optimize
        try:
            atoms = self.pre_optimizer.run_pre_optimization(atoms)
            self.generated_structures.append(atoms)
        except ValueError as e:
            logger.warning(f"Structure discarded during pre-optimization: {e}")

    @abstractmethod
    def generate_all(self) -> List[Atoms]:
        """Generate all structures for this strategy."""
        pass

    def export_config(self, filepath: str = "generator_config.yaml"):
        """Exports the generation configuration to a YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f)
