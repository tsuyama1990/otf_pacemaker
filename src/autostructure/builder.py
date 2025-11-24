from typing import List, Union, Optional, Dict
from ase import Atoms
from pymatgen.core import Structure

from .intelligence import MaterialIntelligence, MaterialType
from .ionic import IonicGenerator
from .alloy import AlloyGenerator
from .covalent import CovalentGenerator
from .molecular import MolecularGenerator
from .interface import InterfaceBuilder
from .preopt import PreOptimizer
from src.core.config import Config

class AutoStructureBuilder:
    """
    Main entry point for the AutoStructureBuilder library.
    Automatically selects and executes generation strategies.
    """

    def __init__(self, config: Optional[Config] = None, api_key: Optional[str] = None):
        """
        Args:
            config (Config, optional): Global configuration containing LJ params, etc.
            api_key (str, optional): MP API Key.
        """
        self.intelligence = MaterialIntelligence(api_key=api_key)
        self.config = config

        # Extract LJ params or use defaults
        if self.config:
            self.lj_params = {
                "epsilon": self.config.lj_params.epsilon,
                "sigma": self.config.lj_params.sigma,
                "cutoff": self.config.lj_params.cutoff
            }
        else:
            # Fallback (Should be avoided)
            self.lj_params = {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}

    def generate(self, structure: Union[Atoms, Structure], formula_or_id: Optional[str] = None) -> List[Atoms]:
        """
        Analyzes the input structure and generates a comprehensive training set.

        Args:
            structure: Input crystal structure (ASE Atoms or Pymatgen Structure).
            formula_or_id: Optional chemical formula or MP-ID to aid classification.

        Returns:
            List[Atoms]: Generated structures.
        """
        from pymatgen.io.ase import AseAtomsAdaptor

        if isinstance(structure, Atoms):
            pmg_struct = AseAtomsAdaptor.get_structure(structure)
        else:
            pmg_struct = structure

        # 1. Classify
        mat_type = self.intelligence.analyze(pmg_struct, formula_or_id)

        generators = []

        # 2. Select Strategies
        # Inject lj_params into generators
        if mat_type == MaterialType.IONIC:
            generators.append(IonicGenerator(pmg_struct, lj_params=self.lj_params))
        elif mat_type == MaterialType.ALLOY:
            generators.append(AlloyGenerator(pmg_struct, lj_params=self.lj_params))
        elif mat_type == MaterialType.COVALENT:
            generators.append(CovalentGenerator(pmg_struct, lj_params=self.lj_params))
        elif mat_type == MaterialType.MOLECULAR:
            generators.append(MolecularGenerator(pmg_struct, lj_params=self.lj_params))
        else:
            # Default fallback
            generators.append(AlloyGenerator(pmg_struct, lj_params=self.lj_params))

        # 3. Execute
        results = []
        for gen in generators:
            results.extend(gen.generate_all())
            # Dump config
            gen.export_config(f"generator_config_{mat_type.value}.yaml")

        return results

    def build_interface(self, struct_a, struct_b) -> List[Atoms]:
        """
        Explicitly builds interface structures between two materials.
        """
        builder = InterfaceBuilder(struct_a, struct_b, lj_params=self.lj_params)
        return builder.generate_all()
