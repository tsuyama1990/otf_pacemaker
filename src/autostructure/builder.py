from typing import List, Union, Optional
from ase import Atoms
from pymatgen.core import Structure

from .intelligence import MaterialIntelligence, MaterialType
from .ionic import IonicGenerator
from .alloy import AlloyGenerator
from .covalent import CovalentGenerator
from .molecular import MolecularGenerator
from .interface import InterfaceBuilder
from .preopt import PreOptimizer

class AutoStructureBuilder:
    """
    Main entry point for the AutoStructureBuilder library.
    Automatically selects and executes generation strategies.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.intelligence = MaterialIntelligence(api_key=api_key)

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
        # Determine Tm factor if needed (passed to generators via config potentially)

        generators = []

        # 2. Select Strategies
        # Note: A material can trigger multiple strategies (e.g. Ionic + Covalent for some oxides)
        # But for now we pick the primary one.

        if mat_type == MaterialType.IONIC:
            generators.append(IonicGenerator(pmg_struct))
        elif mat_type == MaterialType.ALLOY:
            generators.append(AlloyGenerator(pmg_struct))
        elif mat_type == MaterialType.COVALENT:
            generators.append(CovalentGenerator(pmg_struct))
        elif mat_type == MaterialType.MOLECULAR:
            generators.append(MolecularGenerator(pmg_struct))
        else:
            # Default fallback: Treat as Alloy (Random/Chaos is good for learning) + Covalent (Distortion)
            generators.append(AlloyGenerator(pmg_struct))

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
        builder = InterfaceBuilder(struct_a, struct_b)
        return builder.generate_all()
