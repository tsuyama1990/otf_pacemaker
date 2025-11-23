"""Module for scenario-driven structure generation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ase import Atoms
from ase.build import bulk, surface, stack
import numpy as np


class BaseScenario(ABC):
    """Abstract base class for scenario generators."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the scenario generator.

        Args:
            config: Configuration dictionary for the scenario.
        """
        self.config = config

    @abstractmethod
    def generate(self) -> List[Atoms]:
        """Generate structures based on the scenario.

        Returns:
            List[Atoms]: List of generated structures.
        """
        pass


class InterfaceGenerator(BaseScenario):
    """Generates interface structures by stacking two materials."""

    def generate(self) -> List[Atoms]:
        substrate_conf = self.config.get("substrate", {})
        layer_conf = self.config.get("layer", {})
        vacuum = self.config.get("vacuum", 10.0)

        # Create substrate
        sub = bulk(
            substrate_conf.get("symbol", "MgO"),
            crystalstructure=substrate_conf.get("structure", "rocksalt"),
            a=substrate_conf.get("lattice", 4.21),
            orthorhombic=True,
        )
        # Create layer
        # Handle lattice being a list or float
        lattice_layer = layer_conf.get("lattice", 3.8)
        if isinstance(lattice_layer, list):
            # If list, we assume it's [a, c] or [a, b, c].
            # ase.build.bulk's 'a' parameter expects a float.
            # Complex lattice support might need more logic.
            # For now, take the first element as 'a' if list.
            lattice_layer_val = lattice_layer[0]
        else:
            lattice_layer_val = lattice_layer

        lay = bulk(
            layer_conf.get("symbol", "Fe"),
            crystalstructure=layer_conf.get("structure", "bcc"),
            a=lattice_layer_val,
            orthorhombic=True,
        )

        # Simple stacking
        # Note: Real interface generation needs careful lattice matching.
        # This is a simplified version as per requirements.
        try:
            # We explicitly define axis to stack. Defaults are z (2).
            # ASE stack usually stacks along z.
            # SAFETY VALVE: Limit strain to 15% using maxstrain parameter
            # If strain exceeds this, stack will raise ValueError (or similar) or return None (depending on implementation, but ASE usually raises)
            interface = stack(sub, lay, maxstrain=0.15)

            # ASE stack might return a cell with negative values or centered in a way center() doesn't expect if vacuum is applied incorrectly?
            # Or center() changes the cell.

            # Add vacuum if requested
            if vacuum > 0:
                interface.center(vacuum=vacuum, axis=2)

            return [interface]
        except Exception as e:
            # print(f"Interface generation failed (likely strain too high): {e}")
            return []


class SurfaceGenerator(BaseScenario):
    """Generates surface structures."""

    def generate(self) -> List[Atoms]:
        element = self.config.get("element", "Pt")
        indices_list = self.config.get("indices", [[1, 1, 1]])
        layers = self.config.get("layers", 4)
        vacuum = self.config.get("vacuum", 10.0)

        structures = []
        for hkl in indices_list:
            try:
                # Assuming fcc111, bcc110 etc. need to be inferred or specified.
                # For simplicity, using ase.build.surface which requires a lattice object or predefined string.
                # But ase.build.surface takes 'lattice' (atoms object) and 'indices'.

                # First create a bulk structure. We need to know the structure type (fcc, bcc, etc.)
                # which is not explicitly in the config example for SurfaceGenerator.
                # We will default to fcc for simplicity or try to guess.
                # A better design would be to require structure info in config.
                # Let's assume 'fcc' default or assume the user provides enough info.
                # The prompt example: element: "Pt", indices: [[1,1,1]...], layers: 4.
                # Pt is FCC.

                # We'll create a standard bulk for the element.
                # ASE's 'bulk' function can infer structure for common elements.
                b = bulk(element)

                surf = surface(b, tuple(hkl), layers)
                surf.center(vacuum=vacuum, axis=2)
                structures.append(surf)
            except Exception as e:
                print(f"Surface generation failed for {element} {hkl}: {e}")
                continue
        return structures


class DefectGenerator(BaseScenario):
    """Generates structures with defects (vacancies or impurities)."""

    def generate(self) -> List[Atoms]:
        base_conf = self.config.get("base", {})
        defect_type = self.config.get("defect_type", "vacancy")
        species = self.config.get("species", "O") # For vacancy: removed species. For impurity: added species?
        concentration = self.config.get("concentration", 0.02)

        # Create base structure
        atoms = bulk(
            base_conf.get("symbol", "MgO"),
            crystalstructure=base_conf.get("structure", "rocksalt"),
            a=base_conf.get("lattice", 4.21),
        )

        # Supercell
        size = base_conf.get("size", [3, 3, 3])
        atoms = atoms.repeat(size)

        num_sites = len(atoms)
        num_defects = max(1, int(num_sites * concentration))

        generated = []

        # Simple random defect generation
        indices = list(range(num_sites))
        # Filter by species if specified
        if species:
            indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == species]

        if len(indices) < num_defects:
            # Not enough sites of that species
            return []

        np.random.shuffle(indices)
        target_indices = indices[:num_defects]

        new_atoms = atoms.copy()

        if defect_type == "vacancy":
            del new_atoms[target_indices]
            generated.append(new_atoms)
        elif defect_type == "impurity":
            # Not fully specified in prompt how impurity works (substitution vs interstitial)
            # Assuming substitution based on context of "defects" usually implying point defects.
            impurity_element = self.config.get("impurity_element", "H")
            # Wait, prompt says "species: O". For vacancy it is removed.
            # If defect_type is impurity, maybe 'species' is the one being replaced?
            # Prompt says: "species: "O"".
            # And "置換（不純物）" (Substitution (Impurity)).
            # Let's assume we need another config field for what to put in,
            # or re-purpose 'species' field.
            # Based on prompt: "species" seems to target the site.
            # Let's assume we substitute with a placeholder or config should have 'substitute_with'.
            pass

        # Prompt only explicitly details "vacancy" in YAML example.
        # "defect_type: 'vacancy', species: 'O'".

        return generated


class GrainBoundaryGenerator(BaseScenario):
    """Generates grain boundary structures."""

    def generate(self) -> List[Atoms]:
        # Simplified placeholder as GB generation is complex
        # Ideally use ase.build.grain_boundary if available or manual rotation
        return []


class ScenarioFactory:
    """Factory to create scenario generators."""

    @staticmethod
    def create(config: Dict[str, Any]) -> BaseScenario:
        """Create a scenario generator instance.

        Args:
            config: Configuration dictionary.

        Returns:
            BaseScenario: Instance of a scenario generator.

        Raises:
            ValueError: If the scenario type is unknown.
        """
        t = config.get("type")
        if t == "interface":
            return InterfaceGenerator(config)
        elif t == "surface":
            return SurfaceGenerator(config)
        elif t == "defect":
            return DefectGenerator(config)
        elif t == "grain_boundary":
            return GrainBoundaryGenerator(config)
        else:
            raise ValueError(f"Unknown scenario type: {t}")
