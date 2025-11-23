import os
import logging
from typing import Dict, Optional, Union
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from enum import Enum

logger = logging.getLogger(__name__)

class MaterialType(Enum):
    IONIC = "ionic"
    ALLOY = "alloy" # Metallic/HEAs
    COVALENT = "covalent"
    MOLECULAR = "molecular"
    UNKNOWN = "unknown"

class MaterialIntelligence:
    """
    Intelligent classifier for materials to determine generation strategy.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        if not self.api_key:
            logger.warning("MP_API_KEY not found. MaterialIntelligence will run in heuristic mode only.")

    def _get_mp_data(self, formula_or_id: str):
        """Fetches data from Materials Project."""
        if not self.api_key:
            return None

        try:
            # Delayed import to avoid environment issues if mp_api is broken or optional
            from mp_api.client import MPRester

            with MPRester(self.api_key) as mpr:
                # Determine if it's an ID or Formula
                if "mp-" in formula_or_id:
                     docs = mpr.materials.summary.search(material_ids=[formula_or_id],
                                                         fields=["structure", "band_gap", "is_metal", "is_magnetic", "elements"])
                else:
                     docs = mpr.materials.summary.search(formula=[formula_or_id],
                                                         fields=["structure", "band_gap", "is_metal", "is_magnetic", "elements"])

                if docs:
                    return docs[0] # Return best match
        except ImportError:
            logger.error("mp-api not installed or incompatible.")
            return None
        except Exception as e:
            logger.error(f"MP API Error: {e}")
            return None
        return None

    def analyze(self, structure: Structure, formula_or_id: str = None) -> MaterialType:
        """
        Determines the material type based on structure and optional MP data.

        Args:
            structure: Pymatgen Structure object.
            formula_or_id: Chemical formula or MP-ID (optional, for lookup).

        Returns:
            MaterialType enum.
        """
        # 1. Check MP Data if available
        mp_data = None
        if formula_or_id:
            mp_data = self._get_mp_data(formula_or_id)

        if mp_data:
            # MP Classifier Logic
            if mp_data.is_metal:
                return MaterialType.ALLOY
            if mp_data.band_gap > 3.0: # Wide gap often Ionic
                 # Need deeper check for molecular
                 pass

        # 2. Heuristic Logic (Structure + Composition)

        # Check for single element -> Likely Covalent or Metallic
        if len(structure.composition.elements) == 1:
            el = structure.composition.elements[0]
            if el.is_metal:
                return MaterialType.ALLOY
            elif el.symbol in ["C", "Si", "Ge", "B"]:
                return MaterialType.COVALENT

        # Check for Molecules (Graph based)
        # Using CrystalNN or checking for isolated clusters
        # Simple heuristic: weak bonds or large voids?
        # Better: let MolecularGenerator's detailed check decide, but here we need high level.
        # We can check density or packing fraction.

        # Check Electronegativity Difference for Ionic
        elements = structure.composition.elements
        if len(elements) > 1:
            chi = [e.X for e in elements]
            max_diff = max(chi) - min(chi)
            if max_diff > 1.7:
                return MaterialType.IONIC
            elif max_diff < 0.5 and all(e.is_metal for e in elements):
                return MaterialType.ALLOY
            elif all(not e.is_metal for e in elements):
                 # Could be covalent or molecular
                 # Simple molecular check: distances
                 return MaterialType.COVALENT # Fallback, needs refinement

        return MaterialType.UNKNOWN

    def get_melting_point_factor(self, mat_type: MaterialType) -> float:
        """Returns safety factor for melting point based on type."""
        mapping = {
            MaterialType.MOLECULAR: 0.4,
            MaterialType.IONIC: 1.0, # Standard
            MaterialType.ALLOY: 0.9,
            MaterialType.COVALENT: 1.1,
            MaterialType.UNKNOWN: 0.8
        }
        return mapping.get(mat_type, 0.8)
