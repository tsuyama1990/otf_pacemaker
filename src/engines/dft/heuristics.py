"""Pymatgen-based Heuristics for Physical Parameters.

This module analyzes atomic structures to infer optimal physical parameters for DFT calculations,
such as smearing settings and initial magnetic moments.
It gracefully falls back to internal logic if pymatgen is not installed.
"""

import logging
from typing import Dict, Any, Set, List
from ase import Atoms
from ase.data import chemical_symbols

logger = logging.getLogger(__name__)

# Fallback Data
# Transition Metals (roughly Sc-Zn, Y-Cd, La-Hg)
FALLBACK_TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"
}

# Rare Earths (Lanthanides)
FALLBACK_RARE_EARTHS = {
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"
}

class PymatgenHeuristics:
    """Infers physical parameters using knowledge of materials physics."""

    # Typical magnetic elements and their initial moments
    MAGNETIC_ELEMENTS = {
        "Fe": 5.0, "Co": 2.0, "Ni": 1.0, "Mn": 5.0, "Cr": 3.0,
        "Eu": 7.0, "Gd": 7.0, "Tb": 6.0, "Dy": 10.0, "Ce": 1.0, "Nd": 3.0, "Sm": 1.0
    }

    # Anions used for oxide/insulator detection
    ANIONS = {"O", "F", "Cl", "S", "N", "P", "Br", "I", "Se", "Te"}

    @staticmethod
    def _is_transition_metal(symbol: str) -> bool:
        """Check if element is a transition metal."""
        try:
            from pymatgen.core import Element
            e = Element(symbol)
            if hasattr(e, "is_transition_metal"):
                return e.is_transition_metal
            return symbol in FALLBACK_TRANSITION_METALS
        except (ImportError, AttributeError):
            return symbol in FALLBACK_TRANSITION_METALS

    @staticmethod
    def _is_rare_earth(symbol: str) -> bool:
        """Check if element is a rare earth metal."""
        try:
            from pymatgen.core import Element
            e = Element(symbol)
            if hasattr(e, "is_rare_earth_metal"):
                return e.is_rare_earth_metal
            if hasattr(e, "is_rare_earth"):
                return e.is_rare_earth
            # Check lanthanoids + Sc, Y
            if symbol in {"Sc", "Y"}:
                return True
            if hasattr(e, "is_lanthanoid"):
                return e.is_lanthanoid
            return symbol in FALLBACK_RARE_EARTHS
        except (ImportError, AttributeError):
            return symbol in FALLBACK_RARE_EARTHS

    @classmethod
    def get_recommended_params(cls, atoms: Atoms) -> Dict[str, Any]:
        """Infer recommended DFT parameters for the given structure.

        Args:
            atoms: The ASE Atoms object to analyze.

        Returns:
            dict: Dictionary containing recommendations for 'system' and magnetic settings.
        """
        symbols = set(atoms.get_chemical_symbols())

        has_transition = any(cls._is_transition_metal(s) for s in symbols)
        has_rare_earth = any(cls._is_rare_earth(s) for s in symbols)
        has_anion = any(s in cls.ANIONS for s in symbols)
        has_magnetic = any(s in cls.MAGNETIC_ELEMENTS for s in symbols)

        recommendations = {
            "system": {},
            "magnetism": {
                "nspin": 1,
                "starting_magnetization": {}
            }
        }

        # 1. Smearing (DOS at Ef) Logic
        # Strategy:
        # - Transition/Rare Earth present -> Likely Metallic DOS -> Occupations = 'smearing'
        # - Anion present -> Likely Oxide/Semiconductor/Insulator -> Low Smearing (degauss=0.01)
        # - No Anion (Pure Metal/Alloy) -> Standard Smearing (degauss=0.02)
        # - No TM/RE and No Anion (e.g. Si, C) -> Could be anything, default to small smearing for safety or fixed if sure.
        #   "Safety-First" approach: defaulting to smearing is safer for convergence than fixed.

        if has_transition or has_rare_earth:
            recommendations["system"]["occupations"] = "smearing"
            recommendations["system"]["smearing"] = "mv" # Methfessel-Paxton is standard for metals

            if has_anion:
                # Likely oxide/insulator, sharpen the smearing
                recommendations["system"]["degauss"] = 0.01
            else:
                # Metal/Alloy
                recommendations["system"]["degauss"] = 0.02
        else:
            # Main group elements or simple s-block
            # Still safer to use small smearing for DFT convergence unless strictly insulating
            recommendations["system"]["occupations"] = "smearing"
            recommendations["system"]["smearing"] = "mv"
            recommendations["system"]["degauss"] = 0.01

        # 2. Magnetism Logic
        if has_magnetic:
            recommendations["magnetism"]["nspin"] = 2

            # Build starting_magnetization dict
            # In Quantum Espresso, starting_magnetization(i) is set per species index.
            # We return a map {Element: moment} which the Configurator will map to QE species indices.
            mag_map = {}
            for s in symbols:
                if s in cls.MAGNETIC_ELEMENTS:
                    mag_map[s] = cls.MAGNETIC_ELEMENTS[s]
                else:
                    mag_map[s] = 0.0 # Explicitly 0 for others if needed, or leave out

            recommendations["magnetism"]["moments"] = mag_map

        return recommendations
