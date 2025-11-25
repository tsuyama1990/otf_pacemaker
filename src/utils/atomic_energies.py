import yaml
import logging
from pathlib import Path
from typing import Dict, List, Callable
from ase import Atoms
from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)

class AtomicEnergyManager:
    """Manages isolated atomic energies (E0) for Delta learning."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path

    def get_e0(self, elements: List[str], calculator_factory: Callable[[str], Calculator]) -> Dict[str, float]:
        """
        Loads E0 from file or calculates them if missing.
        calculator_factory: Function that accepts an element symbol and returns a new Calculator instance.
        """
        if self.storage_path.exists():
            logger.info(f"Loading E0 from {self.storage_path}")
            with open(self.storage_path, 'r') as f:
                return yaml.safe_load(f)

        logger.info("E0 file not found. Calculating isolated atomic energies...")
        e0_dict = {}
        for el in elements:
            # Create isolated atom in a large box
            atom = Atoms(el, cell=[15.0, 15.0, 15.0], pbc=True)
            atom.center()

            calc = calculator_factory(el)
            atom.calc = calc

            try:
                e = atom.get_potential_energy()
                e0_dict[el] = float(e)
                logger.info(f"Calculated E0 for {el}: {e:.4f} eV")
            except Exception as e:
                logger.error(f"Failed to calculate E0 for {el}: {e}")
                raise e

        # Save to file
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            yaml.dump(e0_dict, f)

        return e0_dict
