"""Calculators for the project."""
from typing import List, Optional, Dict
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms

class SumCalculator(Calculator):
    """Calculator that sums the results of multiple calculators."""

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, calculators: List[Calculator], e0: Optional[Dict[str, float]] = None):
        super().__init__()
        self.calculators = calculators
        self.e0 = e0 or {}

    def calculate(self, atoms: Atoms = None, properties: List[str] = ['energy'],
                  system_changes: List[str] = all_changes):
        super().calculate(atoms, properties, system_changes)

        self.results = {
            'energy': 0.0,
            'forces': np.zeros((len(atoms), 3)),
            'stress': np.zeros(6)
        }

        # Sum contributions from calculators
        for calc in self.calculators:
            # Trigger calculation by calling get_potential_energy on the atoms object.
            # We assume 'atoms' is consistent.
            # However, calc might not be attached to 'atoms'.
            # We temporarily attach calc to a copy? No, that's slow.
            # We can use calc.get_potential_energy(atoms=atoms) if supported.
            # Most ASE calculators support this.

            # Note: We need to handle exceptions if a property is not implemented
            try:
                e = calc.get_potential_energy(atoms)
                self.results['energy'] += e
            except Exception:
                pass

            try:
                f = calc.get_forces(atoms)
                self.results['forces'] += f
            except Exception:
                pass

            try:
                 # get_stress returns Voigt (6,) array
                s = calc.get_stress(atoms)
                self.results['stress'] += s
            except Exception:
                pass

        # Add E0 contribution
        if self.e0:
            e_offset = sum(self.e0.get(sym, 0.0) for sym in atoms.get_chemical_symbols())
            self.results['energy'] += e_offset
