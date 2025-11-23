import numpy as np
from typing import List
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.units import fs

from .base import BaseGenerator

class IonicGenerator(BaseGenerator):
    """
    Strategy for Ionic Crystals (e.g., NaCl, MgO, LiCoO2).
    Focus: Charge balance, Polar surfaces, High-T fluctuations.
    """

    def generate_charged_defects(self):
        """
        Generates charged defects (Vacancies) ignoring charge compensation.
        Explicitly sets 'charge' in atoms.info for DFT.
        """
        # Create a supercell for defects
        sc_struct = self.structure.copy()
        sc_struct.make_supercell([2, 2, 2])

        # Determine unique sites to avoid redundancy
        # Using symmetry analyzer
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        try:
            # Attempt to guess oxidation states if not present
            if not sc_struct.site_properties.get("oxidation_state"):
                sc_struct.add_oxidation_state_by_guess()
        except:
            pass # Can't guess, proceed with 0 or warn

        sga = SpacegroupAnalyzer(sc_struct)
        symm_structure = sga.get_symmetrized_structure()

        # Iterate over unique equivalent indices
        unique_indices = [equiv[0] for equiv in symm_structure.equivalent_indices]

        for idx in unique_indices:
            # Create vacancy by removing site at idx
            d_struct = sc_struct.copy()
            removed_site = d_struct[idx]
            d_struct.remove_sites([idx])

            # Calculate charge imbalance
            # Removed species
            oxi = 0
            if hasattr(removed_site.specie, "oxi_state"):
                 oxi = removed_site.specie.oxi_state
            elif getattr(removed_site.specie, "oxidation_state", None):
                 # Sometimes it's a property
                 oxi = removed_site.specie.oxidation_state

            # If we remove +2, system charge is -2
            system_charge = -oxi

            self._add_structure(d_struct, meta={
                "type": "charged_defect",
                "charge": system_charge,
                "defect_type": "vacancy",
                "removed_element": str(removed_site.specie.symbol)
            })

    def generate_polar_surfaces(self):
        """
        Generates polar surfaces to encourage reconstruction.
        """
        from pymatgen.core.surface import SlabGenerator

        # Identify polar planes?
        # Simple strategy: Generate low index surfaces (100, 110, 111)
        # 111 is often polar for Rocksalt/Zincblende

        miller_indices = [(1,0,0), (1,1,0), (1,1,1)]

        for miller in miller_indices:
            try:
                slabgen = SlabGenerator(self.structure, miller, min_slab_size=10, min_vacuum_size=12, center_slab=True)
                slabs = slabgen.get_slabs(bonds=None, ftol=0.1, tol=0.1)

                for slab in slabs:
                    # Check polarity if possible, or just add all as candidates
                    # Note: We want "unstable" ones too for training

                    # For polar surfaces, we might want to make them non-stoichiometric or standard
                    # The requirement says "create polar surfaces... to encourage reconstruction"
                    # So raw cuts are good.
                    self._add_structure(slab, meta={"type": "surface", "miller": miller, "polar_candidate": True})
            except Exception as e:
                # Some crystals might fail generation for specific indices
                continue

    def high_temp_md_snapshots(self, T_m: float = 1000.0):
        """
        MD snapshots at 0.6 ~ 0.8 Tm.
        """
        # Convert to ASE
        atoms = self.pre_optimizer.run_pre_optimization(AseAtomsAdaptor.get_atoms(self.structure))
        atoms *= (2,2,2) # Supercell for MD

        target_temps = [0.6 * T_m, 0.8 * T_m]

        for T in target_temps:
            atoms_md = atoms.copy()
            atoms_md.calc = self.pre_optimizer.emt_elements.issubset(set(atoms_md.get_chemical_symbols())) and self.pre_optimizer.run_pre_optimization(atoms).calc or None

            # If we don't have a cheap force field (EMT), we can't run MD easily in this builder
            # The prompt implies WE build the dataset FOR MLIP training.
            # Using MD to generate structures requires a potential.
            # "High-temp MD snapshots... (amplitude is large)"
            # If we have no potential, we can simulate this via Rattling (Monte Carlo) or
            # just simple Debye-like thermal displacement.

            # Since running actual MD requires a potential (and we are building the dataset FOR the potential),
            # the standard approach in "Active Learning Start" is to use "Rattling".
            # However, the user explicitly asked for "MD snapshots".
            # If "Pre-optimization" uses LJ, we can run MD with LJ.

            # We use the fallback LJ from PreOptimizer for this MD.
            try:
                # PreOptimizer works on a copy and returns it with calculator attached/relaxed
                atoms_md = self.pre_optimizer.run_pre_optimization(atoms_md)
            except:
                pass

            # Run short MD
            if atoms_md.calc:
                MaxwellBoltzmannDistribution(atoms_md, temperature_K=T)
                dyn = Langevin(atoms_md, 0.5*fs, temperature_K=T, friction=0.02, logfile=None)

                # Take snapshots
                for step in range(5):
                    dyn.run(50) # Run 50 steps
                    self._add_structure(atoms_md.copy(), meta={"type": "md_snapshot", "temperature": T})

    def generate_all(self) -> List[Atoms]:
        self.generate_charged_defects()
        self.generate_polar_surfaces()
        # Assume generic Tm if unknown, e.g. 1500K
        self.high_temp_md_snapshots(T_m=1500.0)
        return self.generated_structures
