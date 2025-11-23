"""SSSP Pseudopotential Database Loader.

This module provides utilities for loading and using the SSSP (Standard Solid State
Pseudopotentials) database for automatic pseudopotential selection and DFT parameter
configuration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


def load_sssp_database(json_path: str) -> Dict[str, Dict[str, Any]]:
    """Load SSSP database from JSON file.
    
    Args:
        json_path: Path to SSSP JSON file (e.g., SSSP_1.3.0_PBE_precision.json)
        
    Returns:
        Dictionary mapping element symbols to their pseudopotential metadata.
        Each entry contains: filename, md5, type, cutoff_wfc, cutoff_rho
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON format is invalid
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"SSSP JSON file not found: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded SSSP database from {json_path}")
        logger.info(f"Available elements: {sorted(data.keys())}")
        
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_path}: {e}")


def get_pseudopotential_info(element: str, sssp_db: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Get pseudopotential information for a specific element.
    
    Args:
        element: Element symbol (e.g., 'Al', 'Cu')
        sssp_db: SSSP database loaded from JSON
        
    Returns:
        Dictionary with keys: filename, md5, type, cutoff_wfc, cutoff_rho
        
    Raises:
        KeyError: If element not found in database
    """
    if element not in sssp_db:
        available = sorted(sssp_db.keys())
        raise KeyError(
            f"Element '{element}' not found in SSSP database. "
            f"Available elements: {available}"
        )
    
    return sssp_db[element]


def calculate_cutoffs(elements: List[str], sssp_db: Dict[str, Dict[str, Any]]) -> Tuple[float, float]:
    """Calculate ecutwfc and ecutrho for a list of elements.
    
    For compounds, takes the maximum cutoff values across all elements
    to ensure accuracy for all species.
    
    Args:
        elements: List of element symbols
        sssp_db: SSSP database loaded from JSON
        
    Returns:
        Tuple of (ecutwfc, ecutrho) in Ry
        
    Raises:
        KeyError: If any element not found in database
    """
    if not elements:
        raise ValueError("Elements list cannot be empty")
    
    ecutwfc_values = []
    ecutrho_values = []
    
    for element in elements:
        info = get_pseudopotential_info(element, sssp_db)
        ecutwfc_values.append(info['cutoff_wfc'])
        ecutrho_values.append(info['cutoff_rho'])
    
    ecutwfc = max(ecutwfc_values)
    ecutrho = max(ecutrho_values)
    
    logger.info(f"Calculated cutoffs for {elements}: ecutwfc={ecutwfc} Ry, ecutrho={ecutrho} Ry")
    
    return ecutwfc, ecutrho


def get_pseudopotentials_dict(elements: List[str], sssp_db: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Get pseudopotential filenames for a list of elements.
    
    Args:
        elements: List of element symbols
        sssp_db: SSSP database loaded from JSON
        
    Returns:
        Dictionary mapping element symbols to pseudopotential filenames
        
    Raises:
        KeyError: If any element not found in database
    """
    pseudopotentials = {}
    
    for element in elements:
        info = get_pseudopotential_info(element, sssp_db)
        pseudopotentials[element] = info['filename']
    
    logger.debug(f"Pseudopotentials: {pseudopotentials}")
    
    return pseudopotentials


def calculate_kpoints(
    cell,
    kpoint_density: float = 60.0,
    min_kpoints: int = 1
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Calculate k-point grid based on cell dimensions using k-point density approach.
    
    Uses the rule: L × N ≈ kpoint_density (in Å)
    where L is the cell length and N is the number of k-points along that direction.
    
    Args:
        cell: ASE cell object or 3x3 array of lattice vectors
        kpoint_density: Target k-point density in Å (default: 60 for high precision)
        min_kpoints: Minimum number of k-points per direction (default: 1)
        
    Returns:
        Tuple of (kpts, shift) where:
            - kpts: (nx, ny, nz) k-point grid
            - shift: (1, 1, 1) for Monkhorst-Pack with offset
            
    Examples:
        >>> # Small cell (3 Å): 60/3 = 20 k-points
        >>> calculate_kpoints(cell_3A, kpoint_density=60)
        ((20, 20, 20), (1, 1, 1))
        
        >>> # Large cell (20 Å): 60/20 = 3 k-points
        >>> calculate_kpoints(cell_20A, kpoint_density=60)
        ((3, 3, 3), (1, 1, 1))
    """
    # Get cell lengths (magnitudes of lattice vectors)
    if hasattr(cell, 'lengths'):
        # ASE Cell object
        lengths = cell.lengths()
    else:
        # Assume 3x3 array
        lengths = np.linalg.norm(cell, axis=1)
    
    # Calculate k-points: N = kpoint_density / L
    kpts = []
    for length in lengths:
        if length > 0:
            n_k = max(min_kpoints, round(kpoint_density / length))
        else:
            # Non-periodic direction
            n_k = 1
        kpts.append(int(n_k))
    
    kpts_tuple = tuple(kpts)
    shift = (1, 1, 1)  # Monkhorst-Pack with offset
    
    logger.debug(
        f"K-points calculated: {kpts_tuple} (density={kpoint_density} Å, "
        f"cell lengths={lengths}, shift={shift})"
    )
    
    return kpts_tuple, shift


def validate_pseudopotentials(
    pseudo_dir: str,
    elements: List[str],
    sssp_db: Dict[str, Dict[str, Any]],
    check_md5: bool = False
) -> bool:
    """Validate that pseudopotential files exist in the specified directory.
    
    Args:
        pseudo_dir: Directory containing pseudopotential files
        elements: List of element symbols to check
        sssp_db: SSSP database loaded from JSON
        check_md5: If True, verify MD5 checksums (requires hashlib)
        
    Returns:
        True if all pseudopotentials are found (and valid if check_md5=True)
        
    Raises:
        FileNotFoundError: If any pseudopotential file is missing
        ValueError: If MD5 checksum doesn't match (when check_md5=True)
    """
    pseudo_dir = Path(pseudo_dir)
    
    if not pseudo_dir.exists():
        raise FileNotFoundError(f"Pseudopotential directory not found: {pseudo_dir}")
    
    for element in elements:
        info = get_pseudopotential_info(element, sssp_db)
        pp_file = pseudo_dir / info['filename']
        
        if not pp_file.exists():
            raise FileNotFoundError(
                f"Pseudopotential file not found: {pp_file}\n"
                f"Expected for element '{element}' from SSSP database."
            )
        
        if check_md5:
            import hashlib
            md5_hash = hashlib.md5()
            with open(pp_file, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
            
            calculated_md5 = md5_hash.hexdigest()
            expected_md5 = info['md5']
            
            if calculated_md5 != expected_md5:
                raise ValueError(
                    f"MD5 checksum mismatch for {pp_file}\n"
                    f"Expected: {expected_md5}\n"
                    f"Got: {calculated_md5}"
                )
    
    logger.info(f"All pseudopotentials validated for elements: {elements}")
    return True
