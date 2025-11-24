"""Validator interface and implementation for Pacemaker potentials."""

import logging
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class Validator(ABC):
    """Interface for potential validation."""

    @abstractmethod
    def validate(self, potential_path: str) -> Dict[str, Any]:
        """Validate the potential and return metrics."""
        pass

class PacemakerValidator(Validator):
    """Validates ACE potentials using pace_diagnostics CLI."""

    def __init__(self, test_structure_path: Optional[str] = None):
        """Initialize validator.

        Args:
            test_structure_path: Path to structure file for calculating properties (e.g. elastic constants).
                                 If None, assumes pace_diagnostics uses internal or default data.
        """
        self.test_structure_path = test_structure_path

    def validate(self, potential_path: str) -> Dict[str, Any]:
        """Run pace_diagnostics and parse results.

        Args:
            potential_path: Path to the potential file (.yace).

        Returns:
            Dict[str, Any]: Validation metrics (Elastic Constants, VDoS, etc).
        """
        if not Path(potential_path).exists():
            raise FileNotFoundError(f"Potential not found: {potential_path}")

        results = {
            "elastic_constants": {},
            "vdos": {},
            "status": "UNKNOWN"
        }

        # Construct command
        # Assuming pace_diagnostics -p potential.yace -s structure.xyz
        cmd = ["pace_diagnostics", "-p", potential_path]
        if self.test_structure_path:
            cmd.extend(["-s", self.test_structure_path])

        # In a real scenario, pace_diagnostics might output to stdout or a file.
        # We'll assume stdout contains parseable YAML or text.

        logger.info(f"Running validation: {' '.join(cmd)}")
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = process.stdout

            # Simple parsing of hypothetical output
            # Output format assumed to be YAML-like or key-value pairs

            # Example hypothetical output:
            # Elastic Constants:
            #   C11: 200.0
            #   C12: 150.0
            # VDoS:
            #   Peak: 12.5

            # For robustness, let's look for known keywords or try to parse generic YAML if possible.
            try:
                data = yaml.safe_load(output)
                if isinstance(data, dict):
                    results.update(data)
                    results["status"] = "SUCCESS"
            except yaml.YAMLError:
                # Fallback text parsing
                for line in output.splitlines():
                    if "C11" in line:
                        results["elastic_constants"]["C11"] = float(line.split(":")[-1])
                    if "status" in line.lower():
                        results["status"] = line.split(":")[-1].strip()

        except subprocess.CalledProcessError as e:
            logger.error(f"Validation failed: {e.stderr}")
            results["status"] = "FAILED"
            results["error"] = e.stderr

        return results
