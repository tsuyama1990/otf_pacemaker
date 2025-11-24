import unittest
import subprocess
from unittest.mock import MagicMock, patch
from src.validation.pacemaker_validator import PacemakerValidator

class TestPacemakerValidator(unittest.TestCase):
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_validate_success(self, mock_exists, mock_run):
        mock_exists.return_value = True

        # Mock successful output (YAML format)
        mock_process = MagicMock()
        mock_process.stdout = """
        elastic_constants:
          C11: 250.0
          C12: 140.0
        vdos:
          peak_freq: 10.5
        """
        mock_run.return_value = mock_process

        validator = PacemakerValidator()
        results = validator.validate("potential.yace")

        self.assertEqual(results["elastic_constants"]["C11"], 250.0)
        self.assertEqual(results["status"], "SUCCESS")

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_validate_failure(self, mock_exists, mock_run):
        mock_exists.return_value = True

        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="Error calc")

        validator = PacemakerValidator()
        results = validator.validate("potential.yace")

        self.assertEqual(results["status"], "FAILED")
        self.assertIn("Error calc", results["error"])

if __name__ == "__main__":
    unittest.main()
