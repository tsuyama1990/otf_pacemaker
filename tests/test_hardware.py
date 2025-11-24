import unittest
from unittest.mock import MagicMock, patch
from src.utils.hardware import get_available_vram, suggest_batch_size

class TestHardware(unittest.TestCase):

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.mem_get_info")
    def test_get_available_vram_cuda(self, mock_mem, mock_is_available):
        mock_is_available.return_value = True
        mock_mem.return_value = (8000, 16000) # free, total

        vram = get_available_vram()
        self.assertEqual(vram, 16000)

    @patch("torch.cuda.is_available")
    def test_get_available_vram_no_cuda(self, mock_is_available):
        mock_is_available.return_value = False

        vram = get_available_vram()
        self.assertEqual(vram, 0)

    def test_suggest_batch_size(self):
        GB = 1024**3
        self.assertEqual(suggest_batch_size(2 * GB), 16)
        self.assertEqual(suggest_batch_size(6 * GB), 32)
        self.assertEqual(suggest_batch_size(12 * GB), 64)
        self.assertEqual(suggest_batch_size(20 * GB), 128)
        self.assertEqual(suggest_batch_size(40 * GB), 256)
        self.assertEqual(suggest_batch_size(0), 32) # Fallback

if __name__ == "__main__":
    unittest.main()
