import unittest
import torch
from torch import Tensor
from src.utils import get_reduction_method


class TestUtils(unittest.TestCase):

    def test_get_reduction_method(self):
        values = Tensor([1., 3., 5.]).reshape(-1, 1)

        self.assertRaises(KeyError, get_reduction_method, 'invalid_method')
        self.assertEqual(get_reduction_method('mean')(values).item(), 3.)
        self.assertEqual(get_reduction_method('sum')(values).item(), 9.)
        self.assertTrue(
            torch.eq(get_reduction_method('none')(values), values).all().item()
        )


if __name__ == '__main__':
    unittest.main()
