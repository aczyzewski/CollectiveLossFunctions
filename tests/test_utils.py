import unittest
import torch
from torch import Tensor
from src.utils import get_reduction_method, \
    convert_logits_to_class_distribution


class TestUtils(unittest.TestCase):

    # Helpers
    def tensorsAlmostEqual(self, a: Tensor, b: Tensor, eps: float = 1e-4
                           ) -> bool:
        return (torch.abs((a - b)) < eps).all().item()

    def test_get_reduction_method(self):
        values = Tensor([1., 3., 5.]).reshape(-1, 1)

        self.assertRaises(KeyError, get_reduction_method, 'invalid_method')
        self.assertEqual(get_reduction_method('mean')(values).item(), 3.)
        self.assertEqual(get_reduction_method('sum')(values).item(), 9.)
        self.assertTrue(
            torch.eq(get_reduction_method('none')(values), values).all().item()
        )

    def test_convert_logits_to_class_distribution(self):

        a = Tensor([[0, 1, 2, 3], [1, 1, 2, 2], [0, 0, 0, 3], [0, 0, 0, 0]])
        no_classes = 4
        output = convert_logits_to_class_distribution(a, no_classes)
        valid_output = Tensor(
            [[0.25, 0.25, 0.25, 0.25],
             [0.00, 0.50, 0.50, 0.00],
             [0.75, 0.00, 0.00, 0.25],
             [1.00, 0.00, 0.00, 0.00]]
        )

        self.assertTrue(self.tensorsAlmostEqual(output, valid_output))


if __name__ == '__main__':
    unittest.main()
