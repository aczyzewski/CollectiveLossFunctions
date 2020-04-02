import unittest
import numpy as np

from torch import Tensor
from src.functional import entropy


class TestFunctionals(unittest.TestCase):

    def test_zero_entropy(self):
        sample = Tensor(np.array([[0, 0, 0, 0, 0]]))
        self.assertEqual(entropy(sample), 0.)

    def test_mixed_entropy(self):
        sample = Tensor(np.array([[0, 0, 0, 0, 1, 1]]))
        expected_result = -1 * ((1/3) * np.log2(1/3) + (2/3) * np.log2(2/3))
        self.assertEqual(entropy(sample), expected_result)

    def test_max_entropy(self):
        sample = Tensor(np.array([[0, 0, 0, 1, 1, 1]]))
        self.assertEqual(entropy(sample), 1.)

    def test_multiple_class_entropy(self):
        num_classes = 5
        sample_size = 10
        sample = Tensor([np.random.choice(np.arange(num_classes), size=sample_size)])
        self.assertGreaterEqual(entropy(sample), 0.)
        self.assertLessEqual(entropy(sample), np.log(sample_size))


if __name__ == '__main__':
    unittest.main()
