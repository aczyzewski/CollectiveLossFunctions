import unittest
import numpy as np
import torch

from src.losses import entropy


class TestLosses(unittest.TestCase):

    def test_zero_entropy(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 0, 0]]))
        self.assertEqual(entropy(sample), 0.)

    def test_max_entropy(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 1, 1, 1]]))
        self.assertEqual(entropy(sample), 1.)

    def test_multiple_class_entropy(self):
        num_classes = 5
        sample_size = 10
        sample = torch.Tensor([np.random.choice(np.arange(num_classes), size=sample_size)])
        self.assertGreaterEqual(entropy(sample), 0.)
        self.assertLessEqual(entropy(sample), np.log(sample_size))


if __name__ == '__main__':
    unittest.main()
