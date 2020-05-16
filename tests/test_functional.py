import unittest
import numpy as np
import torch

from src.functional import entropy, theil, gini, atkinson, kl_divergence, \
    scaled_variance


class TestFunctionals(unittest.TestCase):

    def test_zero_entropy(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 0, 0]]))
        self.assertEqual(entropy(sample), 0.)

    def test_mixed_entropy(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 0, 1, 1]]))
        expected_result = -1 * ((1/3) * np.log2(1/3) + (2/3) * np.log2(2/3))
        self.assertEqual(entropy(sample), expected_result)

    def test_max_entropy(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 1, 1, 1]]))
        self.assertEqual(entropy(sample), 1.)

    def test_multiple_class_entropy(self):
        num_classes = 5
        sample_size = 10
        sample = torch.Tensor([np.random.choice(np.arange(num_classes), size=sample_size)])
        self.assertGreaterEqual(entropy(sample), 0.)
        self.assertLessEqual(entropy(sample), np.log(sample_size))

    def test_entropy_multiple_vectors(self):
        sample = torch.Tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
        self.assertTrue(torch.all(torch.eq(entropy(sample), torch.Tensor([0., 0.]))))

    def test_weighted_entropy(self):
        sample = torch.Tensor([[0, 0, 1, 1]])
        distances = torch.Tensor([[1, 1, 2, 2]])
        self.assertLess(entropy(sample, distances, use_weights=True), 1.)

        sample = torch.Tensor([[0, 0, 0, 1, 1, 1]])
        distances = torch.Tensor([[1, 2, 3, 1, 2, 3]])
        self.assertEqual(entropy(sample, distances, use_weights=True), 1.)

    def test_weighted_entropy_without_distances(self):
        sample = torch.Tensor([[0, 0, 1, 1]])
        with self.assertRaises(AssertionError):
            entropy(sample, use_weights=True)

    def test_kl_divergence_identical_distributions(self):
        predictions = torch.Tensor([[.75, .25]])
        neighbors = torch.Tensor([[0, 0, 0, 1]])
        self.assertEqual(kl_divergence(predictions, neighbors), 0.)

    def test_kl_divergence_almost_identical_distributions(self):
        predictions = torch.Tensor([[.9, .1]])
        neighbors = torch.Tensor([[0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 1]])
        self.assertLess(torch.abs(kl_divergence(predictions, neighbors) - 0.), 0.001)

    def test_kl_divergence_non_identical_distributions(self):
        predictions = torch.Tensor([[.75, .25]])
        neighbors = torch.Tensor([[0, 0, 0, 0, 0, 1]])
        self.assertGreater(kl_divergence(predictions, neighbors), 0.)

    def test_kl_divergence_large_difference(self):
        predictions = torch.Tensor([[.1, .9]])
        neighbors = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.assertGreater(kl_divergence(predictions, neighbors), 1.)

    def test_kl_divergence_multiple_vectors(self):
        predictions = torch.Tensor([[.75, .25], [.5, .5]])
        neighbors = torch.Tensor([[0, 0, 0, 1], [0, 0, 1, 1]])
        self.assertTrue(torch.all(torch.eq(kl_divergence(predictions, neighbors), torch.Tensor([0., 0.]))))

    def test_zeros_theil(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 0, 0]]))
        self.assertEqual(theil(sample), 0.)

    def test_ones_theil(self):
        sample = torch.Tensor(np.array([[1, 1, 1, 1, 1]]))
        self.assertEqual(theil(sample), 0.)

    def test_equal_mix_theil(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 1, 1, 1]]))
        self.assertEqual(theil(sample), 0.)

    def test_random_theil(self):
        num_classes = 5
        sample_size = 10
        sample = torch.Tensor([np.random.choice(np.arange(num_classes), size=sample_size)])
        self.assertGreaterEqual(theil(sample), 0.)
        self.assertLessEqual(theil(sample), 1.)

    def test_theil(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 0, 0, 1]]))
        self.assertTrue(abs(theil(sample).item() - (1/2) * ((5/3) * np.log(5/3) + (1/3) * np.log(1/3))) < 1e-7)

    def test_theil_multiple_vectors(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 0], [1, 1, 1, 1]]))
        self.assertTrue(torch.all(torch.eq(theil(sample), torch.Tensor([0., 0.]))))

    def test_zeros_gini(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 0, 0]]))
        self.assertEqual(gini(sample), 0.)

    def test_ones_gini(self):
        sample = torch.Tensor(np.array([[1, 1, 1, 1, 1, 1]]))
        self.assertEqual(gini(sample), 0.)

    def test_random_gini(self):
        num_classes = 2
        sample_size = 10
        sample = torch.Tensor([np.random.choice(np.arange(num_classes), size=sample_size)])
        self.assertGreaterEqual(gini(sample), 0.)
        self.assertLessEqual(gini(sample), 1.)

    def test_gini_multiple_vectors(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 0], [1, 1, 1, 1]]))
        self.assertTrue(torch.all(torch.eq(gini(sample), torch.Tensor([0., 0.]))))

    def test_zeros_atkinson(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 0, 0]]))
        self.assertEqual(atkinson(sample), 0.)

    def test_ones_atkinson(self):
        sample = torch.Tensor(np.array([[1, 1, 1, 1, 1, 1]]))
        self.assertEqual(atkinson(sample), 0.)

    def test_random_atkinson(self):
        num_classes = 2
        sample_size = 10
        sample = torch.Tensor([np.random.choice(np.arange(num_classes), size=sample_size)])
        self.assertGreaterEqual(atkinson(sample), 0.)
        self.assertLessEqual(atkinson(sample), 1.)

    def test_atkinson_multiple_vectors(self):
        sample = torch.Tensor(np.array([[0, 0, 0, 0], [1, 1, 1, 1]]))
        self.assertTrue(torch.all(torch.eq(atkinson(sample), torch.Tensor([0., 0.]))))

    def test_scaled_variance(self):
        mean = torch.Tensor([60.])
        values = torch.Tensor([[20, 40, 80, 100]])
        result = scaled_variance(mean, values)
        self.assertEqual(result, torch.Tensor([111.]))


if __name__ == '__main__':
    unittest.main()
