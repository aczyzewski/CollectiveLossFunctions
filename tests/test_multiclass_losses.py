import unittest

import numpy as np
import torch
from torch import nn
from src.neighboorhood import FaissKNN
from src.losses import CrossEntropy, WeightedCrossEntropy, \
    RegularizedCrossEntropy, CollectiveCrossEntropy


class TestMulticlassLosses(unittest.TestCase):

    example_dataset = np.array([
        [0., 1., 0., 1., 1.],   # K = 2 (D = 2)
        [1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0.],   # INPUTS
        [1., 0., 1., 1., 2.],   # K = 3 (D = 3)
        [1., 0., 0., 0., 0.]    # K = 1 (D = 1)
    ])
    # CLASSES = []

    x = example_dataset[:, :-1]
    y = example_dataset[:, -1].reshape(-1, 1)
    knn = FaissKNN(x, y)

    def test_cross_entropy(self):
        ce, custom_ce = nn.CrossEntropyLoss(), CrossEntropy()

        prediction = torch.zeros((3, 3)) + 0.5
        target = torch.Tensor([2, 0, 1]).long()
        calculated_loss = custom_ce(prediction, target.reshape(-1, 1)).item()
        expected_loss = ce(prediction, target).item()

        self.assertEqual(calculated_loss, expected_loss)

    def test_weighted_cross_entropy(self):
        loss = WeightedCrossEntropy(self.knn)

        prediction = torch.zeros((1, 3)) + 0.5
        target = torch.Tensor([0]).long()
        inputs = torch.Tensor(self.x[2].reshape(1, -1))

        calculated_loss = loss(prediction, target.reshape(-1, 1), inputs)
        self.assertAlmostEqual(calculated_loss.item(), 0.225166589)

    def test_regularized_cross_entropy(self):
        loss = RegularizedCrossEntropy(self.knn)

        prediction = torch.zeros((1, 3)) + 0.5
        target = torch.Tensor([0]).long()
        inputs = torch.Tensor(self.x[2].reshape(1, -1))

        calculated_loss = loss(prediction, target.reshape(-1, 1), inputs)
        self.assertAlmostEqual(calculated_loss.item(), 1.0986123085)

    def test_collective_cross_entropy(self):
        loss = CollectiveCrossEntropy(self.knn)

        prediction = torch.zeros((1, 3)) + 0.5
        target = torch.Tensor([0]).long()
        inputs = torch.Tensor(self.x[2].reshape(1, -1))

        calculated_loss = loss(prediction, target.reshape(-1, 1), inputs)
        self.assertAlmostEqual(calculated_loss.item(), 1.647918343)
