import unittest

import numpy as np
import torch
from torch import nn, Tensor

from src.neighboorhood import FaissKNN
from src.losses import MSE, MAE, VarianceWeightedMSE, VarianceWeightedMAE, \
    CollectiveMSE, CollectiveMAE


class TestRegressionLosses(unittest.TestCase):

    example_dataset = np.array([
        [0., 1., 0., 1., 60.],   # K = 2 (D = 2)
        [1., 1., 1., 1., 100.],
        [0., 0., 0., 0., 60.],   # INPUTS
        [1., 0., 1., 1., 80.],   # K = 3 (D = 3)
        [1., 0., 0., 0., 20.]    # K = 1 (D = 1)
    ])
    # NEAREST_VALUES = [20, 60, 80]

    x = example_dataset[:, :-1]
    y = example_dataset[:, -1].reshape(-1, 1)
    knn = FaissKNN(x, y)

    # Helpers
    def tensorsAlmostEqual(self, a: Tensor, b: Tensor,
                           eps: float = 1e-4) -> bool:
        return (torch.abs((a - b)) < eps).all().item()

    def test_mse(self):
        mse, custom_mse = nn.MSELoss(), MSE()
        target = Tensor([10, 20, 30]).reshape(-1, 1)
        prediction = Tensor([20, 30, 40]).reshape(-1, 1)
        self.assertEqual(mse(prediction, target), custom_mse(prediction, target))

    def test_mae(self):
        mae, custom_mae = nn.L1Loss(), MAE()
        target = Tensor([10, 20, 30]).reshape(-1, 1)
        prediction = Tensor([20, 30, 40]).reshape(-1, 1)
        self.assertEqual(mae(prediction, target), custom_mae(prediction, target))

    # def test_var_weighted_mse(self):
    #     loss = VarianceWeightedMSE(self.knn)
    #     target = Tensor([[60]])
    #     prediction = Tensor([[40]])
    #     inputs = Tensor(self.x[2].reshape(1, -1))
    #     calcuated_loss = loss(prediction, target, inputs)

    # def test_var_weighted_mae(self):
    #     loss = VarianceWeightedMAE(self.knn)
    #     target = Tensor([[60]])
    #     prediction = Tensor([[40]])
    #     inputs = Tensor(self.x[2].reshape(1, -1))
    #     calcuated_loss = loss(prediction, target, inputs)

    def test_collective_mse(self):
        loss = CollectiveMSE(self.knn, alpha=0.5)
        target = Tensor([[60]])
        prediction = Tensor([[40]])
        inputs = Tensor(self.x[2].reshape(1, -1))
        calculated_loss = loss(prediction, target, inputs)
        self.assertTrue(self.tensorsAlmostEqual(calculated_loss, Tensor([488.88888])))

    def test_collective_mae(self):
        loss = CollectiveMAE(self.knn, alpha=0.5)
        target = Tensor([[60]])
        prediction = Tensor([[40]])
        inputs = Tensor(self.x[2].reshape(1, -1))
        calculated_loss = loss(prediction, target, inputs)
        self.assertTrue(self.tensorsAlmostEqual(calculated_loss, Tensor([26.66666])))
