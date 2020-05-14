import unittest

import numpy as np
import torch
from torch import Tensor

import src.losses as L
from src.neighboorhood import FaissKNN


class TestLosses(unittest.TestCase):

    example_dataset = np.array([
        [0., 1., 0., 1., 1.],   # K = 2 (D = 2)
        [1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0.],   # INPUTS
        [1., 0., 1., 1., 1.],   # K = 3 (D = 3)
        [1., 0., 0., 0., 0.]    # K = 1 (D = 1)
    ])
    # CLASSES = [0, 1, 1]

    x = example_dataset[:, :-1]
    y = example_dataset[:, -1].reshape(-1, 1)
    knn = FaissKNN(x, y)

    # Simulate a batch of 4 examples (4 different combinations of targets)
    inputs = Tensor(example_dataset[2, :-1]).repeat(4, 1)
    default_prediction = Tensor([1., 1., -1., -1.]).reshape(-1, 1)
    default_target = Tensor([1., -1., 1., -1.]).reshape(-1, 1)

    # Loss / std range / valid_result
    # std range == True -> y e [0, 1] else y e [-1, 1]
    loss_functions = [
        [L.HingeLoss(), False, Tensor([0., 2., 2., 0])],
        [L.SquaredHingeLoss(), False, Tensor([0., 4., 4., 0])],
        [L.BinaryCrossEntropy(), True, Tensor([0., 29.8974, 29.8974, 0.])],
        [L.LogisticLoss(), False, Tensor([0.4519, 1.8946, 1.8946, 0.4519])],
        [L.ExponentialLoss(), False, Tensor([0.6065, 1.6487, 1.6487, 0.6065])]
    ]

    # Collective loss functions
    collective_loss_functions = [
        [L.CollectiveHingeLoss(knn), False, Tensor([0., 1.1666, 1.8333, 0.])],
        [L.CollectiveSquaredHingeLoss(knn), False, Tensor([0., 1.3611, 3.3611, 0.])],
        [L.CollectiveBinaryCrossEntropy(knn), True, Tensor([0.29248, 30.6898, 30.1898, 0.7924])],
        [L.CollectiveLogisticLoss(knn), False, Tensor([0.3375, 1.5596, 2.2611, 0.5978])],
        [L.CollectiveExponentialLoss(knn), False, Tensor([0.5134, 1.3956, 1.9477, 0.7165])]
    ]

    # Helpers
    def tensorsAlmostEqual(self, a: Tensor, b: Tensor, eps: float = 1e-4
                           ) -> bool:
        return (torch.abs((a - b)) < eps).all().item()

    def tensorsEqual(self, a: Tensor, b: Tensor) -> bool:
        return (a == b).all().item()

    # Basic losses
    def test_basic_losses(self):

        for lossfunc, std_range, valid_result in self.loss_functions:

            prediction = self.default_prediction if not std_range \
                else torch.max(Tensor([0.]), self.default_prediction)

            target = self.default_target if not std_range \
                else torch.max(Tensor([0.]), self.default_target)

            loss = lossfunc(prediction, target, reduction='none')
            valid_result = valid_result.reshape(-1, 1)

            self.assertTrue(self.tensorsAlmostEqual(loss, valid_result))

    def test_entropy_weighted_losses(self):

        for lossfunc, std_range, valid_result in self.loss_functions:

            prediction = self.default_prediction if not std_range \
                else torch.max(Tensor([0.]), self.default_prediction)

            target = self.default_target if not std_range \
                else torch.max(Tensor([0.]), self.default_target)

            lossfunc = L.EntropyWeightedBinaryLoss(lossfunc, self.knn)
            loss = lossfunc(prediction, target, self.inputs, reduction='none')
            valid_result = (valid_result.reshape(-1, 1)
                            * torch.exp(-Tensor([0.918296])))

            self.assertTrue(self.tensorsAlmostEqual(loss, valid_result))

    def test_entropy_regularized_losses(self):

        for lossfunc, std_range, valid_result in self.loss_functions:

            prediction = self.default_prediction if not std_range \
                else torch.max(Tensor([0.]), self.default_prediction)

            target = self.default_target if not std_range \
                else torch.max(Tensor([0.]), self.default_target)

            lossfunc = L.EntropyRegularizedBinaryLoss(lossfunc, self.knn)
            model_class_distribution = Tensor([0.5, 0.5]).repeat(4, 1)
            loss = lossfunc(prediction, target, self.inputs,
                            model_class_distribution, reduction='none')
            valid_result = (valid_result.reshape(-1, 1)
                            + Tensor([0.0589]))

            self.assertTrue(self.tensorsAlmostEqual(loss, valid_result))

    def test_collective_losses(self):

        for lossfunc, std_range, valid_result in self.collective_loss_functions:

            prediction = self.default_prediction if not std_range \
                else torch.max(Tensor([0.]), self.default_prediction)

            target = self.default_target if not std_range \
                else torch.max(Tensor([0.]), self.default_target)

            loss = lossfunc(prediction, target, self.inputs, reduction='none')
            valid_result = valid_result.reshape(-1, 1)

            self.assertTrue(self.tensorsAlmostEqual(loss, valid_result))


if __name__ == '__main__':
    unittest.main()
