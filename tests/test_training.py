import unittest
import math

from torch import nn, Tensor

import src.training as training
from src.networks import CustomNeuralNetwork


class TrainingTests(unittest.TestCase):

    def test_binary_tanh_evaluation(self):

        model = CustomNeuralNetwork([1, 1], None, 'tanh',
                                    store_output_layer_idx=-1)

        # Set the weights of the linear layer
        model.network[-2].weight = nn.Parameter(Tensor([[1.]]))
        model.network[-2].bias = nn.Parameter(Tensor([0.]))

        # Define inputs
        inputs = Tensor([[math.atanh(item)] for item in [-0.65, -0.30, 0.30, 0.65]])
        targets = Tensor([[-1], [-1], [1], [1]])
        results = training.evaluate_binary(model, inputs, targets)

        self.assertEqual(results['precision'], 1.0)
        self.assertEqual(results['recall'], 1.0)

    def test_binary_sigmoid_evaluation(self):

        model = CustomNeuralNetwork([1, 1], None, 'sigmoid',
                                    store_output_layer_idx=-1)

        # Set the weights of the linear layer
        model.network[-2].weight = nn.Parameter(Tensor([[1.]]))
        model.network[-2].bias = nn.Parameter(Tensor([0.]))

        # Define inputs
        inputs = Tensor([[-3], [-1], [1], [3]])
        targets = Tensor([[0], [0], [1], [1]])
        results = training.evaluate_binary(model, inputs, targets)

        self.assertEqual(results['precision'], 1.0)
        self.assertEqual(results['recall'], 1.0)
