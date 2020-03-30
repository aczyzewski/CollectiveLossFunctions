from typing import List, Any
from torch import nn, Tensor

from .mish import Mish

class CustomNeuralNetwork(nn.Module):
    """ TODO: Doc """
    
    def __init__(self, layers: List[int], hidden_activations: Any = "none", output_activations: Any = "none", initialization: str = "he") -> None:
        super(CustomNeuralNetwork, self).__init__()
    
        self.str_to_activations_converter = {
            "elu": nn.ELU, 
            "hardshrink": nn.Hardshrink, 
            "hardtanh": nn.Hardtanh,
            "leakyrelu": nn.LeakyReLU, 
            "logsigmoid": nn.LogSigmoid, 
            "prelu": nn.PReLU,
            "relu": nn.ReLU, 
            "relu6": nn.ReLU6, 
            "rrelu": nn.RReLU, 
            "selu": nn.SELU,
            "sigmoid": nn.Sigmoid, 
            "softplus": nn.Softplus, 
            "logsoftmax": nn.LogSoftmax,
            "softshrink": nn.Softshrink, 
            "softsign": nn.Softsign, 
            "tanh": nn.Tanh,
            "tanhshrink": nn.Tanhshrink, 
            "softmin": nn.Softmin, 
            "softmax": nn.Softmax,
            "mish": Mish,
            "none": None
        }

        self.str_to_initialiser_converter = {
            "uniform": nn.init.uniform_, 
            "normal": nn.init.normal_,
            "eye": nn.init.eye_,
            "xavier_uniform": nn.init.xavier_uniform_, 
            "xavier": nn.init.xavier_uniform_,
            "xavier_normal": nn.init.xavier_normal_,
            "kaiming_uniform": nn.init.kaiming_uniform_, 
            "kaiming": nn.init.kaiming_uniform_,
            "kaiming_normal": nn.init.kaiming_normal_, 
            "he": nn.init.kaiming_normal_,
            "orthogonal": nn.init.orthogonal_,  
        }

        # Get lists of all elements
        self.__hidden_layers = [
            nn.Linear(input_dim, output_dim) 
            for input_dim, output_dim in zip(layers[:-1], layers[1:])
        ]
        self.__hidden_activations = self.str_to_activations_converter[hidden_activations]
        self.__output_activations = self.str_to_activations_converter[output_activations]

        # Combine them together
        self.list_of_network_blocks = []
        for layer in self.__hidden_layers:
            self.list_of_network_blocks.append(layer)
            if self.__hidden_activations is not None:
                self.list_of_network_blocks.append(self.__hidden_activations())

        # Remove the last activation
        if self.__hidden_activations is not None: 
            self.list_of_network_blocks.pop()

        # Add output activation
        if self.__output_activations is not None:
            self.list_of_network_blocks.append(self.__output_activations())

        # Initalize layers
        init_method = self.str_to_initialiser_converter[initialization]
        for layer in self.__hidden_layers:
            init_method(layer.weight)

        # Create the network
        self.network = nn.Sequential(*self.list_of_network_blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class ExampleNeuralNetwork(CustomNeuralNetwork):
    def __init__(self):
        super().__init__([1, 12, 1], hidden_activations="sigmoid")
