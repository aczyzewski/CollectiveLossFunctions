from typing import List, Any
from torch import nn, Tensor
import src.utils as utils


class CustomNeuralNetwork(nn.Module):
    """ TODO: Doc """

    def __init__(self, layers: List[int], hidden_activations: Any = None,
                 output_activations: Any = None, initialization: str = "he",
                 store_output_layer_idx: int = -4) -> None:

        """ Note: `store_output_layer_idx` allows you to set the
            idx of a layer that output will be stored inside the model
            instance. Keep in mind that activation functions are also
            considered as layers. For example, the default value of
            the parameter (-4) points here:
                 
                *
            [Linear] -> [Hidden Acitivation] -> [Linear] -> [Output activation]
        """
        super(CustomNeuralNetwork, self).__init__()

        self.hidden_activations = hidden_activations
        self.output_activations = output_activations
        self.initialization = initialization

        # Determine hidden activations
        if isinstance(self.hidden_activations, str):
            self.hidden_activations = utils.get_activation_by_name(
                self.hidden_activations)

        # Determine output activations
        if isinstance(self.output_activations, str):
            self.output_activations = utils.get_activation_by_name(
                self.output_activations)

        # Determie initialization
        if isinstance(self.initialization, str):
            self.initialization = utils.get_initialization_by_name(
                self.initialization
            )

        # Store the output of selected layer
        self.stored_output = None

        # Stack linear layers
        self._linear_layers = [
            nn.Linear(input_dim, output_dim)
            for (input_dim, output_dim) in zip(layers[: -1], layers[1:])
        ]

        # Initialize layers
        for layer in self._linear_layers:
            self.initialization(layer.weight)

        # Append an activation to each layer of a network
        self.network = []
        if self.hidden_activations is not None:
            for layer in self._linear_layers:
                self.network.extend([layer, self.hidden_activations()])
            self.network.pop()
        else:
            self.network = self._linear_layers

        # Add an output activation
        if self.output_activations is not None:
            self.network.append(self.output_activations())

        # Convert to Sequential
        self.network = nn.Sequential(*self.network)
        self.network[store_output_layer_idx].register_forward_hook(
            self.get_layer_output_hook)

    def get_layer_output_hook(self, module: nn.Module, _input: Tensor,
                              output: Tensor) -> None:
        """ Retrieves output of a layer of the network """
        self.stored_output = output

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class ExampleNeuralNetwork(CustomNeuralNetwork):
    def __init__(self):
        super().__init__([1, 12, 1], hidden_activations="sigmoid")
