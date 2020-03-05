from torch import nn, Tensor

class ExampleNeuralNetwork(nn.Module):
    """ TODO: Doc """
    
    def __init__(self) -> None:
        super(ExampleNeuralNetwork, self).__init__()
        
        self.fc_network = nn.Sequential(
            nn.Linear(1, 12),
            nn.Sigmoid(),
            nn.Linear(12, 1)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.fc_network(x)