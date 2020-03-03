import matplotlib.pyplot as plt
from typing import List

def plot_loss(loss: List[float]) -> None:
    """ Plots loss values """
    
    plt.title("Loss function over time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--")
    plt.plot(loss)