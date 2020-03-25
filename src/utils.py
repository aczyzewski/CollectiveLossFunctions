import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_values(values: Dict[str, List[float]], xtitle: str = 'Epoch', ytitle = 'Loss', size: [int, int] = (12, 6)):
    """ Plots multiple lines on the same plot """

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(size))
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.grid(True, linestyle="--")
    for title, items in values.items():
        plt.plot(items, label=title)

    plt.legend()
    plt.show()
    
