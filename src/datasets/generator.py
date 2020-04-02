import numpy as np

from typing import Tuple, Callable, List

class SyntheticDataGenerator():

    def __init__(self, function: Callable[[float], float] = lambda x: x ** 3 + x ** 2 - 4 * x - 2) -> None:
        """ TODO: Doc """
        self.function = function
        
    def generate(self, num_points: int, r_min: int = -3, r_max: int =  3, noise_multiplier: float = 3) -> Tuple[List[int], List[int]]:
        """ TODO: Doc """
        x = np.random.choice(np.arange(r_min, r_max, 0.01), num_points)
        y = [self.function(value) + (np.random.uniform(0, 2) - 1) * noise_multiplier for value in x]
        return (x, y)