import numpy as np
from typing import Tuple, Callable, List

class SyntheticDataGenerator():

    def __init__(self, gen_function: Callable[[float], float] = None) -> None:
        """ TODO: Doc """
        self.__default_data_gen_function = lambda x: x ** 3 + x ** 2 - 4 * x - 2
        self.data_gen_function = gen_function if gen_function is not None else self.__default_data_gen_function
        
    def generate_data(self, num_points: int, r_min: int = -3, r_max: int =  3, noise_multiplier: float = 3) -> Tuple[List[int], List[int]]:
        """ TODO: Doc """
        x = np.random.choice(np.arange(r_min, r_max, 0.01), num_points)
        y = [self.data_gen_function(value) + (np.random.uniform() - 0.5) * noise_multiplier for value in x]
        return (x, y)

    def get_data_gen_function(self) -> Callable[[float], float]:
        """ Return data generator function """
        return self.data_gen_function

    def set_data_gen_function(self, gen_function: Callable[[float], float]) -> None:
        """ Sets new generator function """
        self.data_gen_function = gen_function