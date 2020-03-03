from torch import Tensor
from typing import Callable

def MSE() -> Callable[[Tensor, Tensor], Tensor]:
    def mse(target: Tensor, output: Tensor) -> float:
        return ((target - output) ** 2).mean()
    return mse