from torch import Tensor
from typing import Callable


def MSE() -> Callable[[Tensor, Tensor], Tensor]:
    def mse(input: Tensor, target: Tensor) -> float:
        return ((target - input) ** 2).mean()
    return mse
