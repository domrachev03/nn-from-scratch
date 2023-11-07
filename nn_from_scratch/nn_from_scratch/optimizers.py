from .interfaces import Optimizer
from collections.abc import Iterable
import numpy as np


class GradientDescent(Optimizer):
    r'''Gradient Descent class.

    .. math::
        \\W_{i} = W_{i-1} - \eta \frac{\partial L}{\partial W}'''

    def __init__(self, lr: float = 1e-3, eps: float = 1e-3) -> None:
        '''Initialization of gradient descent

        Parameters:
            lr (float) -- learning rate of the optimizer
            eps (float) -- minimal delta per '''

        if lr < 0:
            raise ValueError("Negative learning rate is not allowed")

        self._lr: float = lr
        self._eps: float = eps
        self._last_change: float = 0

    @property
    def lr(self) -> float:
        return self._lr

    @property
    def delta(self) -> float:
        return self._last_change

    def optimize(
            self,
            optimizer_target: Iterable[Optimizer.np_floating],
            gradients: Iterable[Optimizer.np_floating]
    ) -> Iterable[Optimizer.np_floating]:
        super().optimize(optimizer_target, gradients)
        optimized_targets = []
        for target, grad in zip(optimizer_target, gradients):
            optimized_targets.append(target - self._lr * grad)
            self._last_change += self._lr * np.linalg.norm(grad)

        return optimized_targets

    def reset(self) -> None:
        self._last_change = None

    def limit_reached(self) -> bool:
        if self._last_change < self._eps:
            return False
        return True
