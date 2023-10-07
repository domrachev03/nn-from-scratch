from .interfaces import Optimizer
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
        self._last_change: float = None

    @property
    def lr(self) -> float:
        return self._lr

    @property
    def delta(self) -> float:
        return self._last_change

    def optimize(self, optimizer_target: Optimizer.np_floating, gradients: Optimizer.np_floating) -> None:
        super().optimize(optimizer_target, gradients)
        optimizer_target = optimizer_target - self._lr * gradients

        if self._last_change is None:
            self._last_change = self._lr * np.linalg.norm(gradients)
        else:
            self._last_change += self._lr * np.linalg.norm(gradients)

    def reset(self) -> None:
        self._last_change = None

    def limit_reached(self) -> bool:
        if self._last_change is None or self._last_change < self._eps:
            return False
        return True
