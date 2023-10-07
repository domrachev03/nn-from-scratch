from .interfaces import Node
import numpy as np


class SoftMax(Node):
    r""" Implementation of softmax node.

    .. math::
        \hat{y}_{i} = \frac{\exp^{x_i}}{\sum_{j=1}^{N} \exp^{x_j}}
    """

    def __init__(self, n_input: int):
        super().__init__(n_input, n_input)

    def f(self, x: Node.np_floating) -> Node.np_floating:
        exp_input = np.exp(x)
        y = exp_input / exp_input.sum()
        return y

    def jacobian(self, x: Node.np_floating) -> Node.np_floating:
        exp_x = np.exp(x)
        exp_sum = exp_x.sum()

        # The jacobian of softmax is symmetric. Hence, it could be
        # constructed via diagonal component and triangular part separately
        jacobian_diag = np.diag([
            exp_xi / exp_sum - (exp_xi / exp_sum) ** 2 for exp_xi in exp_x
        ])

        jacobian_triag = np.array([
            [
                -exp_x[i] * exp_x[j] / exp_sum**2 if i > j else 0
                for i in range(self._input_dim)
            ] for j in range(self._input_dim)
        ])

        return jacobian_diag + jacobian_triag + jacobian_triag.T


class NormalizedSoftMax(SoftMax):
    r""" Implementation of softmax node.
    It inherits the SoftMax node, since its functionality is
    heavily used

    .. math::
        \hat{y}_{i} = \frac{\exp^{x_i/x_{max}}}{\sum_{j=1}^{N} \exp^{x_j/x_{max}}}
    """

    def __init__(self, n_input: int):
        super().__init__(n_input)
        self._max_idx = -1

    def reset(self) -> None:
        self._max_idx = -1

    def f(self, x: Node.np_floating) -> Node.np_floating:
        self._max_idx = np.argmax(x)
        x_norm = x / x[self._max_idx]

        y = super().f(x_norm)
        return y

    def jacobian(self, x: Node.np_floating) -> Node.np_floating:
        x_max = x[self._max_idx]

        # Jacobian could be decomposed as multiplication of two jacobians:
        # 1. d(softmax)/dx (as function of x_norm)
        softmax_jacobian = super().jacobian(x / x_max)

        # 2. d(x_norm)/dx
        dx_norm_dx = np.diag([1/x_max for _ in range(self._input_dim)])
        dx_norm_dx[:, self._max_idx] = np.array([
            -x_i / x_max**2 for x_i in x
        ])
        dx_norm_dx[self._max_idx, :] = np.zeros(self._input_dim)

        return softmax_jacobian @ dx_norm_dx


class ReLU(Node):
    """ Implementation of ReLU function. Note, that d(ReLU)/dx (0) = 0 was chosen.
        .. math::
            ReLU(x) = max(0, x)
    """

    def __init__(self, n_input: int):
        super().__init__(n_input, n_input)

    def f(self, x: Node.np_floating) -> Node.np_floating:
        return np.vectorize(lambda x_i: max(0, x_i))(x)

    def jacobian(self, x: Node.np_floating) -> Node.np_floating:
        return np.diag([1 if x_i > 0 else 0 for x_i in x])


class SoftMaxLoss(Node):
    r""" Implementation of SoftMax-Loss layer.

    .. math::
        \\hat{y}_{i} =-\log \left (y\frac{\exp^{x_i/x_{max}}}{\sum_{j=1}^{N} \exp^{x_j/x_{max}}} \right)
    """

    def __init__(self, n_input: Node.DIM):
        self._softmax: Node = SoftMax(n_input)
        super().__init__(n_input, 1, True)

    def f(self, x: Node.np_floating, y: Node.np_floating) -> Node.np_floating:
        x_softmax: Node.np_floating = self._softmax.f(x)
        return -np.log(np.dot(x_softmax, y))

    def jacobian(self, x: Node.np_floating) -> Node.np_floating:
        softmax_jac: Node.np_floating = self._softmax.jacobian(x)
        denominator = np.dot(self._labels, self._softmax.f(x))
        jac: Node.np_floating = np.zeros((1, self._input_dim))
        for i in range(self._input_dim):
            jac[0][i] = -self._labels[i] * softmax_jac[i, i] / denominator

        return jac
