from .interfaces import Node
import numpy as np


class SoftMax(Node):
    r""" Implementation of softmax node.

    .. math::
        \hat{y}_{i} = \frac{\exp^{x_i}}{\sum_{j=1}^{N} \exp^{x_j}}
    """

    def __init__(self, n_input: Node.DIM):
        if not isinstance(n_input, int) and len(n_input) > 2:
            raise ValueError(
                "SoftMax does not accept tensor inputs"
            )
        super().__init__(n_input, n_input, inner_ndim=2)

    def __call__(self, x: Node.np_floating) -> Node.np_floating:
        exp_input = np.exp(x)
        y = exp_input / exp_input.sum(axis=1).reshape(-1, 1)
        return y

    def jacobian(self, x: Node.np_floating) -> Node.np_floating:
        n_rows, n_elems = x.shape
        exp_x = np.exp(x)                               # (n_input[0] | 1, n_input[-1])
        exp_sum = exp_x.sum(axis=1)                     # (n_input[0] | 1)

        jacobians: Node.np_floating = np.zeros((n_rows, n_elems, n_elems))

        for x_row_idx in range(n_rows):
            # The jacobian of softmax is symmetric. Hence, it could be
            # constructed via diagonal component and triangular part separately
            exp_x_row = exp_x[x_row_idx]
            exp_sum_row = exp_sum[x_row_idx]

            jacobian_diag = np.diag([
                    exp_xi / exp_sum_row - (exp_xi / exp_sum_row) ** 2 for exp_xi in exp_x_row
                ]
            )

            jacobian_triag = np.array([
                [
                    -exp_x_row[i] * exp_x_row[j] / exp_sum_row**2 if i > j else 0
                    for i in range(n_elems)
                ] for j in range(n_elems)
            ])

            jacobians[x_row_idx] = jacobian_diag + jacobian_triag + jacobian_triag.T

        return jacobians

    def backward(self, input_pd: Node.np_floating, reset_after: bool = True) -> Node.np_floating:
        if not self._initialized:
            raise RuntimeError(
                "Backpropogation failed: system not initialized"
            )
        if input_pd.shape != self._input_dim:
            raise ValueError(
                f"invalid p.d. shape: expected {self._input_dim} got {input_pd.shape}"
            )
        input_pd = self._change_dims(input_pd, self._inner_ndim)

        jacobian: Node.np_floating = self.jacobian(self._input_values)

        backprop_pd: Node.np_floating = np.array([
            jacobian[i] @ input_pd[i] for i in range(jacobian.shape[0])
        ])

        return self._change_dims(backprop_pd, self._output_ndim)


class NormalizedSoftMax(SoftMax):
    r""" Implementation of softmax node.
    It inherits the SoftMax node, since its functionality is
    heavily used

    .. math::
        \hat{y}_{i} = \frac{\exp^{x_i/x_{max}}}{\sum_{j=1}^{N} \exp^{x_j/x_{max}}}
    """

    def __init__(self, n_input: SoftMax.DIM):
        super().__init__(n_input)

    def reset(self) -> None:
        self._max_idx: SoftMax.np_floating = None
        self._x_norm: SoftMax.np_floating = None

    def __call__(self, x: Node.np_floating) -> Node.np_floating:
        x_abs = np.abs(x)
        self._max_idx = np.argmax(x_abs, axis=1).reshape(-1, 1)
        self._x_norm = x / np.max(x_abs, axis=1).reshape(-1, 1)

        y = super().__call__(self._x_norm)
        return y

    def jacobian(self, x: Node.np_floating) -> Node.np_floating:
        # Jacobian could be decomposed as multiplication of two jacobians:
        # 1. d(softmax)/dx (as function of x_norm)
        softmax_jacobian = super().jacobian(self._x_norm)
        for row_idx in range(x.shape[0]):
            # 2. d(x_norm)/dx
            row_max_idx = self._max_idx[row_idx][0]
            x_max = np.abs(x)[row_idx, row_max_idx]
            dx_norm_dx = np.diag([1/x_max for _ in range(x.shape[1])])
            dx_norm_dx[:, row_max_idx] = np.array([
                -x_i / x_max**2 for x_i in x[row_idx]
            ])
            dx_norm_dx[row_max_idx] = np.zeros(x.shape[1])

            softmax_jacobian[row_idx] = softmax_jacobian[row_idx] @ dx_norm_dx

        return softmax_jacobian


class ReLU(Node):
    """ Implementation of ReLU function. Note, that d(ReLU)/dx (0) = 0 was chosen.
        .. math::
            ReLU(x) = max(0, x)
    """

    def __init__(self, n_input: int):
        super().__init__(n_input, n_input, 4)

    def __call__(self, x: Node.np_floating) -> Node.np_floating:
        super().__call__(x)
        return np.maximum(x, 0)

    def jacobian(self, x: Node.np_floating) -> Node.np_floating:
        if x.ndim > 2:
            raise NotImplementedError("Jacobian for tensors is not available")

        return np.array([
            np.diag([1 if x_elem > 0 else 0 for x_elem in x_row]) for x_row in x
        ])

    def backward(self, input_pd: Node.np_floating, reset_after: bool = True) -> Node.np_floating:
        if not self._initialized:
            raise RuntimeError(
                "Backpropogation failed: system not initialized"
            )
        if input_pd.shape != self._input_dim:
            raise ValueError(
                f"invalid p.d. shape: expected {self._input_dim} got {input_pd.shape}"
            )

        output_pd = np.where(self._input_values > 0, input_pd, 0)

        if reset_after:
            self.reset()

        return output_pd


class SoftMaxLoss(Node):
    r""" Implementation of SoftMax-Loss layer.

    .. math::
        \\hat{y}_{i} =-\log \left (y\frac{\exp^{x_i/x_{max}}}{\sum_{j=1}^{N} \exp^{x_j/x_{max}}} \right)
    """

    def __init__(self, n_input: Node.DIM):
        if not isinstance(n_input, int) and len(n_input) > 2:
            raise ValueError(
                "loss function supports only vector inputs"
            )
        self._norm_softmax = NormalizedSoftMax(n_input)
        super().__init__(n_input, 1, 2, True)

    def __call__(self, x: Node.np_floating, y: Node.np_floating) -> Node.np_floating:
        self._labels = y
        self._x_softmax = self._norm_softmax(x)
        self._loss_elementwise = np.array([
            -np.log(self._labels[i].dot(self._x_softmax[i]))
            for i in range(x.shape[0])
        ])
        return np.sum(self._loss_elementwise) / x.shape[0]

    def jacobian(self, x: Node.np_floating) -> Node.np_floating:
        softmax_jac: Node.np_floating = self._norm_softmax.jacobian(x)

        jac: Node.np_floating = np.zeros_like(x)
        for i in range(x.shape[0]):
            denom = self._loss_elementwise[i]
            jac[i] = - self._labels[i] * softmax_jac[i].diagonal() / denom

        return jac


class Vectorization(Node):
    """ Implementation of Vectorization layer.

    It transforms tensor or matrix inputs into vectors."""

    def __init__(self, input_dim: Node.DIM):
        self._vector_length = 1
        for dim_i in input_dim[1:]:
            self._vector_length *= dim_i
            
        output_dim = (input_dim[0], self._vector_length)
        super().__init__(input_dim, output_dim, 2)

    def __call__(self, x: Node.np_floating) -> Node.np_floating:
        return x.reshape(x.shape[0], -1)

    def backward(self, input_pd: Node.np_floating) -> Node.np_floating:
        return input_pd.reshape(self._input_dim)
