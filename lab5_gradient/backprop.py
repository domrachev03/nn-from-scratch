import numpy as np
import numpy.typing as npt
from abc import abstractmethod
from collections.abc import Iterable


DIM = Iterable[int, int] | int
np_floating = npt.NDArray[np.float64]


class Node:
    """Abstract class Node.

    Implements interface for a node in a Neural Network
    """

    # Custom datatype for dimension -- integer or two intergers inside iterable
    def __init__(self, input_dim: DIM, output_dim: DIM):
        """
        Object initialization.

        Parameters:
            input_dim (int, int): number of inputs
            output_dim (int, int): number of outputs
        """

        self._intput_dim: DIM = input_dim
        self._output_dim: DIM = output_dim

        jacobian_dimension = []
        if isinstance(self._intput_dim, Iterable):
            jacobian_dimension += self._intput_dim
        else:
            jacobian_dimension.append(self._intput_dim)
        if isinstance(self._output_dim, Iterable):
            jacobian_dimension += self._intput_dim[::-1]
        else:
            jacobian_dimension.append(self._output_dim)
        self._jac_dim: Iterable[int] = tuple(jacobian_dimension)

        self.reset()

    @property
    def n_input(self) -> DIM:
        return self._intput_dim

    @property
    def n_output(self) -> DIM:
        return self._output_dim

    def reset(self) -> None:
        """
        Setting Node to zero state
        """

        self._initialized: bool = False
        self._input_values: np_floating = np.zeros(self._intput_dim, dtype=np.float64)
        self._output_values: np_floating = np.zeros(self._output_dim, dtype=np.float64)

    @abstractmethod
    def f(self, x: np_floating) -> np_floating:
        """
        Abstract method to calculate the node function. Does not update the inner state of the system

        Parameters:
            x (np_floating(n_input)): function input

        Returns:
            y (np_floating(n_output)): function output
        """
        pass

    @abstractmethod
    def jacobian(self, x: np_floating) -> np_floating:
        """
        Abstract method for calculating partial derivatives of the function, given input.

        Parameters:
            x (np_floating(n_input)): function input

        Returns:
            df_dx (np_floating(n_input, n_output[::-1])): jacobian of the function
        """
        pass

    def forward(self, input: np_floating) -> np_floating:
        """
        Forward propogation method. Initialized inner state and returns output of the node
        Parameters:
            input (np_floating(n_input)): input values
        Returns:
            output (np_floating(n_output)): output of the layer
        """
        input = input.astype(np.float64)
        self._input_values = input

        self._output_values = self.f(input)
        self._initialized = True

        return self._output_values

    def backward(self, input_pd: np_floating = None, reset_after: bool = True) -> np_floating:
        """
        Method for backpropogation
        Parameters:
            input_pd (np_floating(n_output)): partial derivatives of the next layer
            reset_after (bool): if True, resets inner state after backpropogation
        Returns:
            output_pd (np_floating(n_input)): partial derivatives to propogate back"""

        if not self._initialized:
            raise RuntimeError("Backpropogation failed: system not initialized")

        input_pd = input_pd.astype(np.float64)
        jacobian = self.jacobian(self._input_values)

        if jacobian.ndim == 1 and np_floating is None:
            # This is assumed to be the last node, returning the value of jacobian itself
            backprop_pd = jacobian
        if jacobian.ndim == 2:
            # In case of matrix jacobian, multiplication is applied
            backprop_pd = jacobian @ input_pd
        else:
            # In case of tensor product, parial derivatives equal to elementwise product with
            # summation over all axes except for the first two
            backprop_pd = (jacobian * input_pd).sum(tuple(i for i in range(2, jacobian.ndim)))

        if reset_after:
            self.reset()

        return backprop_pd


class SoftMax(Node):
    """ Implementation of softmax node.

    .. math::
        \\hat{y}_{i} = \\frac{\\exp^{x_i}}{\\sum_{j=1}^{N} \\exp^{x_j}}
    """

    def __init__(self, n_input: int):
        super().__init__(n_input, n_input)

    def f(self, x: np_floating) -> np_floating:
        exp_input = np.exp(x)
        y = exp_input / exp_input.sum()
        return y

    def jacobian(self, x: np_floating) -> np_floating:
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
                for i in range(self._intput_dim)
            ] for j in range(self._intput_dim)
        ])

        return jacobian_diag + jacobian_triag + jacobian_triag.T


class NormalizedSoftMax(SoftMax):
    """ Implementation of softmax node.
    It inherits the SoftMax node, since its functionality is
    heavily used

    .. math::
        \\hat{y}_{i} = \\frac{\\exp^{x_i/x_{max}}}{\\sum_{j=1}^{N} \\exp^{x_j/x_{max}}}
    """

    def __init__(self, n_input: int):
        super().__init__(n_input)
        self._max_idx = -1

    def reset(self) -> None:
        super().reset()
        self._max_idx = -1

    def f(self, x: np_floating) -> np_floating:
        self._max_idx = np.argmax(x)
        x_norm = x / x[self._max_idx]

        y = super().f(x_norm)
        return y

    def jacobian(self, x: np_floating) -> np_floating:
        x_max = x[self._max_idx]

        # Jacobian could be decomposed as multiplication of two jacobians:
        # 1. d(softmax)/dx (as function of x_norm)
        softmax_jacobian = super().jacobian(x / x_max)

        # 2. d(x_norm)/dx
        dx_norm_dx = np.diag([1/x_max for _ in range(self._intput_dim)])
        dx_norm_dx[:, self._max_idx] = np.array([
            -x_i / x_max**2 for x_i in x
        ])
        dx_norm_dx[self._max_idx, :] = np.zeros(self._intput_dim)

        return softmax_jacobian @ dx_norm_dx


class ReLU(Node):
    """ Implementation of ReLU function. Note, that d(ReLU)/dx (0) = 0 was chosen.
        .. math::
            ReLU(x) = max(0, x)
    """

    def __init__(self, n_input: int):
        super().__init__(n_input, n_input)

    def f(self, x: np_floating) -> np_floating:
        return np.vectorize(lambda x_i: max(0, x_i))(x)

    def jacobian(self, x: np_floating) -> np_floating:
        return np.diag([1 if x_i > 0 else 0 for x_i in x])


class Linear(Node):
    """ Implements fully-connected linear layer

        .. math::
            FC(X) = WX
    """

    def __init__(self, input_dim: Iterable[int, int], output_dim: Iterable[int, int], W: np_floating = None):
        if input_dim[1] != output_dim[1]:
            raise Exception(
                f"Dimension Error: second input dimension {input_dim[1]} not equal to second output dimension {output_dim[1]}"
            )

        self._W_dim: Iterable[int, int] = (output_dim[0], input_dim[0])
        super().__init__(input_dim, output_dim)

        self.W_init(W)

    def W_init(self, W: np_floating = None):
        '''
        Set weights W with given value or initialize it randomly.
        Also resets partial derivatives of W.
        '''

        self._initialized = False
        self._W: np_floating = np.random.uniform(
            0.4, 0.6,
            self._W_dim
        ) if W is None else W
        self.reset()

    def reset(self):
        super().reset()
        self._W_pd: np_floating = np.zeros(self._W_dim, dtype=np.float64)

    @property
    def W(self) -> np_floating:
        return self._W

    def f(self, X: np_floating) -> np_floating:
        return self._W @ X

    def jacobian(self, X: np_floating) -> np_floating:
        """
        The jacobian w.r.t. X.
        Note that backward function is overriden by this class.
        Parameters:
            X (np_floating(input_dim)): input values of the function
        Returns:
            d(FC)/dX (np_floating(input_dim, output_dim)): partial derivatives of function w.r.t. X
        """

        X_jac_dim: Iterable[int] = (*self._intput_dim, *self._output_dim)
        X_jac: np_floating = np.zeros(X_jac_dim, dtype=np.float64)

        for i in range(X_jac_dim[0]):
            for j in range(X_jac_dim[1]):
                X_jac[i, j, :, j] = self._W[:, i]
        return X_jac

    def W_jacobian(self, X: np_floating) -> np_floating:
        """
        The jacobian w.r.t. W.
        Note that partial derivatives of W are computer in overriden backwards function.
        Parameters:
            X (np_floating(input_dim)): input values of the function
        Returns:
            d(FC)/dW (np_floating(W_dim, output_dim)): partial derivatives of function w.r.t. W
        """

        W_jac_dim: Iterable[int] = (*self._W_dim, *self._output_dim)
        W_jac: np_floating = np.zeros(W_jac_dim, dtype=np.float64)
        for i in range(W_jac_dim[0]):
            for j in range(W_jac_dim[1]):
                W_jac[i, j, i] = X[j]
        return W_jac

    def backward(self, input_pd: np_floating, reset_after: bool = False, use_jacobian: bool = False) -> np_floating:
        """
        Overriden method for backpropogation. It does not use jacobians, but computes
        partial derivatives via dot products. Works faster than general approach.
        Parameters:
            input_pd (np_floating(n_output)): partial derivatives of the next layer
            reset_after (bool): if True, resets inner state after backpropogation
            use_jacobian (bool): if True, uses the superclass implemented backward function, and
                                 updates the state of W in similar way
        Returns:
            output_pd (np_floating(n_input)): partial derivatives to propogate back"""

        if not self._initialized:
            raise RuntimeError("Backpropogation failed: system not initialized")

        # Optional calculations via jacobian
        if use_jacobian:
            self._w_pd = (self.W_jacobian(self._input_values) * input_pd).sum((2,3))
            return super().backward(input_pd, reset_after)

        # Optimized calculations via dot products
        input_pd = input_pd.astype(np.float64)

        self._W_pd = input_pd.dot(self._input_values.T)
        backprop_pd = self._W.T.dot(input_pd)

        if reset_after:
            self.reset()

        return backprop_pd
