import numpy as np
from abc import abstractmethod


class Node:
    """Abstract class Node.

    Implements interface for a node in a Neural Network
    """

    def __init__(self, n_input: int, n_output: int):
        """
        Object initialization.

        Parameters:
            n_input (int): number of inputs
            n_output (int): number of outputs
        """

        self._n_input: int = n_input
        self._n_output: int = n_output

        self._initialized: bool = False
        self._input_values: np.ndarray = np.zeros(self._n_input, dtype=np.float64)
        self._output: np.ndarray = np.zeros(self._n_output, dtype=np.float64)
        self._jacobian: np.ndarray = np.zeros((self._n_input, self._n_output), dtype=np.float64)

    @property
    def n_input(self) -> int:
        return self._n_input

    @property
    def n_output(self) -> int:
        return self._n_output

    def reset(self) -> None:
        """
        Clearing inner state of the node
        """

        self._initialized = False
        self._input_values = np.zeros(self._n_input, dtype=np.float64)
        self._output = np.zeros(self._n_output, dtype=np.float64)
        self._jacobian = np.zeros((self._n_input, self._n_output), dtype=np.float64)

    @abstractmethod
    def f(self, x: np.ndarray) -> np.ndarray:
        """
        Abstract method to calculate the node function. Does not update the inner state of the system

        Parameters:
            x (np.ndarray(n_input)): function input

        Returns:
            y (np.ndarray(n_output)): function output
        """
        pass

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Abstract method for calculating partial derivatives of the function, given input.

        Parameters:
            x (np.ndarray(n_input)): function input

        Returns:
            df_dx (np.ndarray(n_input, n_output)): 2D jacobian of the function
        """
        pass

    def forward(self, input: np.ndarray) -> np.ndarray:
        """ 
        Forward propogation method. Initialized inner state and returns output of the node
        Parameters:
            input (np.ndarray(n_input)): input values
        Returns:
            output (np.ndarray(n_output)): output of the layer
        """
        input = input.astype(np.float64)
        self._input_values = input

        self._output = self.f(input)
        self._jacobian = self.jacobian(input)
        self._initialized = True

        return self._output

    def backward(self, input_pd: np.ndarray, reset_after: bool = True) -> np.ndarray:
        """
        Abstract method for backpropogation
        Parameters:
            input_pd (np.ndarray(n_output)): partial derivatives of the next layer
            reset_after (bool): it True, resets inner state after backpropogation
        Returns:
            output_pd (np.ndarray(n_input)): partial derivatives to propogate back"""

        if not self._initialized:
            raise RuntimeError("Backpropogation failed: system not initialized")

        input_pd = input_pd.astype(np.float64)
        backprop_pd = self._jacobian @ input_pd

        if reset_after:
            self.reset()

        return backprop_pd


class SoftMax(Node):
    """ Implementation of softmax node.

    .. math::
        \hat{y}_{i} = \\frac{\exp^{x_i}}{\sum_{j=1}^{N} \exp^{x_j}}
    """

    def __init__(self, n_input: int):
        super().__init__(n_input, n_input)

    def f(self, x: np.ndarray) -> float:
        exp_input = np.exp(x)
        y = exp_input / exp_input.sum()
        return y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
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
                for i in range(self._n_input)
            ] for j in range(self._n_input)
        ])

        return jacobian_diag + jacobian_triag + jacobian_triag.T


class NormalizedSoftMax(SoftMax):
    """ Implementation of softmax node.
    It inherits the SoftMax node, since its functionality is 
    heavily used

    .. math::
        \hat{y}_{i} = \\frac{\exp^{x_i/x_{max}}}{\sum_{j=1}^{N} \exp^{x_j/x_{max}}}
    """

    def __init__(self, n_input: int):
        super().__init__(n_input)
        self._max_idx = -1

    def reset(self) -> None:
        super().reset()
        self._max_idx = -1

    def f(self, x: np.ndarray) -> float:
        self._max_idx = np.argmax(x)
        x_norm = x / x[self._max_idx]

        y = super().f(x_norm)
        return y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        x_max = x[self._max_idx]

        softmax_jacobian = super().jacobian(x / x_max)

        dx_norm_dx = np.diag([1/x_max for _ in range(self._n_input)])
        dx_norm_dx[:, self._max_idx] = np.array([
            -x_i / x_max**2 for x_i in x
        ])
        dx_norm_dx[self._max_idx, :] = np.zeros(self._n_input)

        return softmax_jacobian @ dx_norm_dx
