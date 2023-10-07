import numpy as np
import numpy.typing as npt
from abc import abstractmethod
from collections.abc import Iterable


class Node:
    """Abstract class Node.

    Implements interface for a node in a Neural Network
    """

    DIM = Iterable[int, int] | int

    np_floating = npt.NDArray[np.float64]

    # Custom datatype for dimension -- integer or two intergers inside iterable
    def __init__(self, input_dim: DIM, output_dim: DIM):
        """
        Object initialization.

        Parameters:
            input_dim (int, int): number of inputs
            output_dim (int, int): number of outputs
        """

        self._intput_dim: Node.DIM = input_dim
        self._output_dim: Node.DIM = output_dim

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
        self._input_values: Node.np_floating = np.zeros(self._intput_dim, dtype=np.float64)
        self._output_values: Node.np_floating = np.zeros(self._output_dim, dtype=np.float64)

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
        input: Node.np_floating = input.astype(np.float64)
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

        input_pd: Node.np_floating = input_pd.astype(np.float64)
        jacobian: Node.np_floating = self.jacobian(self._input_values)

        if jacobian.ndim == 1 and Node.np_floating is None:
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


class Optimizer():
    """Abstract class Optimizer.

    Implements interface for a optimizer in a Neural Network"""

    # TODO: organize typing system
    np_floating = Node.np_floating

    @abstractmethod
    def optimize(self, optimized_target: np_floating, gradients: np_floating) -> None:
        """
        Abstract method to apply optimization to target variable. 
        Updates optimized_target using information from gradients

        Parameters:
            optimized_target (np_floating): target to optimize
            gradients (np_floating): partial derivatives of the target. Must be of the same size
                                     as optimized_target
        """
        if optimized_target.shape == gradients.shape:
            raise ValueError(f"Target and gradient must have the same dimension, but {optimized_target.shape} != {gradients.shape}")

    @abstractmethod
    def limit_reached(self) -> bool:
        """
        Abstract method to determine whether optimization limit is reached or not

        Returns:
        limit_reached (bool): True, if optimizer reached its limit, False otherwise"""

        pass


class Neuron(Node):
    """Abstract class Neuron

    Extends functionality of Node by adding update mechanism for inner state"""

    @abstractmethod
    def update_weights(self, optimizer: Optimizer) -> None:
        '''
        Abstract method to update inner states using optimizer. 
        To optimize the weights, the network must be initialized

        Parameters:
        optimizer (Optimizer): instance of optimizer applied to inner variables'''
        
        if not self._initialized:
            raise RuntimeError("Optimization failed: system not initialized")
