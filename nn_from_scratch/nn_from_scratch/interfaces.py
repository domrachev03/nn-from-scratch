import numpy as np
import numpy.typing as npt
from abc import abstractmethod
from collections.abc import Iterable


class NNTypes:
    # Tensors as input are not supported
    DIM = Iterable[int, int] | int
    np_floating = npt.NDArray[np.float64]
    optional_np_floating = np_floating | None


class Node:
    """Abstract class Node.

    Implements interface for a node in a Neural Network
    """
    DIM = NNTypes.DIM
    np_floating = NNTypes.np_floating
    optional_np_floating = NNTypes.optional_np_floating

    # Custom datatype for dimension -- integer or two intergers inside iterable
    def __init__(self, input_dim: DIM, output_dim: DIM, is_output_node: bool = False):
        """
        Object initialization.

        Parameters:
            input_dim (int, int): number of inputs
            output_dim (int, int): number of outputs
        """

        self._input_dim: Node.DIM = input_dim
        self._output_dim: Node.DIM = output_dim
        self._is_output_node: bool = is_output_node

        jacobian_dimension = []
        if isinstance(self._input_dim, Iterable):
            jacobian_dimension += self._input_dim
        else:
            jacobian_dimension.append(self._input_dim)
        if isinstance(self._output_dim, Iterable):
            jacobian_dimension += self._input_dim[::-1]
        else:
            jacobian_dimension.append(self._output_dim)
        self._jac_dim: Iterable[int] = tuple(jacobian_dimension)

        self.reset()

    @property
    def n_input(self) -> DIM:
        return self._input_dim

    @property
    def n_output(self) -> DIM:
        return self._output_dim

    def reset(self) -> None:
        """
        Setting Node to zero state
        """

        self._initialized: bool = False
        if self._is_output_node:
            self._labels: Node.np_floating = np.zeros(self._output_dim, dtype=np.float64)
        self._input_values: Node.np_floating = np.zeros(self._input_dim, dtype=np.float64)
        self._output_values: Node.np_floating = np.zeros(self._output_dim, dtype=np.float64)

    @abstractmethod
    def f(self, x: np_floating, y: optional_np_floating = None) -> np_floating:
        """
        Abstract method to calculate the node function. Does not update the inner state of the system

        Parameters:
            x (np_floating(n_input)): function input

        Returns:
            y (np_floating(n_output)): function output
        """
        if y is None and self._is_output_node:
            raise ValueError(
                "Value Error: labels are not provided"
            )

    @abstractmethod
    def jacobian(self, x: np_floating, y: optional_np_floating = None) -> np_floating:
        """
        Abstract method for calculating partial derivatives of the function, given input.

        Parameters:
            x (np_floating(n_input)): function input
            y (optional_np_floating(n_input)): true labels, optional. For loss function compatibility

        Returns:
            df_dx (np_floating(n_input, n_output[::-1])): jacobian of the function
        """
        if y is None and self._is_output_node:
            raise ValueError(
                "Value Error: labels are not provided"
            )
        if y is not None and x.shape != y.shape:
            if x.ndim > 1 or x.shape != y.shape:
                raise ValueError(
                    f"Dimension Error: predictions dimension {x.shape} not equal to labels dimension {y.shape}"
                )

    def forward(self, input: np_floating, labels: optional_np_floating = None) -> np_floating:
        """
        Forward propogation method. Initialized inner state and returns output of the node
        Parameters:
            input (np_floating(n_input)): input values
        Returns:
            output (np_floating(n_output)): output of the layer
        """

        if labels is None and self._is_output_node:
            raise ValueError(
                "Value Error: labels are not provided"
            )

        self._input_values = input

        if self._is_output_node:
            self._labels = labels
            self._output_values = self.f(input, labels)
        else:
            self._output_values = self.f(input)
        self._initialized = True

        return self._output_values

    def backward(self, input_pd: optional_np_floating = None, reset_after: bool = True) -> np_floating:
        """
        Method for backpropogation
        Parameters:
            input_pd (optional_np_floating(n_output)): partial derivatives of the next layer, optional.
            reset_after (bool): if True, resets inner state after backpropogation
        Returns:
            output_pd (np_floating(n_input)): partial derivatives to propogate back"""

        if input_pd is None and not self._is_output_node:
            raise ValueError(
                "Value Error: partial derivatives are not provided"
            )

        if not self._initialized:
            raise RuntimeError("Backpropogation failed: system not initialized")

        jacobian: Node.np_floating = self.jacobian(self._input_values)

        if self._is_output_node:
            # This is assumed to be the last node, returning the value of jacobian itself
            backprop_pd = jacobian
        else:
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

    np_floating = NNTypes.np_floating

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
    def reset(self) -> None:
        """
        Abstract method for resetting optimizer state after one epoch
        """
        pass

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
    # TODO: class Neuron seems too abstract. Think about the way to generalize weights

    DIM = Node.DIM
    np_floating = Node.np_floating

    @abstractmethod
    def optimize_weights(self, optimizer: Optimizer) -> None:
        '''
        Abstract method to update inner states using optimizer.
        To optimize the weights, the network must be initialized

        Parameters:
        optimizer (Optimizer): instance of optimizer applied to inner variables'''

        if not self._initialized:
            raise RuntimeError("Optimization failed: system not initialized")
