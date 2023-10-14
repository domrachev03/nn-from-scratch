import numpy as np
import numpy.typing as npt
from abc import abstractmethod
from collections.abc import Iterable


class NNTypes:
    M_DIM = Iterable[int, int]
    T_DIM = Iterable[int, int, int]
    DIM = int | M_DIM | T_DIM
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
    def __init__(self, input_dim: DIM, output_dim: DIM, inner_ndim: int, is_output_node: bool = False):
        """
        Object initialization.

        Parameters:
            input_dim (int, int): number of inputs
            output_dim (int, int): number of outputs
        """

        self._input_dim: Node.DIM = (input_dim, ) if isinstance(input_dim, int) else tuple(input_dim)
        self._input_ndim: int = 1 if isinstance(input_dim, int) else len(self._input_dim)
        self._output_dim: Node.DIM = (output_dim, ) if isinstance(output_dim, int) else tuple(output_dim)
        self._output_ndim: int = 1 if isinstance(input_dim, int) else len(self._output_dim)
        self._inner_ndim: int = inner_ndim

        self._is_output_node: bool = is_output_node

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
        self._input_values: Node.np_floating = None
        self._output_values: Node.np_floating = None
        self._labels: Node.np_floating = None

    @abstractmethod
    def __call__(self, x: np_floating, y: optional_np_floating = None) -> np_floating:
        """
        Abstract, to call before implementation.
        Calculates the node function. 
        Parameters:
            x (np_floating(n_input)): function input

        Returns:
            y (np_floating(n_output)): function output
        """
        if y is None and self._is_output_node:
            raise ValueError(
                "labels are not provided"
            )

    @abstractmethod
    def jacobian(self, x: np_floating, y: optional_np_floating = None) -> np_floating:
        """
        Abstract, to call before implementation.
        Calculates partial derivatives of the function, given input.

        Parameters:
            x (np_floating(n_input)): function input
            y (optional_np_floating(n_input)): true labels, optional. For loss function compatibility

        Returns:
            df_dx (np_floating(n_input, n_output[::-1])): jacobian of the function
        """
        if y is None and self._is_output_node:
            raise ValueError(
                "labels are not provided"
            )

    def _change_dims(self, x: np_floating, ndim: int) -> np_floating:
        if x.ndim > ndim:
            try:
                return x.reshape(x.shape[-ndim:])
            except Exception:
                raise ValueError(
                    "downscale dimension change is available only for empty leading axes."
                )
        elif x.ndim == ndim:
            return x
        else:
            return np.expand_dims(x, tuple(i for i in range(ndim - x.ndim)))

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
                "labels are not provided"
            )

        self._input_values = self._change_dims(input, self._inner_ndim)
        if self._is_output_node:
            self._labels = self._change_dims(labels, self._inner_ndim)
            self._output_values = self(self._input_values, self._labels)
        else:
            self._output_values = self(self._input_values)

        self._initialized = True
        return self._change_dims(self._output_values, self._output_ndim)

    def backward(self, input_pd: optional_np_floating = None, reset_after: bool = True) -> np_floating:
        """
        Method for backpropogation.
        Could be overriden for simplifying and/or speeding up computations.

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

        if not self._is_output_node:
            input_pd = self._change_dims(input_pd, self._input_ndim)
        
        jacobian: Node.np_floating = self.jacobian(self._input_values)

        if self._is_output_node:
            # This is assumed to be the last node, returning the value of jacobian itself
            backprop_pd = jacobian
        elif jacobian.ndim < 2:
            # In case of matrix jacobian, multiplication is applied
            backprop_pd = jacobian @ input_pd
        else:
            # In case of tensor product, parial derivatives equal to elementwise product with
            # summation over all axes except for the first two
            backprop_pd = (jacobian * input_pd).sum(tuple(i for i in range(2, jacobian.ndim)))

        if reset_after:
            self.reset()

        return self._change_dims(backprop_pd, self._input_ndim)


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
        if optimized_target.shape != gradients.shape:
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
