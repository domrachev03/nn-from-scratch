from .interfaces import Neuron, Optimizer
import numpy as np
from collections.abc import Iterable


class Linear(Neuron):
    """ Implements fully-connected linear layer. 
    The input there is concatenated with column vector of ones 

        .. math::
            FC(X) = WX + b
    """

    def __init__(
            self,
            input_dim: Iterable[int, int],
            output_dim: Iterable[int, int],
            W: Neuron.np_floating = None
    ):
        if input_dim[1] != output_dim[1]:
            raise Exception(
                f"Dimension Error: second input dimension {input_dim[1]} not equal to second output dimension {output_dim[1]}"
            )

        # All input vectors would be concatenated with row of ones,
        # to transform matrix multiplication into WX + B
        self._W_dim: Iterable[int, int] = (output_dim[0], input_dim[0]+1)

        super().__init__(input_dim, output_dim)

        self.W_init(W)

    def _augment_input(self, X: Neuron.np_floating) -> Neuron.np_floating:
        '''
        Private method to augment the input values, by adding column vector of ones'''
        to_augment: Neuron.np_floating = np.ones(self._input_dim[1])
        return np.vstack([to_augment, X])

    def W_init(self, W: Neuron.np_floating = None):
        '''
        Set weights W with given value or initialize it randomly.
        Also resets partial derivatives of W.
        '''

        self._initialized = False
        self._W: Neuron.np_floating = np.random.uniform(
            0.4, 0.6,
            self._W_dim
        ) if W is None else W
        self.reset()

    def reset(self):
        super().reset()
        self._W_pd: Neuron.np_floating = np.zeros(self._W_dim, dtype=np.float64)

    @property
    def W(self) -> Neuron.np_floating:
        return self._W

    def forward(self, X: Neuron.np_floating) -> Neuron.np_floating:
        '''Overriding forward function to augment the input with ones'''

        X_augmented = self._augment_input(X)
        return super().forward(X_augmented)

    def f(self, X: Neuron.np_floating) -> Neuron.np_floating:
        return self._W @ X

    def jacobian(self, X: Neuron.np_floating) -> Neuron.np_floating:
        """
        The jacobian w.r.t. X.
        Note that backward function is overriden in this class, and by default
        using jacobian is avoieded.
        Parameters:
            X (np_floating(input_dim)): input values of the function
        Returns:
            d(FC)/dX (np_floating(input_dim, output_dim)): partial derivatives of function w.r.t. X
        """

        X_jac_dim: Iterable[int] = (*self._input_dim, *self._output_dim)
        X_jac: Neuron.np_floating = np.zeros(X_jac_dim, dtype=np.float64)

        for i in range(X_jac_dim[0]):
            for j in range(X_jac_dim[1]):
                X_jac[i, j, :, j] = self._W[:, i+1]
        return X_jac

    def W_jacobian(self, X: Neuron.np_floating) -> Neuron.np_floating:
        """
        The jacobian w.r.t. W.
        Note that partial derivatives of W are computer in overriden backwards function.
        Parameters:
            X (np_floating(input_dim)): input values of the function
        Returns:
            d(FC)/dW (np_floating(W_dim, output_dim)): partial derivatives of function w.r.t. W
        """

        W_jac_dim: Iterable[int] = (*self._W_dim, *self._output_dim)
        W_jac: Neuron.np_floating = np.zeros(W_jac_dim, dtype=np.float64)
        X_augmented = self._augment_input(X)

        for i in range(W_jac_dim[0]):
            for j in range(W_jac_dim[1]):
                W_jac[i, j, i] = X_augmented[j]
        return W_jac

    def backward(
            self,
            input_pd: Neuron.np_floating,
            reset_after: bool = False,
            use_jacobian: bool = False
    ) -> Neuron.np_floating:
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
            self._w_pd = (self.W_jacobian(self._input_values) * input_pd).sum((2, 3))
            return super().backward(input_pd, reset_after)

        # Optimized calculations via dot products
        input_pd = input_pd.astype(np.float64)

        self._W_pd = input_pd.dot(self._input_values.T)
        backprop_pd = self._W[:, 1:].T.dot(input_pd)

        if reset_after:
            self.reset()

        return backprop_pd

    def optimize_weights(self, optimizer: Optimizer) -> None:
        super().optimize_weights()
        optimizer.optimize(self._W, self._W_pd)
