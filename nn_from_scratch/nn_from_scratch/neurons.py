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
            input_dim: Neuron.DIM,
            output_dim: Neuron.DIM,
            W: Neuron.np_floating = None,
            W_init_range: float = 1.0
    ):
        super().__init__(input_dim, output_dim, inner_ndim=2)

        if (self._input_ndim != 1) and self._input_dim[0] != self._output_dim[0]:
            raise Exception(
                f"Dimension Error: second input dimension {self._input_dim[0]} not equal to second output dimension {self._output_dim[0]}"
            )
        self._dtype = np.float32
        self._W_init_range = W_init_range
        # All input vectors would be concatenated with row of ones,
        # to transform matrix multiplication into WX + B
        self._W_dim = (self._input_dim[-1]+1, self._output_dim[-1])
        self.W_init(W)

    def _augment_input(self, X: Neuron.np_floating) -> Neuron.np_floating:
        '''
        Private method to augment the input values, by adding column vector of ones'''
        to_augment: Neuron.np_floating = np.ones((X.shape[0], 1), dtype=self._dtype)
        return np.concatenate([to_augment, X], axis=1)

    def W_init(self, W: Neuron.np_floating = None):
        '''
        Set weights W with given value or initialize it randomly.
        Also resets partial derivatives of W.
        '''

        self._initialized = False
        self._W: Neuron.np_floating = np.random.uniform(
            -self._W_init_range, self._W_init_range,
            self._W_dim
        ).astype(self._dtype) if W is None else W
        self.reset()

    def reset(self):
        super().reset()
        self._W_pd: Neuron.np_floating = None

    @property
    def W(self) -> Neuron.np_floating:
        return self._W

    def __call__(self, X: Neuron.np_floating) -> Neuron.np_floating:
        X_augmented = self._augment_input(X)
        self._input_values = X_augmented
        return self._change_dims(X_augmented @ self._W, self._output_ndim)

    def jacobian(self, X: Neuron.np_floating = None ) -> Neuron.np_floating:
        """
        The jacobian w.r.t. X.
        Note that backward function is overriden in this class, and by default
        using jacobian is avoieded.
        Parameters:
            X (np_floating(input_dim)): input values of the function
        Returns:
            d(FC)/dX (np_floating(input_dim, output_dim)): partial derivatives of function w.r.t. X
        """

        if self._input_ndim == 1:
            X_jac_dim: Iterable[int] = (1, self._input_dim[0], 1, self._output_dim[0])
        else:
            X_jac_dim: Iterable[int] = (*self._input_dim, *self._output_dim)

        X_jac: Neuron.np_floating = np.zeros(X_jac_dim, dtype=self._dtype)

        for i in range(X_jac_dim[0]):
            for j in range(X_jac_dim[1]):
                X_jac[i, j, i] = self._W[j+1]
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
        if self._output_ndim == 1:
            W_jac_dim: Iterable[int] = (*self._W_dim, 1, self._output_dim[0])
        else:
            W_jac_dim: Iterable[int] = (*self._W_dim, *self._output_dim)

        W_jac: Neuron.np_floating = np.zeros(W_jac_dim, dtype=self._dtype)
        for i in range(W_jac_dim[0]):
            for j in range(W_jac_dim[1]):
                W_jac[i, j, :, i] = X[:, i]
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

        input_pd = self._change_dims(input_pd, self._inner_ndim)
        # Optional calculations via jacobian
        if use_jacobian:
            self._w_pd = (self.W_jacobian(self._input_values) * input_pd).sum((2, 3))
            return super().backward(input_pd, reset_after)
        # Optimized calculations via dot products
        input_pd = input_pd.astype(self._dtype)
        self._W_pd = self._input_values.T.dot(input_pd)
        backprop_pd = input_pd.dot(self._W[1:, :].T)

        if reset_after:
            self.reset()

        return self._change_dims(backprop_pd, self._output_ndim)

    def optimize_weights(self, optimizer: Optimizer) -> None:
        super().optimize_weights(optimizer)
        self._W = optimizer.optimize([self._W], [self._W_pd])[0]


class Convolution(Neuron):
    """ Implements convolution layer."""

    def __init__(
        self,
        input_dim: Neuron.DIM,
        kernel_size: int,
        output_layers: int = 1,
        padding: int = 0,
        W: Neuron.np_floating = None,
        B: Neuron.np_floating = None,
        use_bias: bool = True,
        W_init_range: float = 1.0
    ):
        self._use_bias = use_bias
        self._kernel_size = kernel_size
        self._batch_size = input_dim[0]
        self._padding = padding
        self._output_layers = output_layers
        self._dtype = np.float32
        self._W_init_range = W_init_range
        output_dim = (
            self._batch_size, output_layers, input_dim[2]-kernel_size+1 + 2*padding, input_dim[3]-kernel_size+1 + 2*padding
        )
        super().__init__(input_dim, output_dim, inner_ndim=4)
        self._W_dim = (
            output_layers, input_dim[1], kernel_size, kernel_size
        )
        if use_bias:
            self._B_dim = output_dim[1:]
            self.B_init(B)

        self.W_init(W)

    def B_init(self, B: Neuron.np_floating = None):
        '''
        Set weights B with given value or initialize it randomly.
        Also resets partial derivatives of B.
        '''

        self._initialized = False
        self._B: Neuron.np_floating = np.random.uniform(
            -self._W_init_range, self._W_init_range,
            self._B_dim
        ).astype(self._dtype) if B is None else B
        self.reset()

    def W_init(self, W: Neuron.np_floating = None):
        '''
        Set weights W with given value or initialize it randomly.
        Also resets partial derivatives of W.
        '''

        self._initialized = False
        self._W: Neuron.np_floating = np.random.uniform(
            -self._W_init_range, self._W_init_range,
            self._W_dim
        ).astype(self._dtype) if W is None else W
        self.reset()

    def reset(self):
        super().reset()
        self._W_pd: Neuron.np_floating = None
        if self._use_bias:
            self._B_pd: Neuron.np_floating = None

    @property
    def W(self) -> Neuron.np_floating:
        return self._W

    @property
    def B(self) -> Neuron.np_floating:
        if not self._use_bias:
            raise ValueError("Bias is not initialized")
        return self._B

    def _convolve(self, T: Neuron.np_floating, W: Neuron.np_floating, add_padding: bool = False) -> Neuron.np_floating:
        T = self._change_dims(T, 4)
        W = self._change_dims(W, 4)
        if add_padding:
            pad_values = [
                (0, 0),
                (0, 0),
                (W.shape[2]-1, W.shape[2]-1),
                (W.shape[3]-1, W.shape[3]-1)
            ]
            T = np.pad(T, pad_width=pad_values).astype(np.float64)

        shape = (
            T.shape[0], T.shape[1],
            T.shape[2] - W.shape[2] + 1, T.shape[3] - W.shape[3] + 1,
            W.shape[2], W.shape[3]
        )
        batch_str, channel_str, kern_h_str, kern_w_str = T.strides
        strides = (
            batch_str,
            channel_str,
            kern_h_str, kern_w_str,
            kern_h_str, kern_w_str,
        )
        M = np.lib.stride_tricks.as_strided(T, shape=shape, strides=strides)
        y = np.einsum('bihwkl,oikl->bohw', M, W)
        return y

    def __call__(self, X: Neuron.np_floating) -> Neuron.np_floating:
        self._input_values = self._change_dims(X, 4).astype(self._dtype)

        self._input_values = np.pad(
            self._input_values,
            [
                (0, 0),
                (0, 0),
                (self._padding, self._padding),
                (self._padding, self._padding)
            ]
        )
        res = self._convolve(self._input_values, self._W)
        if self._use_bias:
            res += self._B

        return res

    def jacobian(self, X: Neuron.np_floating) -> Neuron.np_floating:
        raise NotImplementedError("Jacobians is not implemented for the convolution layer")

    def W_jacobian(self, X: Neuron.np_floating) -> Neuron.np_floating:
        raise NotImplementedError("Jacobians is not implemented for the convolution layer")

    def backward(
            self,
            input_pd: Neuron.np_floating,
            reset_after: bool = False
    ) -> Neuron.np_floating:
        """
        Overriden method for backpropogation. It does not use jacobians, but computes
        partial derivatives via dot products. Works faster than general approach.
        Parameters:
            input_pd (np_floating(n_output)): partial derivatives of the next layer
            reset_after (bool): if True, resets inner state after backpropogation
        Returns:
            output_pd (np_floating(n_input)): partial derivatives to propogate back"""

        if not self._initialized:
            raise RuntimeError("Backpropogation failed: system not initialized")

        input_pd = self._change_dims(input_pd, self._inner_ndim).astype(self._dtype)

        self._W_pd = np.zeros(self._W_dim, dtype=self._dtype)
        backprop_pd = np.zeros(self._input_dim, dtype=self._dtype)

        if self._use_bias:
            self._B_pd = np.sum(input_pd, axis=0)

        for batch_idx in range(self._batch_size):
            self._W_pd += np.concatenate(
                [
                    np.concatenate(
                        [
                            self._convolve(
                                self._input_values[
                                    batch_idx,
                                    depth,
                                ],
                                input_pd[batch_idx, conv_layer]
                            )
                            for depth in range(self._input_dim[1])
                        ], axis=1
                    )
                    for conv_layer in range(self._output_layers)
                ], axis=0
            )
        if self._padding != 0:
            for batch_idx in range(self._batch_size):
                backprop_pd[batch_idx] = np.sum(
                    [
                        np.concatenate(
                            [
                                self._convolve(
                                    input_pd[
                                        batch_idx,
                                        conv_layer,
                                        self._padding:-self._padding,
                                        self._padding:-self._padding
                                    ],
                                    self._W[conv_layer, depth, ::-1, ::-1],
                                    add_padding=True
                                )
                                for depth in range(self._input_dim[1])
                            ], axis=1
                        )
                        for conv_layer in range(self._output_layers)
                    ], axis=0
                )[0, :, :, :]
        else:
            for batch_idx in range(self._batch_size):
                backprop_pd[batch_idx] = np.sum(
                    [
                        np.concatenate(
                            [
                                self._convolve(
                                    input_pd[batch_idx, conv_layer],
                                    self._W[conv_layer, depth, ::-1, ::-1],
                                    add_padding=True
                                )
                                for depth in range(self._input_dim[1])
                            ], axis=1
                        )
                        for conv_layer in range(self._output_layers)
                    ], axis=0
                )[0, :, :, :]
        if reset_after:
            self.reset()

        return self._change_dims(backprop_pd, self._output_ndim)

    def optimize_weights(self, optimizer: Optimizer) -> None:
        super().optimize_weights(optimizer)
        if self._use_bias:
            self._W, self._B = optimizer.optimize([self._W, self._B], [self._W_pd, self._B_pd])
        else:
            self._W = optimizer.optimize([self._W], [self._W_pd])[0]
