from nn_from_scratch.interfaces import Node, Optimizer, NNTypes, Neuron
from collections.abc import Iterable


class NeuralNetwork:
    """One layer classifier NN.

    Layers: Linear -> SoftMax -> SoftMaxLoss
    """
    np_floating = NNTypes.np_floating

    def __init__(
            self,
            n_input: int, n_output: int, batch_size: int,
            optimizer: Optimizer,
            layers: Iterable[Node],
            loss_fn: Node
    ):
        self._n_input: int = n_input
        self._n_output: int = n_output
        self._batch_size: int = batch_size
        self._layers: Iterable[Node] = layers
        self._loss: Node = loss_fn
        self._opt: Optimizer = optimizer

    def fit(
            self,
            X: np_floating,
            y: np_floating,
            n_epochs: int | None = None
    ) -> float:
        assert X.shape[1:] == (self._batch_size, self._n_input)
        assert y.shape[1:] == (self._batch_size, self._n_output)
        self._epoch_no: int = 0
        self._epoch_loss: float = 0

        while True:
            # Resetting loss
            self._epoch_loss = 0
            # Train iteration
            for batch_idx in range(X.shape[0]):
                # Input iteration
                state: NeuralNetwork.np_floating = X[batch_idx]
                label: NeuralNetwork.np_floating = y[batch_idx]

                # Forward pass
                for layer in self._layers:
                    state = layer.forward(state)
                self._epoch_loss += self._loss.forward(state, label)

                # Backpropogation
                partial_derivative = self._loss.backward()
                for layer in self._layers[::-1]:
                    partial_derivative = layer.backward(partial_derivative)
                    if isinstance(layer, Neuron):
                        layer.optimize_weights(self._opt)
            self._epoch_no += 1
            self._visualize()
            # Termination check
            if n_epochs is None and self._opt.limit_reached():
                break
            elif n_epochs <= self._epoch_no:
                break

        return self._epoch_loss

    def _visualize(self) -> None:
        print(f"Epoch {self._epoch_no}, Loss: {self._epoch_loss}")

    def predict(self, x: np_floating) -> np_floating:
        # Forward pass
        state = x.copy()
        for layer in self._layers:
            state = layer.forward(state)
        return state
