from typing import List, Tuple
from random import gauss
from pickle import dump, load


# This class is intended to be used through a subclass
# implementation, such as `AgentDQN`.
class DQN:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layers: List[int] = [16, 16],
        alpha: float = 0.01,
        tau: float = 0.005
    ) -> None:
        self.__input_size = input_size
        self.__output_size = output_size
        self.__layers = layers + [self.__output_size]
        self.__alpha = alpha
        self.__tau = tau  # Soft update not implemented

        self.__weights, self.__biases = self.__add_layer(self.__layers)

    def __he(
        self,
        input_size: int,
        output_size: int
    ) -> List[List[float]]:
        std_dev = (2 / (input_size*(1 + self.__alpha**2)))**0.5

        return [
            [gauss(0, std_dev) for _ in range(input_size)]
            for _ in range(output_size)
        ]

    def __add_layer(
        self,
        layers: List[int]
    ) -> Tuple[List[List[List[float]]], List[List[float]]]:
        layers_weights = []
        layers_bias = []
        input_size = self.__input_size

        for layer_size in layers:
            weights = self.__he(
                input_size=input_size,
                output_size=layer_size
            )

            layers_weights.append(weights)
            layers_bias.append([0.0 for _ in range(layer_size)])

            input_size = layer_size

        return layers_weights, layers_bias

    def __leaky_relu(
        self,
        x: float,
        /
    ) -> float:
        return max(self.__alpha*x, x)

    def __leaky_relu_derivative(
        self,
        x: float,
        /
    ) -> float:
        return 1.0 if x >= 0 else self.__alpha

    def _fit_batch(
        self,
        batch: List[Tuple[List[float], int, float, List[float], bool]],
        target_nn: 'DQN',
        learning_rate: float = 0.001,
        gamma: float = 0.99
    ) -> None:
        for state, action, reward, next_state, done in batch:
            inputs, pre_activations, activations = self._forward(state)

            if done:
                y_target = reward
            else:
                _, _, online_output = self._forward(next_state)
                best_action = online_output[-1].index(max(online_output[-1]))

                _, _, target_output = target_nn._forward(next_state)
                q_value = target_output[-1][best_action]

                y_target = reward + gamma * q_value

            self.__backward(
                target=y_target,
                inputs=inputs,
                action=action,
                activations=activations,
                pre_activations=pre_activations,
                learning_rate=learning_rate
            )

    def _forward(
        self,
        x: List[float],
        /
    ) -> Tuple:
        inputs = [x]
        residuals = [x]
        pre_activations = []
        activations = []

        for i, layer_weights in enumerate(self.__weights):
            z = [
                sum(weight * inp for weight, inp in zip(weights, x))
                + self.__biases[i][j]
                for j, weights in enumerate(layer_weights)
            ]
            pre_activations.append(z)

            if i == len(self.__weights) - 1:
                activations.append(z)
            else:
                x = [self.__leaky_relu(val) for val in z]

                if len(x) == len(residuals[-1]):
                    x = [x[i] + residuals[-1][i] for i in range(len(x))]

                residuals.append(x)
                activations.append(x)
                inputs.append(x)

        return inputs, pre_activations, activations

    def __backward(
        self,
        target: float,
        inputs: List[List[float]],
        action: int,
        activations: List[List[float]],
        pre_activations: List[List[float]],
        learning_rate: float
    ) -> None:
        output = activations[-1]
        error = [0.0]*self.__output_size
        error[action] = output[action] - target

        deltas = [error]

        for j in range(len(self.__weights[-1])):
            for k in range(len(inputs[-1])):
                self.__weights[-1][j][k] -= (
                    learning_rate*deltas[0][j]*inputs[-1][k]
                )
            self.__biases[-1][j] -= learning_rate*deltas[0][j]

        for i in range(len(self.__weights)-2, -1, -1):
            layer_deltas = []
            current_weights = self.__weights[i + 1]

            for j in range(len(self.__weights[i])):
                error_sum = sum(
                    deltas[0][k]*current_weights[k][j]
                    for k in range(len(current_weights))
                )
                derivative = self.__leaky_relu_derivative(
                    pre_activations[i][j]
                )
                delta = error_sum*derivative
                layer_deltas.append(delta)

                for k in range(len(inputs[i])):
                    self.__weights[i][j][k] -= learning_rate*delta*inputs[i][k]
                self.__biases[i][j] -= learning_rate*delta

            deltas.insert(0, layer_deltas)

    def _save_model(
        self,
        filename: str
    ) -> None:
        with open(f'{filename}.pkl', 'wb') as f:
            dump(self, f)

    @staticmethod
    def _load_model(filename: str) -> 'DQN':
        with open(f'{filename}.pkl', 'rb') as f:
            return load(f)
