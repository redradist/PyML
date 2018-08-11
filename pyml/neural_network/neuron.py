import inspect


class Neuron:
    """
    Neuron class for calculating output of the Neuron
    """
    def __init__(self, *thetas, bias=None, activation_func=None):
        if activation_func:
            arg_spec = inspect.signature(activation_func)
            if len(arg_spec.parameters) != 1:
                raise ValueError(f'Activation function [{activation_func}] should have 1 argument !!')
        self._bias = bias
        self._thetas = thetas
        self._activation = activation_func

    def output(self, *inputs):
        bias_value = self._bias if self._bias else 0
        summary = bias_value + sum(theta * input for theta, input in zip(self._thetas, inputs))
        if self._activation:
            return self._activation(summary)
        else:
            return summary

    @property
    def bias(self, bias):
        self._bias = bias

    @bias.getter
    def bias(self):
        return self._bias

    @property
    def thetas(self, thetas):
        self._thetas = thetas

    @thetas.getter
    def thetas(self):
        return self._thetas

    @property
    def activation(self, activation_func):
        self._activation = activation_func

    @activation.getter
    def activation(self):
        return self._activation
