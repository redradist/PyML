import math

from pyml.neural.neuron import Neuron


class DeepNetwork:
    def __init__(self, inputs, levels, neurons_per_level, classes, activation_func=lambda y: math.exp(y)):
        self._inputs = []
        for num_level in range(0, inputs):
            self._inputs.append(Neuron(1,
                                       activation_func=lambda x: x,
                                       level=0))

        prev_level = self._inputs
        for num_level in range(1, levels+1):
            curr_level = []
            for num_neuron in range(0, neurons_per_level):
                neuron = Neuron(*([1] * neurons_per_level),
                                activation_func=activation_func,
                                level=num_level)
                curr_level.append(neuron)

            DeepNetwork.connect_levels(prev_level, curr_level)
            prev_level = curr_level

        self._outputs = []
        for num_output_level in range(0, classes):
            neuron = Neuron(1,
                            activation_func=lambda y: math.exp(y),
                            level=levels+1)
            self._outputs.append(neuron)
        DeepNetwork.connect_levels(prev_level, self._outputs)

    @staticmethod
    def connect_levels(prev_level, next_level):
        for prev_neuron in prev_level:
            for next_neuron in next_level:
                prev_neuron >> next_neuron

    def load_data(self, input):
        if type(input) == list:
            pass
        elif hasattr(input, 'read'):
            pass

    def learn(self):
        pass

    def inputs(self):
        pass

    def save_to(self):
        pass

    def save_to_file(self):
        pass

    def remove_data(self, some_data):
        pass

    def clear_data(self):
        pass

    def output(self, *args):
        if not args:
            raise ValueError()
        elif len(args) == 1:
            args = (args[0],)

        for input, arg in zip(self._inputs, args):
            result = input.outputs(arg)
        return self.activate()

    def activate(self):
        for output in self._outputs:
            yield output.activate()

    def __call__(self, *args, **kwargs):
        pass
