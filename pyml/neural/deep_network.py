import math

from pyml.neural.neuron import Neuron


class DeepNetwork:
    def __init__(self, num_levels, neurons_per_level):
        self._levels = []
        prev_level = []
        for num_level in range(0, num_levels):
            cur_level = []
            for num_neuron in range(0, neurons_per_level):
                neuron = Neuron([1] * neurons_per_level,
                                       activation_func=lambda y: math.exp(y),
                                       level=num_level)
                cur_level.append(neuron)

            self._levels = cur_level
            if prev_level:
                DeepNetwork.connect_levels(prev_level, cur_level)
            prev_level = cur_level

    @staticmethod
    def connect_levels(prev_level, curr_level):
        for prev_neuron in prev_level:
            for curr_neuron in curr_level:
                prev_neuron >> curr_neuron