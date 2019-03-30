import math
import unittest

from pyml.neural.network import Network


class Testing_DeepNetwork(unittest.TestCase):
    def setUp(self):
        """Currently nothing to do. Use it for initialization data before test"""
        pass

    def tearDown(self):
        """Currently nothing to do. Use it for reinitialization data after test"""
        pass

    def test__Neuron_DeepNetwork_3Inputs_1Middle_2Outputs__Valid(self):
        network = Network(number_of_inputs=3,
                          number_of_levels=1,
                          neurons_per_level=4,
                          number_of_classes=2,
                          activation_function=lambda y: math.exp(-y))
        # network.load_data([1, 2, 3])
        # network.remove_data(some_data)
        # network.clear_data()
        # network.learn()
        results = network.outputs(1, 1, 1)
        # network.save_to()
        # network.save_to_file()
        self.assertAlmostEqual(results[0],
                               0.819428384834,
                               places=5,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')
        self.assertAlmostEqual(results[1],
                               0.819428384834,
                               places=5,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')

    def test__Neuron_DeepNetwork_3Inputs_1Middle_3Outputs__Valid(self):
        network = Network(number_of_inputs=3,
                          number_of_levels=1,
                          neurons_per_level=4,
                          number_of_classes=3,
                          activation_function=lambda y: math.exp(-y))
        # network.load_data([1, 2, 3])
        # network.remove_data(some_data)
        # network.clear_data()
        # network.learn()
        results = network.outputs(1, 1, 1)
        # network.save_to()
        # network.save_to_file()
        self.assertAlmostEqual(results[0],
                               0.819428384834,
                               places=5,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')
        self.assertAlmostEqual(results[1],
                               0.819428384834,
                               places=5,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')
        self.assertAlmostEqual(results[2],
                               0.819428384834,
                               places=5,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')