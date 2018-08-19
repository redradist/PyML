import math
import unittest

from pyml.neural.deep_network import DeepNetwork


class Testing_DeepNetwork(unittest.TestCase):
    def setUp(self):
        """Currently nothing to do. Use it for initialization data before test"""
        pass

    def tearDown(self):
        """Currently nothing to do. Use it for reinitialization data after test"""
        pass

    def test__Neuron_DeepNetwork__Valid(self):
        network = DeepNetwork(inputs=3,
                              levels=1,
                              neurons_per_level=4,
                              classes=2,
                              activation_func=lambda y: math.exp(-y))
        # network.load_data([1, 2, 3])
        # network.remove_data(some_data)
        # network.clear_data()
        # network.learn()
        print('')
        results = network.output(0.2, 2, 1)
        for result in results:
            print(f'Result is {result}')
        # network.save_to()
        # network.save_to_file()
        # self.assertAlmostEqual(network.output(2, 1, 3, 4),
        #                        10,
        #                        places=5,
        #                        msg='Values are not equals !! '
        #                            'Neuron output value is not valid !!')
