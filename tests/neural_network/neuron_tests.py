import math
import unittest

from pyml.neural_network.neuron import Neuron
from pyml.regression.models.linear import Linear


class Testing_LinearRegression(unittest.TestCase):
    def setUp(self):
        """Currently nothing to do. Use it for initialization data before test"""
        pass

    def tearDown(self):
        """Currently nothing to do. Use it for reinitialization data after test"""
        pass

    def test__Neuron__Valid(self):
        neuron = Neuron(1, 1, 1, 1, activation_func=lambda y: y)
        self.assertAlmostEqual(neuron.output(2, 1, 3, 4),
                               10,
                               places=5,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')

    def test__Neuron1__Valid(self):
        neuron = Neuron(1, 1, 1, 1, activation_func=lambda y: math.exp(y))
        self.assertAlmostEqual(neuron.output(2, 1, 3, 4),
                               22026,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')

    def test__Neuron2__Valid(self):
        neuron = Neuron(1, 1, 1, 1, activation_func=lambda y: math.pow(y, 2))
        self.assertAlmostEqual(neuron.output(2, 1, 3, 4),
                               100,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')

    def test__NeuronConnection0__Valid(self):
        neuron0 = Neuron(1, 1, 1, 1, activation_func=lambda y: math.pow(y, 2))
        neuron1 = Neuron(1, 1, 1, 1, activation_func=lambda y: math.pow(y, 2))
        neuron0 >> neuron1
        self.assertAlmostEqual(neuron0.output(2, 1, 3, 4),
                               100,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')
        self.assertAlmostEqual(neuron1.output(),
                               10000,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')

    def test__NeuronConnection1__Valid(self):
        neuron0 = Neuron(1, 1, 1, 1, activation_func=lambda y: math.pow(y, 2))
        neuron1 = Neuron(1, 1, 1, 1, activation_func=lambda y: math.pow(y, 2))
        neuron0 << neuron1
        self.assertAlmostEqual(neuron1.output(2, 1, 3, 4),
                               100,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')
        self.assertAlmostEqual(neuron0.output(),
                               10000,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')