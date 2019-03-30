import math
import unittest

from pyml.neural.neuron import Neuron


class Testing_LinearRegression(unittest.TestCase):
    def setUp(self):
        """Currently nothing to do. Use it for initialization data before test"""
        pass

    def tearDown(self):
        """Currently nothing to do. Use it for reinitialization data after test"""
        pass

    def test__Neuron_LinearOutput__Valid(self):
        neuron = Neuron(1, 1, 1, 1, activation_function=lambda y: y)
        self.assertAlmostEqual(neuron.outputs(2, 1, 3, 4),
                               10,
                               places=5,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')

    def test__Neuron_ExponentialOutput__Valid(self):
        neuron = Neuron(activation_function=lambda y: math.exp(y))
        neuron.add_input().add_input().add_input().add_input()
        self.assertAlmostEqual(neuron.outputs(2, 1, 3, 4),
                               22026,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')

    def test__Neuron_SquareOutput__Valid(self):
        neuron = Neuron(activation_function=lambda y: math.pow(y, 2))
        neuron.add_input().add_input().add_input().add_input()
        self.assertAlmostEqual(neuron.outputs(2, 1, 3, 4),
                               100,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')

    def test__TwoNeuronsConnection_RightShiftOpr__Valid(self):
        neuron0 = Neuron(activation_function=lambda y: math.pow(y, 2))
        neuron0.add_input().add_input().add_input().add_input()
        neuron1 = Neuron(activation_function=lambda y: math.pow(y, 2))
        neuron1.add_input().add_input().add_input().add_input()
        neuron0 >> neuron1
        self.assertAlmostEqual(neuron0.outputs(2, 1, 3, 4),
                               100,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')
        self.assertAlmostEqual(neuron1.outputs(),
                               10000,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')

    def test__ThreeNeuronsConnection_RightShiftOpr__Valid(self):
        neuron0 = Neuron(1, 1, 1, 1, activation_function=lambda y: math.pow(y, 2))
        neuron1 = Neuron(activation_function=lambda y: math.pow(y, 2))
        neuron1.add_input().add_input().add_input().add_input()
        neuron2 = Neuron(1, 1, 1, 1, activation_function=lambda y: math.pow(y, 2))
        neuron0 >> neuron1
        neuron0 >> neuron2
        neuron1 >> neuron2
        self.assertAlmostEqual(neuron0.outputs(2, 1, 3, 4),
                               100,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')
        self.assertAlmostEqual(neuron1.outputs(),
                               10000,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')
        self.assertAlmostEqual(neuron2.outputs(),
                               102010000,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')

    def test__NeuronConnection1_LeftShiftOpr__Valid(self):
        neuron0 = Neuron(activation_function=lambda y: math.pow(y, 2))
        neuron0.add_input().add_input().add_input().add_input()
        neuron1 = Neuron(1, 1, 1, 1, activation_function=lambda y: math.pow(y, 2))
        neuron0 << neuron1
        self.assertAlmostEqual(neuron1.outputs(2, 1, 3, 4),
                               100,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')
        self.assertAlmostEqual(neuron0.outputs(),
                               10000,
                               places=0,
                               msg='Values are not equals !! '
                                   'Neuron output value is not valid !!')