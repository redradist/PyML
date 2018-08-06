import unittest

from pyml.hypothesis import Hypothesis


class Testing_Hypothesis(unittest.TestCase):
    def setUp(self):
        """Currently nothing to do. Use it for initialization data before test"""
        pass

    def tearDown(self):
        """Currently nothing to do. Use it for reinitialization data after test"""
        pass

    def test__LinearHypothesisQuadratic_Theta0Zero__Valid(self):
        hypo = Hypothesis(0, 1, theta1=lambda x: pow(x, 2))
        self.assertEqual(hypo(x0=1, x1=2), 4)

    def test__LinearHypothesisQuadratic__Valid(self):
        hypo = Hypothesis(1, 1, theta1=lambda x: pow(x, 2))
        self.assertEqual(hypo(x0=1, x1=2), 5)

    def test__LinearHypothesisQuadratic_SetTheta0__Valid(self):
        hypo = Hypothesis(1, 1, theta1=lambda x: pow(x, 2))
        hypo[0] = 2
        self.assertEqual(hypo[0], 2)

    def test__LinearHypothesisQuadratic_SetTheta1__Valid(self):
        hypo = Hypothesis(1, 1, theta1=lambda x: pow(x, 2))
        hypo[1] = 2
        self.assertEqual(hypo[1], 2)

    def test__LinearHypothesisQuadratic_SetTheta3__Valid(self):
        hypo = Hypothesis(1, 1, theta1=lambda x: pow(x, 2))
        with self.assertRaises(ValueError) as context:
            hypo[3] = 2

    def test__LinearHypothesisCubic_Theta0Zero__Valid(self):
        hypo = Hypothesis(0, 1, theta1=lambda x: pow(x, 3))
        self.assertEqual(hypo(x0=1, x1=2), 8)

    def test__LinearHypothesisCubic__Valid(self):
        hypo = Hypothesis(1, 1, theta1=lambda x: pow(x, 3))
        self.assertEqual(hypo(x0=1, x1=2), 9)

    def test__LinearHypothesisCubic_SetTheta0__Valid(self):
        hypo = Hypothesis(1, 1, theta1=lambda x: pow(x, 3))
        hypo[0] = 2
        self.assertEqual(hypo[0], 2)

    def test__LinearHypothesisCubic_SetTheta1__Valid(self):
        hypo = Hypothesis(1, 1, theta1=lambda x: pow(x, 3))
        hypo[1] = 2
        self.assertEqual(hypo[1], 2)

    def test__LinearHypothesisCubic_SetTheta3__Valid(self):
        hypo = Hypothesis(1, 1, theta1=lambda x: pow(x, 3))
        with self.assertRaises(ValueError) as context:
            hypo[3] = 2