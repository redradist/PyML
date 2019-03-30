import unittest

from pyml.regression.models.linear import Linear


class Testing_LinearRegression(unittest.TestCase):
    def setUp(self):
        """Currently nothing to do. Use it for initialization data before test"""
        pass

    def tearDown(self):
        """Currently nothing to do. Use it for reinitialization data after test"""
        pass

    def test__LinearRegression0__Valid(self):
        training_examples = [
            ((1, 1), 0.5),
            ((2, 2), 1),
            ((4, 4), 2)
        ]
        alg = Linear(*training_examples)
        alg.alpha = 0.1
        for i in range(1, 100000):
            alg.perform_one_step()
        theta0 = alg.thetas[0]
        theta1 = alg.thetas[1]
        theta2 = alg.thetas[2]
        self.assertAlmostEqual(theta0 + theta1 * 8 + theta2 * 8,
                               4,
                               places=5,
                               msg='Values are not equals !! '
                                   'Linear regression has not found proper prediction for test data !!')

    def test__LinearRegression1__Valid(self):
        training_examples = [
            ((1, 2), 0.8),
            ((4, 2), 1.4),
            ((4, 9), 3.5)
        ]
        alg = Linear(*training_examples)
        alg.alpha = 0.01
        for i in range(1, 100000):
            alg.perform_one_step()
        theta0 = alg.thetas[0]
        theta1 = alg.thetas[1]
        theta2 = alg.thetas[2]
        self.assertAlmostEqual(theta0 + theta1 * 16 + theta2 * 8,
                               5.6,
                               places=5,
                               msg='Values are not equals !! '
                                   'Linear regression has not found proper prediction for test data !!')

    def test__LinearRegressionAutomatic__Valid(self):
        training_examples = [
            ((1, 2), 0.8),
            ((4, 2), 1.4),
            ((4, 9), 3.5)
        ]
        alg = Linear(alpha=0.01,
                     max_num_of_attempts = 100000,
                     *training_examples)
        alg.perform()
        theta0 = alg.thetas[0]
        theta1 = alg.thetas[1]
        theta2 = alg.thetas[2]
        self.assertAlmostEqual(theta0 + theta1 * 16 + theta2 * 8,
                               5.6,
                               places=5,
                               msg='Values are not equals !! '
                                   'Linear regression has not found proper prediction for test data !!')
