import math

from pyml.hypothesis import Hypothesis


class Linear:
    """
    Linear regression model
    """
    params_index = 0
    result_index = 1

    def __init__(self, *training_examples, alpha=None, max_num_of_attempts=10000):
        """
        Method for initializing the object LinearRegression algorithm for the following model
        θ0*x0 + θ1*x1 + ... + θn*xn = y
        :param training_examples: Training example should be the following structure:
        ((x1, x2, ..., xn), y)
        x0=1 for every example
        """
        self._number_of_params = Linear.valid_training_examples(*training_examples) + 1
        self._training_examples = tuple(((1, *params), results) for params, results in training_examples)
        self._thetas = [1 for x in range(self._number_of_params)]
        self._alpha = alpha
        self._max_num_of_attempts = max_num_of_attempts

    @staticmethod
    def valid_training_examples(*training_examples):
        """
        Checking that training examples are valid
        :param training_examples: Training examples
        :return: None
        """
        params_len = None
        for example in training_examples:
            if len(example) != 2:
                raise ValueError('Training example should have two elements of the following structure: '
                                 '((x1, x2, ..., xn), y) !!')
            if params_len is None:
                params_len = len(example[Linear.params_index])
            if type(example[Linear.result_index]) != int and \
               type(example[Linear.result_index]) != float:
                raise ValueError(f'Result should be single value (Integer or Float number) !!')
            if params_len != len(example[Linear.params_index]):
                raise ValueError(f'Not all of training examples has the same length !!')
        return params_len

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def max_num_of_attempts(self):
        return self._max_num_of_attempts

    @max_num_of_attempts.setter
    def max_num_of_attempts(self, num_of_attempts):
        self._max_num_of_attempts = num_of_attempts

    @property
    def thetas(self):
        return list(self._thetas)

    def perform(self):
        origin_thetas = list(self._thetas)
        if self._alpha is None:
            self._alpha = 0.1
        number_of_attempts = 0
        proceed = True
        prev_step = None
        prev_relative_changes = None
        while proceed:
            try:
                prev_step = list(self._thetas)
                self.perform_one_step()
                relative_changes = sum(abs(prev_theta - cur_theta) for prev_theta, cur_theta in zip(prev_step, self._thetas)) / \
                                   sum(abs(cur_theta) for cur_theta in self._thetas)

                if prev_relative_changes is not None and relative_changes > prev_relative_changes:
                    if number_of_attempts < self._max_num_of_attempts:
                        self._alpha /= 3
                    else:
                        raise ValueError('Exceeded number of attempts to find the optimal parameters !!')
                elif relative_changes < 0.000000000001:
                    proceed = False
                prev_relative_changes = relative_changes
            except ValueError as ex:
                if 'NaN value !!' in ex.args[0]:
                    if number_of_attempts < self._max_num_of_attempts:
                        self._alpha /= 3
                        self._thetas = list(origin_thetas)
                        number_of_attempts += 1
                    else:
                        raise ValueError('Exceeded number of attempts to find the optimal parameters !!')

    def perform_one_step(self):
        temp_thetas = list(range(self._number_of_params))
        index_theta = 0
        hypothesis = Hypothesis(*self._thetas)
        num_examples = len(self._training_examples)
        while index_theta < len(temp_thetas):
            sum_of_examples = 0
            for num_example in range(0, num_examples):
                x_index_theta = self._training_examples[num_example][Linear.params_index][index_theta]
                hyp_res = hypothesis(*self._training_examples[num_example][Linear.params_index])
                res = self._training_examples[num_example][Linear.result_index]
                sum_of_examples += (hyp_res - res) * x_index_theta
            temp_thetas[index_theta] = self._thetas[index_theta] - self._alpha * sum_of_examples / num_examples
            if math.isnan(temp_thetas[index_theta]):
                raise ValueError(f'NaN value !! Value of theta[index_theta={index_theta}] is exceeded the limit !!')
            index_theta += 1
        self._thetas = temp_thetas
