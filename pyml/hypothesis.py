import re


class Hypothesis:
    """
    Class for describing hypothesis for model
    """
    _param_index = re.compile(r'(Th|th|Theta|theta)?(?P<index>\d+)')
    _var_index = re.compile(r'[Xx]?(?P<index>\d+)')
    _default_apply_function = lambda x: x

    def __init__(self, *init_theta, equation=None, **applied_function):
        self._equation = equation
        self._thetas = list(init_theta)
        self._applied_function = dict()
        duplicated_names = []
        invalid_names = []
        for theta, function in applied_function.items():
            if theta not in self._applied_function:
                match = re.match(Hypothesis._var_index, theta)
                if match:
                    theta_index = int(match.group('index'))
                    if theta_index < len(self._thetas):
                        self._applied_function[theta_index] = function
                    else:
                        raise ValueError(f'{theta} has index {theta_index} bigger (>) than '
                                         f'max parameter index {len(self._thetas)-1}')
                else:
                    invalid_names.append(theta)
            else:
                duplicated_names.append(theta)

        duplicated_names_error = None
        if duplicated_names:
            duplicated_names_error = f'{duplicated_names} are duplicated !! Remove duplication !!'

        invalid_names_error = None
        if invalid_names:
            invalid_names_error = f'{invalid_names} are invalid !! Should be something like this: Th*, th*, Theta* or theta*'

        if duplicated_names_error or invalid_names_error:
            raise ValueError(duplicated_names_error, invalid_names_error)

        for index in range(len(self._thetas)):
            if index not in self._applied_function:
                self._applied_function[index] = Hypothesis._default_apply_function

    def __setitem__(self, param, value):
        try:
            value = int(value)
        except:
            raise ValueError(f'{value} is not an integer !!')

        if type(param) is int:
            if param < len(self._thetas):
                self._thetas[param] = value
            else:
                raise ValueError(f'{param} is bigger than max index {len(self._thetas)-1}')
        elif type(param) is str:
            match = re.match(Hypothesis._param_index, param)
            if match:
                self._thetas[int(match.group('index'))] = value
            else:
                raise ValueError(f'{param} are invalid !! Should be something like this: Th*, th*, Theta* or theta*')

    def __getitem__(self, param):
        if type(param) is int:
            return self._thetas[param]
        elif type(param) is str:
            match = re.match(Hypothesis._param_index, param)
            if match:
                return self._thetas[int(match.group('index'))]
            else:
                raise ValueError(f'{param} are invalid !! Should be something like this: Th*, th*, Theta* or theta*')

    def __call__(self, *values, **kwargs):
        x = dict()
        duplicated_names = []
        invalid_names = []
        index = 0
        for value in values:
            x[index] = value
            index += 1
        for name, value in kwargs.items():
            if name not in x:
                match = re.match(Hypothesis._var_index, name)
                if match:
                    x[int(match.group('index'))] = value
                else:
                    invalid_names.append(name)
            else:
                duplicated_names.append(name)

        duplicated_names_error = None
        if duplicated_names:
            duplicated_names_error = f'{duplicated_names} are duplicated !! Remove duplication !!'

        invalid_names_error = None
        if invalid_names:
            invalid_names_error = f'{invalid_names} are invalid !! Should be something like this: X* or x*'

        if duplicated_names_error or invalid_names_error:
            raise ValueError(duplicated_names_error, invalid_names_error)

        if len(x) != len(self._thetas):
            raise ValueError(f'len(x)[{len(x)}] != len(thetas)[{len(self._thetas)}]')

        result = 0
        for index, value in x.items():
            result += self._thetas[index] * self._applied_function[index](value)
        return result
