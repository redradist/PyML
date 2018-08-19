import inspect


def is_neurons_same_level_type(first_neuron, second_neuron):
    has_level = (first_neuron._level is not None and second_neuron._level is not None)
    has_not_level = (first_neuron._level is None and second_neuron._level is None)
    return has_level or has_not_level, has_level


class Neuron:
    """
    Neuron class for calculating output of the Neuron
    """
    class _InputSlot:
        def __init__(self, tie_neuron, slot_index, value=None):
            self._tie_neuron = tie_neuron
            self._slot_index = slot_index
            self._value = value

        def get(self):
            return self._value

        def set(self, value):
            self._value = value

        def get_tie_neuron(self):
            return self._tie_neuron

        def __call__(self, value):
            self._value = value
            self._tie_neuron[self._slot_index] = self._value
            self._tie_neuron.activate()

    class _DeferredInputSlot(_InputSlot):
        def __call__(self, value):
            self._value = value
            self._tie_neuron[self._slot_index] = self._value

    def __init__(self, *thetas, bias=None, level_number=None, activation_function=None):
        if activation_function:
            arg_spec = inspect.signature(activation_function)
            if len(arg_spec.parameters) != 1:
                raise ValueError(f'Activation function [{activation_function}] should have 1 argument !!')
        self._bias = bias
        self._thetas = list(thetas)
        self._inputs = [None] * len(self._thetas)
        self._slots = []
        self._next_slot = 0
        self._output = 0
        self._activation = activation_function
        self._level = level_number
        self._is_output_updating = False

    def __lshift__(self, neuron):
        neuron._connect_to(self)

    def __rshift__(self, neuron):
        self._connect_to(neuron)

    def _connect_to(self, neuron):
        same_level_type, has_level = is_neurons_same_level_type(self, neuron)
        if not same_level_type:
            raise ValueError(f'Neuron [{self}] and Neuron [{neuron}] different level types !!')

        try:
            if neuron._next_slot >= len(neuron.inputs):
                neuron._thetas.append(1)
                neuron._inputs.append(None)
            if not has_level or neuron._level > self._level:
                self._slots.append(Neuron._InputSlot(neuron, neuron._next_slot))
            else:
                self._slots.append(Neuron._DeferredInputSlot(neuron, neuron._next_slot))
            neuron._next_slot += 1
        except Exception as ex:
            print(f'Exception caught: {ex}')
            neuron._next_slot -= 1

    def __setitem__(self, slot_index, value):
        self._inputs[slot_index] = value

    def __getitem__(self, slot_index):
        return self._inputs[slot_index]

    def activate(self):
        return self.outputs(*self._inputs)

    def outputs(self, *inputs):
        if self._is_output_updating:
            raise ValueError(f'Neuron[{self}] is already updating !!')

        self._is_output_updating = True
        if inputs is not None and len(inputs) > 0:
            self.inputs = inputs

        bias_value = self._bias if self._bias else 0
        summary = bias_value + sum(theta * input if input else 0 for theta, input in zip(self._thetas, self.inputs))
        self._output = summary
        if self._activation:
            self._output = self._activation(summary)
        for slot in self._slots:
            slot(self._output)
        self._is_output_updating = False
        return self._output

    @property
    def bias(self, bias):
        self._bias = bias

    @bias.getter
    def bias(self):
        return self._bias

    @property
    def thetas(self, thetas):
        self._thetas = thetas

    @thetas.getter
    def thetas(self):
        return self._thetas

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        if len(self._inputs) != len(inputs):
            raise ValueError()
        self._inputs = list(inputs)

    @property
    def activation(self, activation):
        self._activation = activation

    @activation.getter
    def activation(self):
        return self._activation
