import inspect


def is_neurons_same_level_type(first_neuron, second_neuron):
    has_level = ('level' in first_neuron and 'level' in second_neuron)
    has_not_level = ('level' not in first_neuron and 'level' not in second_neuron)
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

    def __init__(self, *thetas, bias=None, activation_func=None):
        if activation_func:
            arg_spec = inspect.signature(activation_func)
            if len(arg_spec.parameters) != 1:
                raise ValueError(f'Activation function [{activation_func}] should have 1 argument !!')
        self._bias = bias
        self._thetas = thetas
        self._inputs = [None] * len(self._thetas)
        self._slots = []
        self._next_slot = 0
        self._output = 0
        self._activation = activation_func

    def __lshift__(self, neuron):
        neuron._connect_to(self)

    def __rshift__(self, neuron):
        self._connect_to(neuron)

    def _connect_to(self, neuron):
        same_level_type, has_level = is_neurons_same_level_type(self, neuron)
        if not same_level_type:
            raise ValueError(f'Neuron [{self}] and Neuron [{neuron}] different level types !!')

        if neuron._next_slot >= len(neuron.inputs):
            raise ValueError(f'Neuron [{neuron}] has exceeded the maximum number of  connections !!')

        try:
            neuron._next_slot += 1
            if not has_level or neuron.level > self.level:
                self._slots.append(Neuron._InputSlot(neuron, neuron._next_slot))
            else:
                self._slots.append(Neuron._DeferredInputSlot(neuron, neuron._next_slot))
        except:
            neuron._next_slot -= 1

    def __setitem__(self, slot_index, value):
        self._inputs[slot_index] = value

    def __getitem__(self, slot_index):
        return self._inputs[slot_index]

    def activate(self):
        return self.output(*self._inputs)

    def output(self, *inputs):
        if inputs is not None and len(inputs) > 0:
            self.inputs = inputs
        bias_value = self._bias if self._bias else 0
        summary = bias_value + sum(theta * input if input else 0 for theta, input in zip(self._thetas, self.inputs))
        self._output = summary
        if self._activation:
            self._output = self._activation(summary)
        for slot in self._slots:
            slot(self._output)
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
    def activation(self, activation_func):
        self._activation = activation_func

    @activation.getter
    def activation(self):
        return self._activation
