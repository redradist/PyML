import inspect


class Neuron:
    """
    Neuron class for calculating output of the Neuron
    """
    class _InputSlot:
        """
        This class is used for creating connection between two neurons
        """
        def __init__(self, neuron, tie_neuron, slot_index, value=None):
            self._neuron = neuron
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

    @staticmethod
    def is_neurons_same_level_type(first_neuron, second_neuron):
        has_level = (first_neuron._level_number is not None and second_neuron._level_number is not None)
        has_not_level = (first_neuron._level_number is None and second_neuron._level_number is None)
        same_level_type = has_level or has_not_level
        return same_level_type, has_level

    def __init__(self, *thetas, bias=None, level_number=None, activation_function=None):
        if activation_function:
            arg_spec = inspect.signature(activation_function)
            if len(arg_spec.parameters) != 1:
                raise ValueError(f'Activation function [{activation_function}] should have 1 argument !!')
        self._bias = bias
        self._thetas = list(thetas)
        self._inputs = list(None for theta in thetas)
        self._slots = []
        self._next_slot = 0
        self._output = 0
        self._activation_function = activation_function
        self._level_number = level_number
        self._is_output_updating = False

    def __lshift__(self, neuron):
        neuron._connect_to(self)
        return self

    def __rshift__(self, neuron):
        self._connect_to(neuron)
        return neuron

    def _connect_to(self, neuron):
        same_level_type, has_level = Neuron.is_neurons_same_level_type(self, neuron)
        if not same_level_type:
            raise ValueError(f'Neuron [{self}] and Neuron [{neuron}] different level types !!')

        try:
            if neuron._next_slot >= len(neuron.inputs):
                neuron._thetas.append(1)
                neuron._inputs.append(None)
            if not has_level or neuron._level_number > self._level_number:
                self._slots.append(Neuron._InputSlot(self, neuron, neuron._next_slot))
            else:
                self._slots.append(Neuron._DeferredInputSlot(neuron, neuron._next_slot))
            neuron._next_slot += 1
        except Exception as ex:
            print(f'Exception caught: {ex}')
            neuron._next_slot -= 1

    def __enter__(self):
        if self._is_output_updating:
            raise ValueError(f'Neuron[{self}] is already updating !!')
        self._is_output_updating = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._is_output_updating = False

    def __setitem__(self, slot_index, value):
        self._inputs[slot_index] = value

    def __getitem__(self, slot_index):
        return self._inputs[slot_index]

    def activate(self):
        return self.outputs(*self._inputs)

    def outputs(self, *inputs):
        with self:
            if inputs is not None and len(inputs) > 0:
                self.inputs = inputs

            bias_value = self._bias if self._bias else 0
            summary = bias_value + sum(theta * input if input else 0 for theta, input in zip(self._thetas, self.inputs))
            self._output = summary
            if self._activation_function:
                self._output = self._activation_function(summary)
            for slot in self._slots:
                slot(self._output)
        return self._output

    @property
    def bias(self): pass

    @bias.getter
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @property
    def thetas(self): pass

    @thetas.getter
    def thetas(self):
        return self._thetas

    @thetas.setter
    def thetas(self, thetas):
        if len(self._thetas) != len(thetas):
            raise ValueError("Size of input thetas [size=%d] is not equal internal thetas [size=%d] !!"
                             .format(len(thetas), len(self._thetas)))
        self._thetas = list(thetas)

    def add_input(self):
        self._thetas.append(1)
        self._inputs.append(None)
        return self

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        if len(self._inputs) != len(inputs):
            raise ValueError("Size of input inputs [size=%d] is not equal internal inputs [size=%d] !!"
                             .format(len(inputs), len(self._inputs)))
        self._inputs = list(inputs)

    @property
    def activation(self, activation):
        self._activation_function = activation

    @activation.getter
    def activation(self):
        return self._activation_function
