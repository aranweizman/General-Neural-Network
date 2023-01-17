class Net:
    def __init__(self, neuron_mapping, initial_values, biases, activation_functions, net_size):
        self._neuron_mapping = neuron_mapping  # mappings between neurons
        self._net_size = net_size  # number of neurons in net
        self._input_len = net_size - len(neuron_mapping)
        self._biases = biases
        if len(activation_functions) == 1:
            self._activation_functions = [activation_functions for _ in range(len(neuron_mapping))]
        else:
            self._activation_functions = activation_functions

        self._neuron_values = initial_values  # these include only non-input neurons
        self._mapper = {}

        # construct map between neurons
        self.INDEX = 0
        self.WEIGHT = 1
        neuron_index = self._input_len
        for neuron_feeders in self._neuron_mapping:
            self._mapper[neuron_index] = []
            for feeder in neuron_feeders:

                if feeder[self.INDEX] == neuron_index:  # self-referencing neuron
                    raise Exception(f"self referencing neuron - index #{feeder[self.INDEX]}")

                self._mapper[neuron_index] += [(feeder[self.INDEX], feeder[self.WEIGHT])]
            neuron_index += 1  # increment neuron index

    def propagate(self, input_layer=None):
        # default value of input layer is 0 array
        if input_layer is None:
            input_layer = [0 for _ in range(self._input_len)]

        # validate input length
        elif len(input_layer) != self._input_len:
            raise ValueError(f"input length{len(input_layer)} "
                             f"is incongruent to net input size({self._input_len})")

        # propagate once
        for i in range(len(self._neuron_values)):
            self._neuron_values[i] = 0
            for feeder in self._mapper[i + self._input_len]:
                if feeder[self.INDEX] >= self._input_len:  # feeder is non-input layer(hidden layer)
                    self._neuron_values[i] += feeder[self.WEIGHT] * \
                                              self._neuron_values[feeder[self.INDEX]]
                else:  # feeder is input layer
                    self._neuron_values[i] += feeder[self.WEIGHT] * \
                                              input_layer[feeder[self.INDEX]]
            self._neuron_values[i] = self._activation_functions[i](self._neuron_values[i])

    def __len__(self):
        return self._net_size
