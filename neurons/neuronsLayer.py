from neurons.neuron import Neuron
from neurons.inputNeuron import InputNeuron

class NeuronsLayer:
    def __init__(self, neurons):
        self._set_neurons(neurons)
        self._size = len(self._neurons)
        if not self._size:
            raise Exception("It expected at least 1 object of class Neuron in 'neurons'")
        self._input_nums = [neuron.get_inputs_num() for neuron in self._neurons]
        self._samesized_inputs = not (False in 
                        [self._neurons[0].get_inputs_num() == neuron.get_inputs_num() for neuron in self._neurons])


    def _set_neurons(self, neurons):
        if False in map(lambda x: isinstance(x, Neuron), neurons):
            raise Exception("Each element in iterable object 'neurons' must be of class 'Neuron'")
        else:
            self._neurons = list(neurons)
    

    def feed(self, X, overall_input=False):
        if overall_input:
            if self._samesized_inputs:
                res = [neuron.feed(X) for neuron in self._neurons]
            else:
                raise Exception("Can not apply one input to each neuron in layer when neurons "+
                                "in it have different numbers of input values")
        else:
            res = list()
            for i in range(self._size):
                res.append(self._neurons[i].feed(X[i]))
        return res
