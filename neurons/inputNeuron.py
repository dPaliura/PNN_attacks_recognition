from numpy import array

from neurons.neuron import Neuron
from neurons.functions.activ_funcs import get_activ_fun

class InputNeuron(Neuron):
    def __init__(self, activ_f="linear", activ_par=1, name=None):
        self._inputs_num = 1
        self._shift = False
        self._weights = array([1])
        self._transfer_f = None
        self._activ_f = get_activ_fun(activ_f, activ_par)
        if not (name is None):
            self.set_name(name)
        else:
            self._name = None


    def feed(self, x):
        return self._activ_f(x)

    def get_name(self):
        return self._name
    
    def set_name(self, name):
        if not isinstance(name, str):
            raise Exception("Can not set name of "+str(name.__class__)+". Only 'str' available")
        self._name = name
