from numpy import array as ar, exp

from neurons.neuron import Neuron
from neurons.functions.activ_funcs import get_activ_fun

class PatternNeuron(Neuron):
    def __init__(self, weights, gauss_radius=0.3):
        self._weights = ar(weights)
        self._inputs_num = len(self._weights)
        self._shift = False
        self._sigma_sq = gauss_radius**2
        self._transfer_f = lambda X: ar(X) - self._weights
        self._activ_f = lambda X: sum(exp(-(X**2)/self._sigma_sq))
    

    def _feed(self, X):
        return self._activ_f(self._transfer_f(X))

