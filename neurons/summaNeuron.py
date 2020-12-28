from neurons.neuron import Neuron


class SummaNeuron(Neuron):
    def __init__(self, inputs_num):
        Neuron.__init__(self, inputs_num, 
                        transfer_f="sum", activ_f="linear", activ_p=1/inputs_num, 
                        shift=False, weights=[1 for i in range(inputs_num)])

    
    def _feed(self, X):
        return self._activ_f(self._transfer_f(X))
