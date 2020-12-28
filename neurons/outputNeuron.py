from neurons.neuron import Neuron


class OutputNeuron(Neuron):
    def __init__(self, inputs_num, class_names=None):
        Neuron.__init__(self, inputs_num, 
                        transfer_f="whichmax", 
                        shift=False, weights=[1 for i in range(inputs_num)])
        self.set_class_names(class_names)
        self._activ_f = lambda x: self._class_names[x]
    

    def set_class_names(self, class_names):
        if class_names is None:
            class_names = ["class-"+str(i) for i in range(1, self._inputs_num+1)]
            self._class_names = class_names
            return
        if len(class_names) != self._inputs_num:
            raise Exception("list 'class_names' must be of length equal to 'inputs_num' (" +
                            str(self._inputs_num) + ")")
        for name in class_names:
            if not isinstance(name, str):
                raise Exception("list 'class_names' must contain only strings")
        self._class_names = class_names
    

    def _feed(self, X):
        return self._activ_f(self._transfer_f(list(X)))
