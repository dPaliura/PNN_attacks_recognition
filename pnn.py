from numpy import matrix, array as ar, unique

from neurons.inputNeuron import InputNeuron
from neurons.patternNeuron import PatternNeuron
from neurons.summaNeuron import SummaNeuron
from neurons.outputNeuron import OutputNeuron
from neurons.neuronsLayer import NeuronsLayer

class PNN:
    def __init__(self, train_in, train_out, input_names=None, gaussian_radius=0.3):
        train_in = matrix(train_in, dtype=float)
        train_out = ar(train_out)
        order = train_out.argsort()
        train_in = train_in[order,]
        train_out = train_out[order]
        inputs_num = train_in.shape[1]
        self._classes = unique(train_out, return_index=True, return_inverse=True, return_counts=True)
        self._classes_num = len(self._classes[0])
        if input_names is None: input_names = [None for i in range(inputs_num)]
        self._input_layer = NeuronsLayer([InputNeuron(name=input_names[i]) for i in range(inputs_num)])
        self._pattern_layer = NeuronsLayer([PatternNeuron(train_in[i,].tolist()[0], gaussian_radius) for i in range(train_in.shape[0])])
        self._summa_layer = NeuronsLayer([SummaNeuron(int(self._classes[3][i])) for i in range(self._classes_num)])
        self._out_neuron = OutputNeuron(int(self._classes_num), self._classes[0].astype(str))
    

    def _recognize(self, X):
        inp = self._input_layer.feed(X, False)
        pattern = ar(self._pattern_layer.feed(inp, True))
        summa_inp = list()
        for i in range(len(self._classes[0])):
            summa_inp.append(pattern[self._classes[2]==i])
        summa = self._summa_layer.feed(summa_inp, False)
        return self._out_neuron.feed(summa)


    def recognize(self, X, y=None):
        X = ar(X)
        ndims = len(X.shape)
        if (ndims == 1):
            res = [self._recognize(X)]
        elif (ndims == 2):
            res = list()
            prints = not y is None
            if prints:
                n = X.shape[0]
                i=0
            for row in X:
                recognized = self._recognize(row)
                res.append(recognized)
                if prints:
                    print(f"{i+1}/{n}", y[i], "recognized as", 
                    recognized, f"({+(recognized==y[i])})")
                    i += 1
        return ar(res)
