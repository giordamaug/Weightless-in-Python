#
# Wikie Stonham and Aleksander Recognition Device - WiSARD
# Aleksander, I. (1988) 
#
# Try on Optdigits:
# python wisard.py -i dataset/iris.libsvm -n 2 -z 32
#
import numpy as np
from utilities import *
mypowers = 2**np.arange(32, dtype = np.uint32)[::]
    

class WiSARD:
    """WiSARD Classifier """
    def _mk_tuple(self, X, n_ram, map, size):
        n_bit = self._nobits
        intuple = [0]*n_ram
        for i in range(n_ram):
            for j in range(n_bit):
                intuple[i] += mypowers[n_bit -1 - j] * X[map[((i * n_bit) + j) % size]]
        return intuple
    
    def __init__(self,  nobits, size, classes = [0,1], map=-1, dblvl=0):
        self._nobits = nobits
        self._datatype = 'binary'
        self._seed = map
        self._dblvl = dblvl
        self._retina_size = size
        self._nloc = mypowers[self._nobits]
        self._classes = classes 
        self._nrams = int(size/self._nobits) if size % self._nobits == 0 else int(size/self._nobits + 1)
        self._mapping = np.arange(self._retina_size, dtype=int)
        self._layers = [np.full((self._nrams, self._nloc),0) for c in classes]
        if map > -1: np.random.seed(self._seed); np.random.shuffle(self._mapping)
        
    def train(self, X, y):
        ''' Learning '''
        intuple = self._mk_tuple(X, self._nrams, self._mapping, self._retina_size)
        for i in range(self._nrams):
            self._layers[y][i][intuple[i]] = 1

    def test(self, X):
        ''' Testing '''
        intuple = self._mk_tuple(X, self._nrams, self._mapping, self._retina_size)
        a = [[self._layers[y][i][intuple[i]] for i in range(self._nrams)].count(1) for y in self._classes]
        return max(enumerate(a), key=(lambda x: x[1]))[0]
    
    def fit(self, X, y):
        if self._dblvl > 0: timing_init()
        delta = 0
        for i,sample in enumerate(X):
            if self._dblvl > 1:  print("Label %d"%y[i])
            self.train(sample, y[i])        
            res = self.test(sample)
            delta += abs(y[i] - res)
            if self._dblvl > 0: timing_update(i,y[i]==res,title='train ',size=len(X),error=delta/float(i+1))
        if self._dblvl > 0: print()
        return self

    def predict(self,X):
        if self._dblvl > 0: timing_init()
        y_pred = np.array([])
        for i,sample in enumerate(X):
            y_pred = np.append(y_pred,[self.test(sample)])
            if self._dblvl > 0: timing_update(i,True,title='test  ',clr=color.GREEN,size=len(X))
        if self._dblvl > 0: print()
        return y_pred

    def predict_ck(self,X, y):
        if self._dblvl > 0: timing_init()
        y_pred = np.array([])
        delta = 0
        for i,sample in enumerate(X):
            res = self.test(sample)
            delta += abs(y[i] - res)
            y_pred = np.append(y_pred,[res])
            if self._dblvl > 0: timing_update(i,True,title='test  ',clr=color.GREEN,size=len(X),error=delta/float(i+1))
        if self._dblvl > 0: print()
        return y_pred

    def __str__(self):
        ''' Printing function'''
        rep = "WiSARD (Size: %d, NoBits: %d, Seed: %d, RAMs: %r)\n"%(self._retina_size, self._nobits,self._seed,self._nrams)
        for i,l in enumerate(self._layers):  
            rep += "[%d] "%(i)
            c = 0
            for r in l:
                if c == 0: 
                    rep += ""
                else:
                    rep += "    "
                c += 1
                for e in r:
                    if e == 1:
                        rep += '\x1b[5;34;46m' + '%s'%(self._skip) + '\x1b[0m'   # light blue
                    else:
                        rep += '\x1b[2;35;40m' + '%s'%(self._skip) + '\x1b[0m'   # black
                rep += "\n"
            rep += "\n"
        return rep   

    def getDataType(self):
        return self._datatype

    def getMapping(self):
        return self._mapping

    def getNoBits(self):
        return self._nobits

    def getNoRams(self):
        return self._nrams
    
