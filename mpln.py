#
# Probabilistic Logic Node - MPLN
# in Meyers PhD Thesis 1990 (page. 43) 
# in Myers, C.; Aleksander, I. (1988) 
# Learning algorithms for probabilistic logic nodes. 
# Abstracts of 1st Annual INNS Meeting, Boston, p. 205 (abstract only).
#
#
# Try on Optdigits:
# python mpln.py -i /media/maurizio/LEXAR/TS -n 2 -w 5 -a 25
#

import numpy as np
import argparse
import sys, os
import random
from PIL import Image
from utilities import *
import time

from scipy import misc
import pickle
import copy

# import scikit-learn
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle

parser = argparse.ArgumentParser(description='mpln (aleksander training algorithm)')
parser.add_argument('-D', "--debuglvl", metavar='<debuglevel>', type=int, default=0, help='debug level', required=False)
parser.add_argument('-i', "--inputfile", metavar='<inputfile>', type=str, help='input file ', required=True)
parser.add_argument('-d', "--dumpfile", metavar='<dumpfile>', type=str, help='dump file ', required=False)
parser.add_argument('-C', "--code", metavar='<code>', type=str, default='t', help='coding', required=False, choices=['g', 't','c'])
parser.add_argument('-z', "--tics", metavar='<ticsno>', type=int, default=4, help='tic number', required=False)
parser.add_argument('-w', "--omega", metavar='<omega>', type=int, default=11, help='omega param', required=False)
parser.add_argument('-a', "--alpha", metavar='<alpha>', type=float, default=2.5, help='alpha param', required=False)
parser.add_argument('-m', "--mapping", metavar='<mapping>', type=int, help='map seed', required=False)
parser.add_argument('-x', "--xflag", help='interactive flag', default=False, action='store_true', required=False)
parser.add_argument('-n', "--bits", metavar='<bitsno>', type=int, default=2, help='bit number', required=False)
parser.add_argument('-c', "--cv", help='cv flag', default=False, action='store_true', required=False)

    
mypowers = 2**np.arange(32, dtype = np.uint32)[::]
tm_progress_ = 0.01
tm_starttm_ = time.time()

class PyramMPLN:
    """MPLN Pyramid Classifier (Myers,Aleksander training algorithm).
        (see Meyers PhD Thesis 1990 - page. 43) 
       
        This model uses the MPLN Pyramid neural network.
        It is a weightless neural network model to recognize binary patterns.
        
        Parameters
        ----------
        
        nobits: int, optional, default 8
            number of bits for RAM addresses (connectivity)
            should be in [1, 32]
        
        size: int, required
            input data size
        
        map: {None, int} , optional, default None
           mapping type (linear for None, random otherwise)
                               
        dblvl: {0, int} , optional, default None
           debug level
        
        Attributes
        ----------        
        """
    
    def _mk_tuple(self, X, n_ram, map, size):
        n_bit = self._nobits
        intuple = np.zeros(n_ram, dtype = np.int)
        for i in range(n_ram):
            for j in range(n_bit):
                idx = map[((i * n_bit) + j) % size]
                intuple[i] += mypowers[n_bit -1 - j]  * X[idx]
        return intuple
    
    # creates input-neurons mappings lists
    def _mk_mapping(self, size):
        random.seed(self._seed)
        pixels = np.arange(size, dtype=int)
        map = np.zeros(size, dtype=int)
        for i in range(size):
            j = i + random.randint(0, size - i - 1)
            temp = pixels[i]
            pixels[i] = pixels[j]
            pixels[j] = temp
            map[i] = pixels[i]
        return map

    def __init__(self,nobits,size,omega=11,alpha = 2.5, kappa=1, seed=None, map=None, dblvl=0):
        self._nobits = nobits
        self._datatype = 'binary'
        self._seed = seed
        self._skip = ' '
        self._retina_size = size
        self._kappa = kappa
        self._novals = omega
        self._values = np.arange(0,omega)
        self._colors = [colored.fg('#%02x%02x%02x'%(int(i*255/(omega-1)),int(i*255/(omega-1)),int(i*255/(omega-1)))) + colored.bg('#%02x%02x%02x'%(int(i*255/(omega-1)),int(i*255/(omega-1)),int(i*255/(omega-1)))) for i in range(omega)]
        # eq. 2.16 (Thesis Myers)
        probafunc = np.vectorize(lambda i: 1 / (1 + np.exp(alpha * ( -2 * ( i /float(omega - 1)) + 1 ))))
        # eq. 2.17 (Thesis Myers)
        #probafunc = np.vectorize(lambda i: 1 / float(omega - 1))
        self._probs = probafunc(self._values)
        if map is not None:
            self._mapping = self._mk_mapping(self._retina_size)
        else:
            self._mapping = np.arange(self._retina_size, dtype=int)
        fanin = self._retina_size
        self._nompln = []
        self._nloc = mypowers[self._nobits]
        while fanin > 1:
            if fanin % self._nobits == 0:   # no of mplns in first layer
                n = int(fanin / self._nobits)
            else:
                n = int(fanin / self._nobits + 1)
            self._nompln += [n]
            fanin = n
        # build pyramid layers
        self._layers = []
        self._mappings = [self._mk_mapping(self._retina_size) if map is not None else np.arange(self._retina_size, dtype=int)]
        self._sizes = [self._retina_size]
        for n in self._nompln[:-1]:
            self._mappings += [self._mk_mapping(n) if map is not None else np.arange(n, dtype=int)]
            self._sizes += [n]
        i = 0
        for n in self._nompln:
            self._layers += [np.full((n, self._nloc),self._values[int(self._novals/2)])]   # STEP 1. all nodes have 'u' stored in all addresses
            i += n
        self._nonodes = i
        self._upperbound = int(pow(2,i))
        self._dblvl = dblvl
                    
    def test(self, Xorig):
        output = [[None]*n for n in self._nompln]
        X = Xorig
        for l,n in enumerate(self._nompln):
            datain = X
            X = np.empty(n, dtype=np.int)
            for i,idx in enumerate(self._mk_tuple(datain, n, self._mappings[l], self._sizes[l])):
                storedvalue = self._layers[l][i][idx]
                valueidx = np.where(self._values == storedvalue)[0][0]
                output[l][i] = np.random.choice(2, 1, p=[1-self._probs[valueidx],self._probs[valueidx]])[0]
                #output[l][i] = 1 if storedvalue >= self._novals/2 else 0
                if self._dblvl > 1: print(output[l][i])
                X[i] = output[l][i]
            if self._dblvl > 1: print(X)
        self._lastout = output[-1][0]
        return output[-1][0]

    def train(self, Xorig, y):
        X = Xorig                                  # STEP 3. FEEDFORWARD (allow values to propagate through the net)
        output = [[None]*n for n in self._nompln]
        for l,n in enumerate(self._nompln):   
            datain = X
            X = np.empty(n, dtype=np.int)
            for i,idx in enumerate(self._mk_tuple(datain, n, self._mappings[l], self._sizes[l])):
                storedvalue = self._layers[l][i][idx]
                valueidx = np.where(self._values == storedvalue)[0][0]
                #output[l][i] = 1 if storedvalue >= self._novals/2 else 0
                output[l][i] = np.random.choice(2, 1, p=[1-self._probs[valueidx],self._probs[valueidx]])[0]
                if self._dblvl > 1: print(output[l][i])
                X[i] = output[l][i]
        r = 1 if y == output[-1][0] else -1
        X = Xorig
        for l,n in enumerate(self._nompln):   
            datain = X
            X = np.empty(n, dtype=np.int)
            for i,idx in enumerate(self._mk_tuple(datain, n, self._mappings[l], self._sizes[l])):
                storedvalue = self._layers[l][i][idx]
                valueidx = np.where(self._values == storedvalue)[0][0]
                if output[l][i] == 1: 
                    newvalidx =  valueidx + (self._kappa * r) 
                else:
                    newvalidx =  valueidx - (self._kappa * r) 
                if newvalidx < 0:
                    newvalidx = 0
                elif newvalidx > self._novals - 1:
                    newvalidx = self._novals - 1
                self._layers[l][i][idx] = self._values[newvalidx]
                output[l][i] = np.random.choice(2, 1, p=[1-self._probs[newvalidx],self._probs[newvalidx]])[0]
                X[i] = output[l][i]
        self._lastout = output[-1][0]     # store last output
                
    def fit(self, X, y):
        if self._dblvl > 0: timing_init()
        Error = 1
        LastError = 1
        epoch_count = 1
        while Error > .1:           # loop through epochs
            X_train, y_train = unison_shuffled_copies(X,y)
            mplnprev = copy.deepcopy(self)
            delta = 0
            for i,sample in enumerate(X_train):
                if self._dblvl > 1:  print("Label %d"%y[i])
                self.train(sample, y_train[i])
                delta += abs(y_train[i] - self._lastout)
                Error = delta/float(i+1)
                if self._dblvl > 1: print_data(sample,size); print(self); os.system('clear')
                if self._dblvl > 0: timing_update(i,y_train[i]==self._lastout,title='train %02d'%epoch_count,size=len(X_train),lasterr=LastError,error=Error)
            if LastError < Error:
                break
            else:
                if self._dblvl > 0: print('')
                LastError = Error
            epoch_count += 1
        if LastError < Error:    # restore last state in case of no decrease!
            self = mplnprev
            Error = LastError
        return self

    def predict(self,X):
        if self._dblvl > 0: timing_init()
        y_pred = np.array([])
        delta = 0
        for i,sample in enumerate(X):
            y_pred = np.append(y_pred,[ self.test(sample)])
            if self._dblvl > 0: timing_update(i,True,title='test    ',clr=color.GREEN,size=len(X))
        return y_pred

    def __str__(self,align='h'):
        rep = "MPLN Pyramid (RetinaSize: %d, NoBits: %d, Values: %r, Probs: %r,Layers: %r, Nodes: %d)\n"%(self._retina_size, self.getNoBits(),self.getValues(),self.getProbs(),self.getNoPLN(), self._nonodes)
        if align == 'v':
            endl = "\n"
        else:
            endl = ""
        for i,l in enumerate(self._layers):  
            rep += "[%d] "%(i)
            c = 0
            for r in l:
                if c == 0: 
                    rep += ""
                else:
                    rep += "    "
                c += 1
                rep += self._print__strip(r)
                rep += " "+endl
            rep += "\n"
        return rep
 
    def _print__strip(self,r):
        rep = ""
        for e in r:
            rep += self._colors[e] + '%s'%(self._skip) + colored.attr('reset')
        return rep
    
    def getDataType(self):
        return self._datatype

    def getOutput(self):
        return self.output[-1][0]
    
    def getNoBits(self):
        return self._nobits
    
    def getNoPLN(self):
        return self._nompln
    
    def getNoLayers(self):
        return self._nobits

    def getLayers(self):
        return self._layers
    
    def getLayer(self, index):
        return self._layers[index]
            
    def getMapping(self):
        return self._mapping
    
    def getMappings(self):
        return self._mappings
    
    def getSizes(self):
        return self._sizes

    def getValues(self):
        return self._values

    def getProbs(self):
        return self._probs

    def setDblevel(self, value):
        if value >= 0:
            self._dblvl = value

def main(argv):
    # parsing command line
    args = parser.parse_args(argv)
    debug = args.debuglvl
    size = args.tics

    # load dataset
    if os.path.isdir(args.inputfile):
        X, y = read_pics_dataset(args.inputfile,labels=[0,1])
        #X, y = read_pics_dataset(args.inputfile,labels=[0,1,2,3,4,5,6,7,8,9])
        #y[y == 2] = 0
        #y[y == 3] = 1
        X, y = shuffle(X, y)
        size = len(X[0])/32
    else:
        if not os.path.isfile(args.inputfile):
            raise ValueError("Cannot open file %s" % args.inputfile)
        else:
            X, y = read_dataset_fromfile(args.inputfile)
            X = binarize(X, size, args.code)
            y[y == -1] = 0
    class_names = np.unique(y)
    y = y.astype(np.int32)
        
    if args.cv:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X,X,y,y

    mpln = PyramMPLN(args.bits,len(X[0]),omega=args.omega,alpha=args.alpha,map=args.mapping,dblvl=debug)
    mpln.fit(X_train,y_train)
    if args.dumpfile is not None:
        pickle.dump(mpln,open(args.dumpfile,'w'))
    print('')
    y_pred = mpln.predict(X_test)
    print_confmatrix(confusion_matrix(y_test, y_pred))
    print("MPLN Acc. %.2f f1 %.2f"%(accuracy_score(y_test, y_pred),f1_score(y_test, y_pred, average='macro')))
    if mpln._dblvl > 1: print(mpln)
    return mpln
                   
if __name__ == "__main__":
    main(sys.argv[1:])
