# coding: utf-8
import numpy as np
import argparse
import sys
import random
import os
import collections
from utilities import *
import time
import math
undef = 4096
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import glob
    
    
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='pln')
parser.add_argument('-i', "--inputfile", metavar='<inputfile>', type=str, help='input file ', required=True)
parser.add_argument('-D', "--debuglvl", metavar='<debuglevel>', type=int, default=0, help='debug level', required=False)
parser.add_argument('-x', "--xflag", help='interactive flag', default=False, action='store_true', required=False)
parser.add_argument('-n', "--bits", metavar='<bitsno>', type=int, default=2, help='bit number', required=False)
parser.add_argument('-z', "--tics", metavar='<ticsno>', type=int, default=10, help='tic number', required=False)
parser.add_argument('-M', "--map", metavar='<mapseed>', type=int, default=-1,help='mapping seed', required=False)
parser.add_argument('-m', "--mode", metavar='<train mode>', type=str, default='normal', help='learning mode', required=False, choices=['normal', 'lazy','progressive'])
parser.add_argument('-p', "--policy", metavar='<policy>', type=str, default='c', help='policy', required=False, choices=['c', 'd','s','p'])
parser.add_argument('-C', "--code", metavar='<code>', type=str, default='t', help='coding', required=False, choices=['g', 't','c'])
parser.add_argument('-S', "--scale", default=True, action='store_true')
parser.add_argument('-c', "--cv", help='cv flag', default=False, action='store_true', required=False)


mypowers = 2**np.arange(32, dtype = np.uint32)[::]
tm_progress_ = 0.01
tm_starttm_ = time.time()
    

class WiSARD:
    """WiSARD Classifier """
    def _mk_tuple(self, X, n_ram, map, size):
        n_bit = self._nobits
        intuple = [0]*n_ram
        for i in range(n_ram):
            for j in range(n_bit):
                intuple[i] += mypowers[n_bit -1 - j] * X[map[((i * n_bit) + j) % size]]
        return intuple
    
    def __init__(self,  nobits, size, classes = [0,1], map=-1):
        self._nobits = nobits
        self._seed = map
        self._retina_size = size
        self._nobits = nobits
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
    
class PyramGSN:
    """GSN Pyramid Classifier.
        
        This model uses the GSN Pyramid neural network.
        It is a weightless neural network model to recognize binary patterns.
        For a introduction to GSN, please read 
        (https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2009-6.pdf)
        
        Parameters
        ----------
        
        nobits: int, optional, default 8
            number of bits for RAM addresses (connectivity)
            should be in [1, 32]
        
        size: int, required
            input data size
        
        map: {None, int} , optional, default None
           mapping type (linear for None, random otherwise)
                    
        policy: {'c', 'd'}, optional, default 'c'
           learning policy:
           'c' for random choice in case of multiple addresses
           'd' for deterministic choice (closest in hamming distante with output)
            
        laziness: True|False, optional, default False
           lazy mode of learning
           
        
        Attributes
        ----------        
        """
    
    def _mk_tuple(self, X, n_ram, map, size):
        n_bit = self._nobits
        intuple = [[0]]*n_ram
        for i in range(n_ram):
            newl = [0]
            for j in range(n_bit):
                idx = map[((i * n_bit) + j) % size]      
                if X[idx] == 1:
                    intuple[i] = [e + mypowers[n_bit -1 - j] for e in intuple[i]]
                elif X[idx] == undef:
                    newl = []
                    for k,e in enumerate(intuple[i]):
                        newl += [e, e + mypowers[n_bit -1 - j]]
                    intuple[i] = newl
        if self._dblvl > 3: print("TUPLE %r"%(intuple))
        return intuple
    
    # creates input-neurons mappings lists
    def __init__(self,nobits,size,map=-1, policy='c',mode='normal',dblvl=0):
        self._nobits = nobits
        self._seed = map
        self._retina_size = size
        fanin = self._retina_size
        self._values = [0, undef, 1]
        self._novals = 3   # omega for multivalued
        self._colors = [colored.fg('#%02x%02x%02x'%(int(i*255/(self._novals-1)),int(i*255/(self._novals-1)),int(i*255/(self._novals-1)))) + colored.bg('#%02x%02x%02x'%(int(i*255/(self._novals-1)),int(i*255/(self._novals-1)),int(i*255/(self._novals-1)))) for i in range(self._novals)]
        self._npln = []
        self._nbits = []
        self._nloc = mypowers[self._nobits]
        self._skip = ' '
        self._mode = mode
        if mode == 'normal':
            self.train = self._train_normal
        elif mode == 'lazy':
            self.train = self._train_lazy
        elif mode == 'progressive':
            self.train = self._train_progressive
        else:
            raise Exception('Wrong learning mode!')
            
        if policy == 'c':
            self._choice = self._rnd_choice
        elif policy == 'd':
            self._choice = self._det_choice
        elif policy == 's':
            self._choice = self._1st_choice
        else:
            raise Exception('Wrong learning policy!')
        self._policy = policy

        while fanin > 1:
            if fanin % self._nobits == 0: # no of plns in first layer
                n = int(fanin / self._nobits)
            else:
                n = int(fanin / self._nobits + 1)
            self._npln += [n]
            if int(fanin / self._nobits) == 0: 
                self._nbits += [fanin]
            else:
                self._nbits += [self._nobits]
            fanin = n
        # build pyramid layers
        self._layers = []
        mapping = np.arange(self._retina_size, dtype=int)
        if map > -1:
            np.random.seed(self.getSeed())
            np.random.shuffle(mapping)
        self._mappings = [mapping]
        self._sizes = [self._retina_size]
        for n in self._npln[:-1]:
            mapping = np.arange(n, dtype=int) 
            if map > -1: 
                np.random.shuffle(mapping)
            self._mappings += [mapping]
            self._sizes += [n]
        i = u = 0
        for n in self._npln:
            u += n
            self._layers += [np.full((n, self._nloc),undef)]
            i += n
        self._dblvl = dblvl
                    
    def test(self, Xorig):
        ''' Testing (Recall state of neurons)
            In the recall state the neuron produces outputs according to the following scheme:
            - If the number of ones in the addressable contents is greater than the number of zeros, then the neuron outputs a 1 value.
            - If the number of 0 values is greater then the neuron will output a O value.
            - If the numbers of ones and zeros is equal then the neuron outputs an undefined value.
            Thus, even if the addressed contents only contains a single 1 value, and the remainder are undefined, the neu- ron will output a 1 value. The reason for adopting this scheme is to minimise the propagation of undefined values.
        '''
        output = [[undef for y in range(n)] for n in self._npln]
        X = Xorig
        tuples = []
        if self._dblvl > 0: print("Input"); print("   ",self._print__strip(Xorig)); r = self._print__strip(X); print("Output (Recall)"," "*(len(Xorig)), "Addressable Cells")  # print layer output (and addressable set)
        for l,n in enumerate(self._npln):
            tuples += [self._mk_tuple(X, n, self._mappings[l], self._sizes[l])]
            X = np.empty(n, dtype=np.int)
            for i,idxs in enumerate(tuples[l]): # collect outputs
                a = self._layers[l][i]
                unique, counts = np.unique(a[idxs], return_counts=True)
                counts = collections.Counter(a[idxs]) 
                if counts[1] > counts[0]:
                    X[i] = 1
                elif counts[1] < counts[0]:
                    X[i] = 0
                else:
                    X[i] = undef
                output[l][i] = X[i]
            if self._dblvl > 0: r = self._print__strip(output[l]); print("[%d]"%(l),r,"  "*(len(Xorig)-len(output[l])+1),tuples[l])  # print layer output (and addressable set)
        return output[-1][0]

    def _det_choice(self,value,idxs):
        ''' return the address in the list of addresses 'idxs'
            with minmal hamming distance from 'value'
        ''' 
        return idxs[np.argmin([bin(idx^value).count('1') for idx in idxs])]

    def _rnd_choice(self,value,idxs):
        ''' return a random address in the list of addresses 'idxs'
        ''' 
        return random.choice(idxs)

    def _1st_choice(self,value,idxs):
        ''' return a random address in the list of addresses 'idxs'
        ''' 
        return idxs[0]

    def _seeking(self,Xorig, y):
        ''' Validating step (Seeking state)
            In the seeking state the neuron responds in the following manner:
            - The output is a 1 if all the addressable contents are 1.
            - The output is a 0 if all the addressable contents are 0.
            - The output is a u for all other values of the addressable contents.
            Thus when the addressable contents contain an undefined value, or when there is a mixture of 0 and 1 values then the output is undefined. This mode of operation helps to propagate undefined paths through networks of elements, and to seek out unused or conflicting values.
        '''
        self._output = [[undef for y in range(n)] for n in self._npln]
        X = Xorig
        self._tuples = []
        if self._dblvl > 0: print("Output (Seek)","Addressable Cells")  # print layer output (and addressable set)
        for l,n in enumerate(self._npln):
            tuples = self._mk_tuple(X, n, self._mappings[l], self._sizes[l])
            X = np.empty(n, dtype=np.int)
            for i,idxs in enumerate(tuples): # collect outputs
                countones = self._layers[l][i][idxs].sum()
                if countones == len(idxs):
                    X[i] = 1
                elif countones == 0:
                    X[i] = 0
                else:
                    X[i] = undef
                self._output[l][i] = X[i]
            if self._dblvl > 0: r = self._print__strip(self._output[l]); print("[%d]"%(l),r,tuples)  # print layer output (and addressable set)
            self._tuples += [tuples]
  
    def _propagate_lazy_old(self,l,i,y):
        if l > -1:
            if self._dblvl > 2: print("PROP(%d,%d,%d) out %d"%(l,i,y,self._output[l][i]))
            ram = self._layers[l][i]
            nbits = self._nbits[l]
            idxs = self._tuples[l][i]
            out = self._output[l][i]
            if out == undef:
                aidxs = np.array(idxs)
                undefidxs = aidxs[ram[idxs] == undef]
                if self._dblvl > 2: print("BACK[%d,%d] |%r,%r| %r %r %r"%(l,i,out,y,idxs,ram[idxs],undefidxs))
                idx = self._choice(y,undefidxs)  # select undef cell an set it to output
                bits = list('{0:0{nbit}b}'.format(idx,nbit=nbits))
                ram[idx] = y
                for j in range(nbits):
                    self._propagate_lazy(l-1,self._mappings[l][i*nbits + j],int(bits[j]))

    def _propagate_lazy(self,l,i,y):
        if l > -1:
            if self._dblvl > 2: print("PROP(%d,%d,%d) out %d"%(l,i,y,self._output[l][i]))
            ram = self._layers[l][i]
            nbits = self._nbits[l]
            idxs = self._tuples[l][i]
            out = self._output[l][i]
            size = self._sizes[l]
            #if out != y:
            if out != undef or out == y:
                None
            else:
                aidxs = np.array(idxs)
                newidxs = aidxs[ram[idxs] == y]
                undefidxs = aidxs[ram[idxs] == undef]
                if self._dblvl > 2: print("BACK[%d,%d] |%r,%r| %r %r %r %r"%(l,i,out,y,idxs,ram[idxs],newidxs,undefidxs))
                if len(newidxs) > 0:
                    idx = self._choice(y,newidxs)
                else:
                    idx = self._choice(y,undefidxs)  # select undef cell an set it to output
                bits = list('{0:0{nbit}b}'.format(idx,nbit=nbits))
                ram[idx] = y
                for j in range(nbits):
                    nexti = i*nbits + j
                    if nexti < size:
                        self._propagate_lazy(l-1,self._mappings[l][nexti],int(bits[j]))

    def _propagate(self,l,i,y):
        if l > -1:
            if self._dblvl > 2: print("PROP(%d,%d,%d) out %d"%(l,i,y,self._output[l][i]))
            ram = self._layers[l][i]
            nbits = self._nbits[l]
            idxs = self._tuples[l][i]
            out = self._output[l][i]
            size = self._sizes[l]
            if out == undef or out == y:
                aidxs = np.array(idxs)
                newidxs = aidxs[ram[idxs] == y]
                undefidxs = aidxs[ram[idxs] == undef]
                if self._dblvl > 2: print("BACK[%d,%d] |%r,%r| %r %r %r %r"%(l,i,out,y,idxs,ram[idxs],newidxs,undefidxs))
                if len(newidxs) > 0:
                    idx = self._choice(y,newidxs)
                else:
                    idx = self._choice(y,undefidxs)  # select undef cell an set it to output
                bits = list('{0:0{nbit}b}'.format(idx,nbit=nbits))
                ram[idx] = y
                for j in range(nbits):
                    nexti = i*nbits + j
                    if nexti < size:
                        self._propagate(l-1,self._mappings[l][nexti],int(bits[j]))
            #else:
            #    if l>0:
            #        raise Exception('Conflict in  learning! (Abort)')
            #    for idx in idxs:
            #        bits = list('{0:0{nbit}b}'.format(idx,nbit=nbits))
            #        ram[idx] = undef
            #        for j in range(nbits):
            #            nexti = i*nbits + j
            #            if nexti < size:
            #                self._propagate_nolazy(l-1,self._mappings[l][nexti],undef)

    def _train_normal(self, Xorig, y):
        ''' Learning step (no laziness) :
            In the learning state the neuron tries to associate the desired output with an existing cell in the addressable set. If it fails to do this it chooses an undefined cell and sets its value to the desired output. If there are a number of possible choices, three strategies can be adopted:
                1) policy='c' - a random address is selected. 
                2) policy='s' - the first address is selected
                3) policy='d' - the address with smallest hamming distance with the desired output is chosen
            It is necessary to choose one particular cell to represent the output value because the address of this cell is passed back down the input connections to become the desired output values for the previous layer.
        '''
        self._seeking(Xorig, y)
        # Learning values
        if self._dblvl > 0: print("Output (Learn)")  # print espected output
        self._propagate(len(self._layers)-1,0,y)

    def _train_lazy(self, Xorig, y):
        ''' Learning step (with laziness) :
            In the learning state the neuron tries to associate the desired output with an existing cell in the addressable set. If it fails to do this it chooses an undefined cell and sets its value to the desired output. If there are a number of possible choices, three strategies can be adopted:
                1) policy='c' - a random address is selected. 
                2) policy='s' - the first address is selected
                3) policy='d' - the address with smallest hamming distance with the desired output is chosen
            It is necessary to choose one particular cell to represent the output value because the address of this cell is passed back down the input connections to become the desired output values for the previous layer.
        '''
        self._seeking(Xorig, y)
        # Learning values
        if self._dblvl > 0: print("Output (Learn)")  # print espected output
        self._propagate_lazy(len(self._layers)-1,0,y)

    def _train_progressive(self, Xorig, y):
        ''' Learning step (progressi) :
            In the learning state the neuron tries to associate the desired output with an existing cell in the addressable set. If it fails to do this it chooses an undefined cell and sets its value to the desired output. If there are a number of possible choices, three strategies can be adopted:
                1) policy='c' - a random address is selected. 
                2) policy='s' - the first address is selected
                3) policy='d' - the address with smallest hamming distance with the desired output is chosen
            It is necessary to choose one particular cell to represent the output value because the address of this cell is passed back down the input connections to become the desired output values for the previous layer.
        '''
        self._output = [[undef for y in range(n)] for n in self._npln]
        X = Xorig
        self._tuples = []
        if self._dblvl > 0: print("Output (Seek)","Addressable Cells")  # print layer output (and addressable set)
        for l,n in enumerate(self._npln):
            tuples = self._mk_tuple(X, n, self._mappings[l], self._sizes[l])
            X = np.empty(n, dtype=np.int)
            for i,idxs in enumerate(tuples): # collect outputs
                for idx in idxs:
                    if self._layers[l][i][idx] == undef:   # set to defined value if undefined
                        self._layers[l][i][idx] = y
                X[i] = self._layers[l][i][idx]
            if self._dblvl > 0: r = self._print__strip(self._output[l]); print("[%d]"%(l),r,tuples, X)  # print layer output (and addressable set)

    def __str__(self,align='h'):
        ''' GSN pyramid printing function
        '''
        rep = "GSN Pyramid (Size: %d, NoBits: %d, Seed: %d, Layers: %r, Policy: %s, Mode: %r)\n"%(self._retina_size, self.getNoBits(),self.getSeed(),self.getNoPLN(),self._policy, self._mode)
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
            rep += self._colors[self._values.index(e)] + '%s'%(self._skip) + colored.attr('reset')
        return rep        
   
    def getNoBits(self):
        return self._nobits
    
    def getNoPLN(self):
        return self._npln
    
    def getNoLayers(self):
        return self._nobits

    def getLayers(self):
        return self._layers
    
    def getLayer(self, index):
        return self._layers[index]
            
    def getMapping(self):
        return self._mapping
    
    def getNBis(self):
        return self._nbits

    def getMappings(self):
        return self._mappings
    
    def getRevMappings(self):
        return self._revmappings

    def getSizes(self):
        return self._sizes
    
    def getSeed(self):
        return self._seed

def main(argv):
    # parsing command line
    args = parser.parse_args()
    debug = args.debuglvl
    size = args.tics
    
    # check dataset format (arff, libsvm)
    datafile = args.inputfile
    if os.path.isdir(args.inputfile):
        X, y = read_pics_dataset(args.inputfile,labels=[0,1])
        X, y = shuffle(X, y)
        size = len(X[0])/32
    else:
        if not os.path.isfile(args.inputfile):
            raise ValueError("Cannot open file %s" % args.inputfile)
        else:
            X, y = read_dataset_fromfile(args.inputfile)
            nX = binarize(X, size, args.code)
            y[y == -1] = 0
            y = y.astype(np.int32)

    class_names = np.unique(y)
    dataname = os.path.basename(datafile).split(".")[0]
        
    # create pyramids
    pln = PyramGSN(args.bits,len(nX[0]),map=args.map,dblvl=debug,policy=args.policy,mode=args.mode)

    if args.cv:
        kf = StratifiedKFold(random_state=0,n_splits=10, shuffle=True)
        timing_init()
        ylabels = np.array([])
        y_predRF = np.array([])
        y_predSVC = np.array([])
        y_pred = []
        for train_index, test_index in kf.split(X,y):
            nX_train, nX_test = nX[train_index], nX[test_index]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ylabels = np.append(ylabels,y_test); delta = 0
            for i,sample in enumerate(nX_train):
                if debug > 0:  print("Label %d"%y[i]) ; print_data(sample,args.tics); print(pln)
                pln.train(sample, y_train[i])
                res = pln.test(sample)
                delta += abs(y_test[i] - res)
                if args.xflag: input("[TRAIN] Press Enter to continue..."); os.system('clear')
                timing_update(i,y_train[i]==res,title='train ',size=len(nX_train),error=delta/float(i+1))
            timing_init()
            delta = 0
            for i,sample in enumerate(nX_test):
                res = pln.test(sample)
                y_pred += [res]
                delta += abs(y_test[i] - res)
                timing_update(i,y_test[i]==res,title='test  ',clr=color.GREEN,size=len(nX_test),error=delta/float(i+1))
            y_predRF = np.append(y_predRF, RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test))
            y_predSVC = np.append(y_predSVC, SVC(kernel='rbf').fit(X_train, y_train).predict(X_test))
            print()
        timing_init()
        print()
        print_confmatrix(confusion_matrix(ylabels, y_pred))
        print("GSN Acc. %.2f"%(accuracy_score(ylabels, y_pred)))
        print_confmatrix(confusion_matrix(ylabels, y_predRF))
        print("RF  Acc. %.2f"%(accuracy_score(ylabels, y_predRF)))
        print_confmatrix(confusion_matrix(ylabels, y_predSVC))
        print("SVM Acc. %.2f"%(accuracy_score(ylabels, y_predSVC)))
    else:
        nX_train, nX_test, y_train, y_test = nX,nX,y,y
        X_train, X_test = X,X
        timing_init()
        delta = 0
        for i,sample in enumerate(nX_train):
            if debug > 0:  print("Label %d"%y[i]) ; print_data(sample,args.tics); print(pln)
            pln.train(sample, y_train[i])        
            res = pln.test(sample)
            delta += abs(y_train[i] - res)
            if args.xflag: input("[TRAIN] Press Enter to continue..."); os.system('clear')
            timing_update(i,y_train[i]==res,title='train ',size=len(nX_train),error=delta/float(i+1))
        print()
        timing_init()
        y_pred = []
        delta = 0
        for i,sample in enumerate(nX_test):
            res = pln.test(sample)
            y_pred += [res]
            delta += abs(y_test[i] - res)
            timing_update(i,y_test[i]==res,title='test  ',clr=color.GREEN,size=len(nX_test),error=delta/float(i+1))
        print()
        print_confmatrix(confusion_matrix(y_test, y_pred))
        print("GSN Acc. %.2f"%(accuracy_score(y_test, y_pred)))
        y_predRF = RandomForestClassifier(random_state=0).fit(X_train,y_train).predict(X_test)
        print_confmatrix(confusion_matrix(y_test, y_predRF))
        print("RF  Acc. %.2f"%(accuracy_score(y_test, y_predRF)))
        y_predSVC = SVC(kernel='rbf').fit(X_train,y_train).predict(X_test)
        print_confmatrix(confusion_matrix(y_test, y_predSVC))
        print("SVM Acc. %.2f"%(accuracy_score(y_test, y_predSVC)))
    
if __name__ == "__main__":
    main(sys.argv[1:])
