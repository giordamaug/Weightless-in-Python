# coding: utf-8
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
import time
from scipy import misc
import colored
import pickle
import copy

# import scikit-learn
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import shuffle
import scipy.sparse as sps

parser = argparse.ArgumentParser(description='mpln (aleksander training algorithm)')
parser.add_argument('-D', "--debuglvl", metavar='<debuglevel>', type=int, default=0, help='debug level', required=False)
parser.add_argument('-i', "--inputfile", metavar='<inputfile>', type=str, help='input file ', required=False)
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

white_color = colored.fg('#ffffff') + colored.bg('#ffffff')
black_color = colored.fg('#000000') + colored.bg('#000000')

def path_leaf(path):
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)

def read_pics_dataset(rootdir, labels=[0,1]):
    dirs = next(os.walk(rootdir))[1]
    X = np.empty([0,1024],dtype=int)
    y = np.empty([0],dtype=int)
    for dir in dirs:
        if int(dir) in labels:
            for f in os.listdir(os.path.join(rootdir,dir)):
                if not f.startswith('.'):
                    filename = os.path.join(rootdir,dir,f)
                    if os.path.isfile(filename):
                        #img = Image.open(filename).convert('L')
                        #np_img = np.array(img)
                        np_img = misc.imread(filename)
                        np_img = ~np_img
                        np_img[np_img > 0] = 1
                        np_img = np_img.flatten()
                        X = np.row_stack((X, np_img))
                        y = np.append(y, [int(dir)])
    return X,y

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def read_libsvm_dataset(datapath, size, code='t', scale=True):
    if datapath.endswith('.libsvm'):
        X, y = load_svmlight_file(datapath)
        if sps.issparse(X):
            X = X.toarray()
    else:
        raise Exception('Unsupported file format!')
    # dataset normalization (scaling to 0-1)
    if scale:
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        X = scaler.fit_transform(X)
        X = X.astype(np.float)
        y = y.astype(np.float)

    # binarize (histogram)
    tX = (X * size).astype(np.int32)

    if code == 'g':
        ticsize = size.bit_length()
        nX = np.zeros([tX.shape[0],tX.shape[1]*ticsize], dtype=np.int)
        graycode = genCode(ticsize)
        for r in range(tX.shape[0]):
            newRow = [int(e) for e in list(''.join([graycode[tX[r,i]] for i in range(tX.shape[1])]))]
            for i in range(tX.shape[1]*ticsize):
                 nX[r,i] = newRow[i]
    elif code == 't':
        nX = np.zeros([tX.shape[0],tX.shape[1]*size], dtype=np.int)
        for r in range(tX.shape[0]):
            for i in range(tX.shape[1]*size):
                if i % size < tX[r,int(i / size)]:
                    nX[r,i] = 1
    elif code == 'c':
        nX = np.zeros([tX.shape[0],tX.shape[1]*size], dtype=np.int)
        for r in range(tX.shape[0]):
            for i in range(tX.shape[1]*size):
                if i % size + 1== tX[r,int(i / size)]:
                    nX[r,i] = 1
    else:
        raise Exception('Unsupported data code!')
    return nX, y
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[0;32m'
    WHITEBLACK = '\033[1m\033[40;37m'
    BLUEBLACK = '\033[1m\033[40;94m'
    YELLOWBLACK = '\033[1m\033[40;93m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def timing_init():
    global tm_starttm_
    global tm_progress_
    tm_progress_ = 0.01
    tm_starttm_ = time.time()
    
def timing_update(i,match,clr=color.BLUE,title='train',size=100, lasterr=None,error=None):
    global tm_starttm_
    global tm_progress_
    tm,tme = compTime(time.time()-tm_starttm_,tm_progress_)
    tm_progress_ = printProgressBar(title, tm, tme, clr, color.RED, i+1,match,tm_progress_,size, lasterr, error)

def printProgressBar(title,time,etime,basecolor, cursorcolor, linecnt,match,progress,size, lasterr, error):
    barwidth = 70
    progress = linecnt / float(size);
    str = '%s |' % title
    pos = int(barwidth * progress)
    str += basecolor
    for p in range(barwidth):
        if p < pos:
            str += u'\u2588'
        elif p == pos:
            str += color.END + cursorcolor + u'\u2588' + color.END + basecolor
        else:
            str += u'\u2591'
    str += color.END + '| ' + "{:>3}".format(int(progress * 100.0)) + ' % ' + color.YELLOWBLACK + ' ' + etime + ' ' + color.WHITEBLACK + time + ' ' + color.END
    if lasterr is not None: 
        str += ' LastErr {:.2f} '.format(lasterr)
    if error is not None: 
        str += ' Err {:.2f} '.format(error)
    if match:
        str += color.GREEN + u'\U0001F604' + color.END
    else:
        str += color.RED + u'\U0001F625' + color.END
    sys.stdout.write("\r%s" % str)
    sys.stdout.flush()
    return progress

def compTime(deltatime,progress):
    hours, rem = divmod(deltatime*((1.0-progress) / progress), 3600)
    hourse, reme = divmod(deltatime, 3600)
    minutes, seconds = divmod(rem, 60)
    minutese, secondse = divmod(reme, 60)
    tm = "{:0>2}:{:0>2}:{:02.0f}".format(int(hours),int(minutes),seconds)
    tme = "{:0>2}:{:0>2}:{:02.0f}".format(int(hourse),int(minutese),secondse)
    return tm,tme

def print_confmatrix(table,fieldsize=3,decimals=3):
    nclasses = len(table)
    hfrmt = '{0: >%d}' % fieldsize
    dfrmt = '%%%dd' % fieldsize
    ffrmt = '%%%d.0%df' % (fieldsize,decimals)
    str = ('\n ' * fieldsize)
    for c in range(nclasses):
        str +=  ' '  + color.BOLD + hfrmt.format(c) + color.END
    print(str)
    print((' ' * fieldsize) + '┌' + ('─' * fieldsize + '┬') * (nclasses-1) + ('─' * fieldsize) + '┐')
    for k in range(nclasses):
        str = color.BOLD + hfrmt.format(k) + color.END
        for j in range(nclasses):
            if table[k][j]==0:
                str += '│' + (' '* fieldsize)
                continue
            if j==k:
                str += '│' + dfrmt % (table[k][j])
            else:
                str += '│' + color.RED + dfrmt % (table[k][j]) + color.END
        str += '│'
        print(str + '')
    print((' ' * fieldsize) + '└' + ('─' * fieldsize + '┴') * (nclasses-1) + ('─' * fieldsize) + '┘')

def print_data(r,size,skip=' '):
    rep = '\n'
    for i,e in enumerate(r):
        if e == 0: 
            rep += black_color + '%s'%(skip) + colored.attr('reset')
        else:
            rep += white_color + '%s'%(skip) + colored.attr('reset')
        if (i+1) % size == 0:
            rep += '\n'
    print(rep)

def genCode(n):
    if n == 0:
        return ['']
    
    code1 = genCode(n-1)
    code2 = []
    for codeWord in code1:
        code2 = [codeWord] + code2
        
    for i in range(len(code1)):
        code1[i] += '0'
    for i in range(len(code2)):
        code2[i] += '1'
    return code1 + code2   

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
        self._seed = seed
        self._skip = ' '
        self._retina_size = size
        self._kappa = kappa
        self._novals = omega
        self._values = np.arange(0,omega)
        self._colors = [colored.fg('#%02x%02x%02x'%(i*255/(omega-1),i*255/(omega-1),i*255/(omega-1))) + colored.bg('#%02x%02x%02x'%(i*255/(omega-1),i*255/(omega-1),i*255/(omega-1))) for i in range(omega)]
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
                #output[l][i] = random.choices(population=[0,1], weights=[self._probs[valueidx],1-self._probs[valueidx]])[0]
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
    if args.inputfile:
        if os.path.isdir(args.inputfile):
            X, y = read_pics_dataset(args.inputfile)
            X, y = shuffle(X, y)
            size = len(X[0])/32
        else:
            if not os.path.isfile(args.inputfile):
                raise ValueError("Cannot open file %s" % args.inputfile)
            if args.inputfile.endswith('.libsvm'):
                X, y = read_libsvm_dataset(args.inputfile, size, args.code)
                y[y == -1] = 0
            else:
                raise Exception('Unsupported file format!')
    else:     # toy dataset (try: mpln.py -n 4 -w 5 -D 1 -a 2.5)
        X = np.array([[1, 1, 1, 0, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 1, 1, 1],
                      [1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1],
                      [1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1],
                      [1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 1, 1, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0]], np.int32)

        y = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])
    class_names = np.unique(y)
    y = y.astype(np.int32)
        
    if args.cv:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X,X,y,y

    mpln = PyramMPLN(args.bits,len(X[0]),omega=args.omega,alpha=args.alpha,map=args.mapping,dblvl=debug)
    timing_init()
    if args.xflag: os.system('clear')
    if debug > 0: print(mpln)
    if args.xflag: 
        c = raw_input("[TRAIN] Press Enter to continue..."); 
        if c == 'c':
           args.xflag = False
    if debug > 0: os.system('clear')
    Error = 1
    LastError = 1
    while Error > .1:           # loop through epochs
        X_train, y_train = unison_shuffled_copies(X_train,y_train)
        mplnprev = copy.deepcopy(mpln)
        delta = 0
        for i,sample in enumerate(X_train):
            if debug > 1:  print("Label %d"%y[i])
            mpln.train(sample, y_train[i])
            delta += abs(y_train[i] - mpln._lastout)
            Error = delta/float(i+1)
            if debug > 0: print_data(sample,size); print(mpln)
            if args.xflag: 
                c = raw_input("[Test] Press Enter to continue...")
                print(c)
                if c == 'x':
                   args.xflag = False
            if debug > 0: os.system('clear')
            timing_update(i,y_train[i]==mpln._lastout,title='train',size=len(X_train),lasterr=LastError,error=Error)
        if LastError < Error:
            break
        else:
            LastError = Error
    if LastError < Error:    # restore last state in case of no decrease!
        mpln = mplnprev
        Error = LastError
    if args.dumpfile is not None:
        pickle.dump(mpln,open(args.dumpfile,'w'))
    print('')
    timing_init()
    y_pred = []
    delta = 0
    for i,sample in enumerate(X_test):
        prediction = mpln.test(sample)
        delta += abs(y_test[i] - prediction)
        y_pred += [prediction]
        timing_update(i,y_test[i]==prediction,title='test ',clr=color.GREEN,size=len(X_test),lasterr=Error,error=delta/float(i+1))
    timing_init()
    print_confmatrix(confusion_matrix(y_test, y_pred))
    print("MPLN Acc. %.2f"%(accuracy_score(y_test, y_pred)))
    if debug > 0: print(mpln)
    return mpln
                   
if __name__ == "__main__":
    main(sys.argv[1:])
