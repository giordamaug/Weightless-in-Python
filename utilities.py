# coding: utf-8
import sys, os
import time
import colored

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import scipy.sparse as sps
import numpy as np
from scipy.io import arff
import urllib
from io import StringIO

white_color = colored.fg('#ffffff') + colored.bg('#ffffff')
black_color = colored.fg('#000000') + colored.bg('#000000')

def path_leaf(path):
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)

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
    str = '\n' + ' ' * fieldsize
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

def is_url(url):
    return urllib.parse.urlparse(url).scheme != ""

def read_dataset_fromurl(url):
    try:
        response = urllib.request.urlopen(url)
        datapath = urllib.parse.urlparse(url).path
    except:
        raise ValueError('Cannot open url %s'%url)
    f = StringIO(response.read().decode('utf-8'))
    if datapath.endswith('.libsvm'):
        X, y = load_svmlight_file(f)
        if sps.issparse(X):
            X = X.toarray()
        return X,y
    elif datapath.endswith('.arff'):
        data, meta = arff.loadarff(f)
        y = np.array(data['class'])
        X = np.array(data[meta.names()[:-1]].tolist(), dtype=np.float64)
        y = LabelEncoder().fit_transform(y)
        return X,y
    else:
        raise Exception('Unsupported file format!')


def read_dataset_fromfile(datapath):
    if datapath.endswith('.libsvm'):
        X, y = load_svmlight_file(datapath)
        if sps.issparse(X):
            X = X.toarray()
        return X,y
    elif datapath.endswith('.arff'):
        data, meta = arff.loadarff(open(datapath, 'r'))
        y = np.array(data['class'])
        X = np.array(data[meta.names()[:-1]].tolist(), dtype=np.float64)
        y = LabelEncoder().fit_transform(y)
        return X,y
    else:
        raise Exception('Unsupported file format!')

def binarize(X, size, code='t', scale=True):
    # dataset normalization (scaling to 0-1)
    if scale:
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        X = scaler.fit_transform(X)
        X = X.astype(np.float)

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
    return nX