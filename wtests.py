# coding: utf-8
import numpy as np
import argparse
import sys
import random
import os
from utilities import *
import time
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import glob


# load scikit-learn stuff
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import *
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
# benchmark classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# weightless classifiers
from gsn import *
from wisard import *
from mpln import *

# try:  https://raw.githubusercontent.com/giordamaug/WiSARD4WEKA/master/datasets/ionosphere.arff

parser = argparse.ArgumentParser(description='weightless models')
parser.add_argument('-i', "--inputfile", metavar='<inputfile|dir>', type=str, help='input train dataset file (str) - allowed file format: libsvm, arff - or directory of png images (see OptDigits)', required=True)
parser.add_argument('-D', "--debuglvl", metavar='<debuglevel>', type=int, default=0, help='debug level (int) - range >= 0 (default 0 no debug)', required=False)
parser.add_argument('-n', "--bits", metavar='<bitsno>', type=int, default=2, help='bit number (int) - range: 2-32 (default 2)', required=False)
parser.add_argument('-z', "--tics", metavar='<ticsno>', type=int, default=10, help='tic number (int) - range > 1 (default 10)', required=False)
parser.add_argument('-m', "--map", metavar='<mapseed>', type=int, default=-1,help='mapping seed (int) -  < 0 for linear mapping (default), 0 for rnd mapping with rnd seed, >0 for rnd mapping with fixed seed', required=False)
parser.add_argument('-t', "--trainmode", metavar='<train mode>', type=str, default='progressive', help='learning mode (str) - allowed values: "normal", "lazy", "progressive" (default)', required=False, choices=['normal', 'lazy','progressive'])
parser.add_argument('-p', "--policy", metavar='<policy>', type=str, default='d', help='policy type - allowed values: "d" for deterministic (default), "c" for random choice', required=False, choices=['c', 'd'])
parser.add_argument('-M', "--method", metavar='<method>', type=str, choices=['WiSARD', 'PyramGSN', 'PyramMPLN', 'SVC', 'RF'], default='WiSARD',help='method list (str list) - allowed values: WiSARD, PyramGSN, PyramMPLN, SVC, RF (default WiSARD)', required=False,  nargs='+')
parser.add_argument('-C', "--code", metavar='<code>', type=str, default='t', help='data encoding - allowed values: "g" for graycode, "t" for thermometer (default), "c" for cursor', required=False, choices=['g', 't','c'])
parser.add_argument('-c', "--cv", help='enable flag for 10-fold cross-validation on dataset (default disabled)', default=False, action='store_true', required=False)

def print_measures(method,labels,predictions):
    print_confmatrix(confusion_matrix(labels, predictions))
    print("%s Acc. %.2f f1 %.2f"%(method,accuracy_score(labels, predictions),f1_score(labels, predictions, average='macro')))

def classifier(method, nbit, size, classes, dblvl, map, mode, policy):
    if method=='WiSARD':
        return WiSARD(nbit,size,map=map,classes=classes,dblvl=dblvl)
    elif method== 'PyramGSN':
        return PyramGSN(nbit,size,map=map,dblvl=dblvl,policy=policy,mode=mode)
    elif method=='SVC':
        return SVC(kernel='rbf')
    elif method=='RF':
        return RandomForestClassifier(random_state=0)
    elif method=='PyramMPLN':
        return PyramMPLN(nbit,size,map=map,dblvl=dblvl,omega=11,alpha = 2.5)
    else:
        raise Exception('Unsupported classifier!')


def main(argv):
    # parsing command line
    args = parser.parse_args()
    debug = args.debuglvl
    size = args.tics
    
    # check dataset format (arff, libsvm)
    datafile = args.inputfile

    if is_url(args.inputfile):
        X, y = read_dataset_fromurl(args.inputfile)
        nX = binarize(X, size, args.code)
        y = y.astype(np.int32)
    else:
        if os.path.isdir(args.inputfile):
            X, y = read_pics_dataset(args.inputfile,labels=[0,1])
            X, y = shuffle(X, y)
            nX = X
            size = len(X[0])/32
        else:
            if not os.path.isfile(args.inputfile):
                # try to lad sklearn dataset from name
                try:
                    dataset = globals()['load_'+args.inputfile]()
                    idx = np.argwhere((dataset.target == 0) | (dataset.target == 1))
                    #X = dataset.images[idx]
                    #X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
                    X = dataset.data[idx]
                    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
                    y = dataset.target[idx]
                    y = y.reshape(y.shape[0])
                    nX = binarize(X, size, args.code)
                    y = y.astype(np.int32)
                except:
                    raise ValueError("Cannot open file %s" % args.inputfile)
            else:
                X, y = read_dataset_fromfile(args.inputfile)
                nX = binarize(X, size, args.code)
                y[y == -1] = 0
                y = y.astype(np.int32)

    class_names = np.unique(y)
    dataname = os.path.basename(datafile).split(".")[0]
        
    # train and validate
    if args.cv:
        kf = StratifiedKFold(random_state=0,n_splits=10, shuffle=True)
        predictions = [np.array([])]* len(args.method)
        for i,m in enumerate(args.method):
            ylabels = np.array([])
            for train_index, test_index in kf.split(X,y):
                nX_train, nX_test = nX[train_index], nX[test_index]
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                ylabels = np.append(ylabels,y_test)
                clf = classifier(m, args.bits, len(nX[0]), class_names, args.debuglvl, args.map, args.trainmode, args.policy)
                if m in ['WiSARD', 'PyramGSN', 'PyramMPLN']: 
                    predictions[i] = np.append(predictions[i],clf.fit(nX_train,y_train).predict_ck(nX_test,y_test))
                else:
                    predictions[i] = np.append(predictions[i],clf.fit(X_train,y_train).predict(X_test))
            print_measures(m,ylabels,predictions[i])
    else:
        nX_train, nX_test, y_train, y_test = nX,nX,y,y
        X_train, X_test = X,X
        for m in args.method:
            clf = classifier(m, args.bits, len(nX[0]), class_names, args.debuglvl, args.map, args.trainmode, args.policy)
            if m in ['WiSARD', 'PyramGSN', 'PyramMPLN']: 
                y_pred = clf.fit(nX_train,y_train).predict_ck(nX_test,y_test)
            else:
                y_pred = clf.fit(X_train,y_train).predict(X_test)
            print_measures(m,y_test,y_pred)
    
if __name__ == "__main__":
    main(sys.argv[1:])