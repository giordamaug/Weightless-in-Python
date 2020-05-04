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
parser.add_argument('-i', "--inputfile", metavar='<inputfile>', type=str, help='input file ', required=True)
parser.add_argument('-D', "--debuglvl", metavar='<debuglevel>', type=int, default=0, help='debug level', required=False)
parser.add_argument('-n', "--bits", metavar='<bitsno>', type=int, default=2, help='bit number', required=False)
parser.add_argument('-z', "--tics", metavar='<ticsno>', type=int, default=10, help='tic number', required=False)
parser.add_argument('-m', "--map", metavar='<mapseed>', type=int, default=-1,help='mapping seed', required=False)
parser.add_argument('-t', "--trainmode", metavar='<train mode>', type=str, default='normal', help='learning mode', required=False, choices=['normal', 'lazy','progressive'])
parser.add_argument('-p', "--policy", metavar='<policy>', type=str, default='d', help='policy', required=False, choices=['c', 'd'])
parser.add_argument('-M', "--methods", metavar='<methods>', type=str, choices=['WiSARD', 'PyramGSN', 'PyramMPLN', 'SVC', 'RF'], default='PyramGSN',help='method list', required=False,  nargs='+')
parser.add_argument('-C', "--code", metavar='<code>', type=str, default='t', help='coding', required=False, choices=['g', 't','c'])
parser.add_argument('-S', "--scale", default=True, action='store_true')
parser.add_argument('-c', "--cv", help='cv flag', default=False, action='store_true', required=False)

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
    #elif method=='PyramMPLN':
    #    return PyramMPLN()
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
            nX = X
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
        
    GSNpolicy = "d"
    GSNmode = "progressive"
    # train and validate
    if args.cv:
        kf = StratifiedKFold(random_state=0,n_splits=10, shuffle=True)
        predictions = [np.array([])]* len(args.methods)
        for i,m in enumerate(args.methods):
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
        for m in args.methods:
            clf = classifier(m, args.bits, len(nX[0]), class_names, args.debuglvl, args.map, args.trainmode, args.policy)
            if m in ['WiSARD', 'PyramGSN', 'PyramMPLN']: 
                y_pred = clf.fit(nX_train,y_train).predict_ck(nX_test,y_test)
            else:
                y_pred = clf.fit(X_train,y_train).predict(X_test)
            print_measures(m,y_test,y_pred)
    
if __name__ == "__main__":
    main(sys.argv[1:])