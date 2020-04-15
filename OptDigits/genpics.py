from PIL import Image
import numpy as np

import sys, getopt

def main(argv):
    inputfile = ''
    outputdir = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","odir="])
    except getopt.GetoptError:
        print 'genpics.py -i <inputfile> -o <outputdir>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'genpics.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--odir"):
            outputdir = arg
    counters = [1,1,1,1,1,1,1,1,1,1]
    with open(inputfile, "r") as f:
        for _ in xrange(21):
            next(f)
        linecnt = 0
        index = 0
        M = np.array([])
        for line in f:
            linecnt += 1
            if linecnt % 33 == 0: # this is the label
                index = 0
                M = M.reshape(32,32)
                rescaled = (255.0 / M.max() * (M - M.min())).astype(np.uint8)
                im = Image.fromarray(rescaled)
                label = int(line)
                print "\rWriting %s/%d/%d-%03d.png" % (outputdir,label,label,counters[label]),
                im.save("%s/%d/%d-%03d.png" % (outputdir,label,label,counters[label]))
                counters[int(line)] += 1
                M = np.array([])
            else:
                ar = np.array([255 if c == '0' else 0 for c in line[:-1]])
                M = np.concatenate((M,[255 if c == '0' else 0 for c in line[:-1]]))
                index += 1


if __name__ == "__main__":
    main(sys.argv[1:])
