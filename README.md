# Weightless neural networks implementation suite
This python package is a suite of implementations for the following weightless neural models:
- Multivalued Probabilist Logic Networks (MPLN)
- Global Seeking Nodes (GSN)
- Wilkie, Stonham & Aleksander's Recognition Devices (WiSARD)

## Software Requirements

- Python 3.x
- Scikit-learn package for python (version >= 0.22)
- scipy package for python
- PIL package for python
- colored package for python

## Usage

The software suite includes a python script `wtests.py` to tests different weightless systems on some datasets provided in this package as examples.  

For a complete help of the `wtests.py` script type the following:

```
$ python wtests.py --help
usage: wtests.py [-h] -i <inputfile|dir> [-D <debuglevel>] [-n <bitsno>]
                 [-z <ticsno>] [-m <mapseed>] [-t <train mode>] [-p <policy>]
                 [-M <method> [<method> ...]] [-C <code>] [-c]

weightless models

optional arguments:
  -h, --help            show this help message and exit
  -i <inputfile|dir>, --inputfile <inputfile|dir>
                        input train dataset file (str) - allowed file format:
                        libsvm, arff - or directory of png images (see
                        OptDigits)
  -D <debuglevel>, --debuglvl <debuglevel>
                        debug level (int) - range >= 0 (default 0 no debug)
  -n <bitsno>, --bits <bitsno>
                        bit number (int) - range: 2-32 (default 2)
  -z <ticsno>, --tics <ticsno>
                        tic number (int) - range > 1 (default 10)
  -m <mapseed>, --map <mapseed>
                        mapping seed (int) - < 0 for linear mapping (default),
                        0 for rnd mapping with rnd seed, >0 for rnd mapping
                        with fixed seed
  -t <train mode>, --trainmode <train mode>
                        learning mode (str) - allowed values: "normal",
                        "lazy", "progressive" (default). Note: valid only for PyramGSN!
  -p <policy>, --policy <policy>
                        policy type - allowed values: "d" for deterministic
                        (default), "c" for random choice. Note: valid only for PyramGSN!
  -M <method> [<method> ...], --method <method> [<method> ...]
                        method list (str list) - allowed values: WiSARD,
                        PyramGSN, PyramMPLN, SVC, RF (default WiSARD)
  -C <code>, --code <code>
                        data encoding - allowed values: "g" for graycode, "t"
                        for thermometer (default), "c" for cursor
  -c, --cv              enable flag for 10-fold cross-validation on dataset
                        (default disabled). If not set, testing is done on the training dataset.
  ```

Note that while WiSARD is a multi-class classifier, MPLNs and GSNs are binary classifiers. All weightless models works on binary inputs. Nevertheless, this software offers different encoding schemes for real-to-binary conversion of data. Data transformation in transparent to the user: depending on the original dataset format a default binarization scheme is applied to data in order to be processed by weightless systems.

For example, you can execute a WiSARD system on the `australian.libsvm` dataset with the command:

```
$ python wtests.py -i dataset/australian.libsvm -M WiSARD -n 8 -z 64 -m 0 -c
      0   1
   ┌───┬───┐
  0│353│ 30│
  1│102│205│
   └───┴───┘
WiSARD Acc. 0.81 f1 0.80
```

in this case, LibSVM (real) data  are automatically binarized into a binary vector by the (default) thermometer encoding scheme (`-C t`) with a number of tics equal to 10 (`-z 64`). A `WiSARD` model is built with an addressing scheme of 8 bits (`-n 8`), a random mapping between input binary vector and RAMs (`-m 0`). The `WiSARD` model is validated on the input dataset by a 10-fold cross validation on the same dataset (`-c`).

You can run several cleassifiers at once on the same dataset. For example:

```
$ python wtests.py -i OptDigits/train -M PyramGSN RF -c

      0   1
   ┌───┬───┐
  0│183│  6│
  1│  8│190│
   └───┴───┘
PyramGSN Acc. 0.96 f1 0.96

      0   1
   ┌───┬───┐
  0│188│  1│
  1│  1│197│
   └───┴───┘
RF Acc. 0.99 f1 0.99
```


