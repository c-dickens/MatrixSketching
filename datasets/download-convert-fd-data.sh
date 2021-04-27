#!/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
gunzip covtype.data.gz
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a
python convert_to_numpy.py