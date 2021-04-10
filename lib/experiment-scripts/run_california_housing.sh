#!/bin/bash

python experiment3-ihs-cal-housing.py -i 10 -t 50 -sparsify 0 -s 0.0 &&
python experiment3-ihs-cal-housing.py -i 10 -t 10 -sparsify 1  -s 0.125 &&
python experiment3-ihs-cal-housing.py -i 10 -t 50 -sparsify 1  -s 0.25 &&
python experiment3-ihs-cal-housing.py -i 10 -t 50 -sparsify 1  -s 0.5 
