#!/bin/bash

python experiment0-ihs-ols.py -t 5 &
python experiment1-error-opt-model.py -n 6000 -d 200 -i 10 -t 2 &&
python experiment2-ihs-timings.py -n 6000 -d 200 -i 10 -t 2