#!/bin/bash

# python experiment0-ihs-ols.py -t 10 &
python experiment1-error-opt-model.py -n 6000 -d 200 -i 20 -t 10 &&
python experiment2-ihs-timings.py -n 6000 -d 200 -i 20 -t 10
