# Matrix Sketching


## Dependencies
~Most of this code can be run using standard anaconda libraries.
The necessary packages are stored in the virtual environment `thesis-python` which 
can be activated by `source thesis-python/bin/activate`.~
Just do `workon sketching`.

*The experiments will run using the Discrete Cosine Transform, the following is optional.*
We also use a `fastwht' library for efficient computation of the Fast Walsh Hadamard Transform.
This can be found at `https://bitbucket.org/vegarant/fastwht/src/master/'
Then do `git clone https://bitbucket.org/vegarant/fastwht.git' and navigate to the directory.
Then run `python setup.py install` followed by `python test.py` to execute the tests.
Finally, obtain the path by running `pwd` which will print `*YOUR PATH*`and, as a hack (this bit should be improved somehow??) copy that 
to `line 7` of `srht_sketch.py` in the `sys.path.append(*YOUR PATH*)` so that the `fastwht.py` function can 
be read from the `hadamard.py` file of `https://bitbucket.org/vegarant/fastwht/src/master/`.
This can be avoided by adding the directory to the venv by `add2virtualenv .` if you have used `mkvirtualenv myenv
workon myenv` see `https://stackoverflow.com/questions/4757178/how-do-you-set-your-pythonpath-in-an-already-created-virtualenv/47184788#47184788`

Note that `https://bitbucket.org/vegarant/fastwht/src/master/` requires the `swig` software.
If you are running anaconda python then this can easily be obtained from 
`https://anaconda.org/anaconda/swig`.
Alternatively, this can be obtained via homebrew.

## Experiments
We provide the following experimentst that are all located in the `lib/experiment-scripts/` directory.

_IHS & CountSketch:Synthetic Data_
1. `experiment0-ihs-ols.py` examines the performance of random projections in the ihs vs classical setting.  
2. `experiment1-error-opt-model.py` examines how the ihs model fares under varying the sketches and sketch sizes
3. `experiment2-ihs-timings.py` runs the same experiment as in 2 but also obtains wall-clock times and test error performance.
