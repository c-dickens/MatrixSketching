# Matrix Sketching


## Dependencies
Most of this code can be run using standard anaconda libraries.
The necessary packages are stored in the virtual environment `thesis-python` which 
can be activated by `source thesis-python/bin/activate`.

*The experiments will run using the Discrete Cosine Transform, the following is optional.*
We also use a `fastwht' library for efficient computation of the Fast Walsh Hadamard Transform.
This can be found at `https://bitbucket.org/vegarant/fastwht/src/master/'
Then do `git clone https://bitbucket.org/vegarant/fastwht.git' and navigate to the directory.
Then run `python setup.py install` followed by `python test.py` to execute the tests.
Finally, obtain the path by running `pwd` which will print `*YOUR PATH*`and, as a hack (this bit should be improved somehow??) copy that 
to `line 7` of `srht_sketch.py` in the `sys.path.append(*YOUR PATH*)` so that the `fastwht.py` function can 
be read from the `hadamard.py` file of `https://bitbucket.org/vegarant/fastwht/src/master/`.

Note that `https://bitbucket.org/vegarant/fastwht/src/master/` requires the `swig` software.
If you are running anaconda python then this can easily be obtained from 
`https://anaconda.org/anaconda/swig`.
Alternatively, this can be obtained via homebrew.