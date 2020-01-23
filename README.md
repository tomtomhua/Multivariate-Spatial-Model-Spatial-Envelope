# The Reproduction of P06

This is a reproduction of P06 paper.<br>
Environment: Python 3.7 + Jupyter Lab (or Jupyter NoteBook)<br>
PreInstalled Packages: sklearn, pandas, numpy, autograd, pymanopt, scipy, random, scipy, joblib, multiprocessing<br>
CPUs: the more, the better<br>

## Documents

* envelope_class.py: All functions need to be used for simulations
* final_simulation_one_sample.ipynb: An example of simulation for a single case ($s_i$: 10\*10, generate $Y_i$ with case I), with printed the itertimes, gap between two learning parameters and 
* simulation_single.py: An example of simulation for a single case ($s_i$: 10\*10, generate $Y_i$ with case I)
* simulation_test.py: An example of simple simulation for a single case ($s_i$: 4\*4, generate $Y_i$ with case I)
* simulation_total.py: An example of simulation for all cases (highly not recommend)
* \*.csv the output result of simulation.

## Executions

**Before execution**, change the file \*/lib/python3.7/site-packages/pymanopt/manifolds/rotations.py<br>
change "from scipy.misc import comb" into "from scipy.special import comb" at line 10<br>
This is an original problem of *pymanopt* due to the updating of *scipy*. For more details, see https://stackoverflow.com/questions/47151453/sklearn-import-error-importerror-cannot-import-name-comb

run simulation_test.py for testing<br>
run simulation_single.py for one implementation(for result with different parameters, change the result at line 70)<br>
run final_simulation_one_sample for one implementation with iteration information<br>
run simulation_total.py for whole implementation (highly not recommend, estimated to cost 200 hours for a computer with 16 CPUs)<br>

## Remarks

1. The optimization problem is hard to get the best result and could be very slow, for the following reasons:
    * the Hessian maybe singular (especially for small dimensions)
    * the optimization problem may have too much local minimizers(the difference of parameters between 2 iterations may change from 10e+6 numbers to 10e-6 suddenly, see final_simulation.ipynb)
2. For the first problem above, I add very samll disturbance to $X,Y,\theta$ when meet with the problems above
3. For the second problem above, I add a learning parameter for the update of $\theta$
4. The speed of optimization the highly related to number of CPUs (due to the pymanopt package will call all CPUs when solving manifold optimization problems)
5. for one sample, it cost about 4 hours with 16 CPUs to complete a calculation(simulation_single.py)
6. The running speed of the python file cannot be better than the one with R files.