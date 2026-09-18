# Estimated bounds on Ky Fan k-negativity
This repository contains the numerical code and data for computing the maximum Ky Fan k-negativity
+ `k-neg-bound.py` is the code to generate the figure of estimated bounds.
+ `seesaw_results.pkl` is cached numerical results for k∈[{1,2,3,4,5,6} and p∈[0.1,1].
+ On the first run, the program computes everything and saves `seesaw_results.pkl`.
+ On subsequent runs, it loads the cached data and only redraws the figure.
