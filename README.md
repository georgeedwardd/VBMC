
# Variational Bayesian Monte Carlo (VBMC)

A reimplementation and analysis of the Variational Bayesian Monte Carlo algorithm, which combines Gaussian Process surrogates, Bayesian Quadrature, and Variational Inference to perform sample-efficient Bayesian inference when evaluating the log joint is expensive.

## Overview
### Notebooks
The notebooks included in this repository provide a practical walkthrough of the main ideas behind Variational Bayesian Monte Carlo. The first covers simple Gaussian Process (GP) regression examples to illustrate surrogate modelling and uncertainty quantification. The second demonstrates Bayesian Quadrature (BQ), showing how integrals over functions can be estimated probabilistically using a GP. The final notebook brings these ideas together in the full VBMC algorithm, combining GP modelling, quadrature-based ELBO estimation, and variational optimisation with active sampling. Together, they form a clear path from foundational concepts to a working VBMC implementation.

### Derivations
The `derivations.ipynb` file contains informal, step-by-step outlines of the main mathematical results used throughout the project. The derivations focus on intuition and reasoning rather than full proof formality, providing a readable bridge between the mathematics and its implementation.

## Requirements
All dependencies required to run the notebooks and experiments are listed in `requirements.txt`.


## Features
- Implemented in JAX with autodiff.
- Rank-one GP updates and ARD kernels.
- Warm-up hyperparameter sampling.
- Initial sampling via Sobol sequences.
- Active sampling via CMA-ES.


## Notes
This project investigates VBMC’s efficiency, stability, and performance across various likelihoods, highlighting the influence of GP design, acquisition strategy, and variational flexibility.
