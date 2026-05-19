# VBMC: Overview of Moving Parts

This notebook implements Variational Bayesian Monte Carlo (VBMC) as a coupled system of three interacting components: a Gaussian Process surrogate for an expensive log-likelihood, a variational Gaussian mixture model, and an adaptive data acquisition mechanism. The objective is to iteratively refine both the posterior approximation and the surrogate model while controlling where the expensive function is evaluated.

The structure below follows the execution order of the implementation.

---

## 1. Initialisation

The procedure begins with `initialise(...)`, which constructs:

* an initial set of input locations $x$ using sobol sampling,
* corresponding expensive log-likelihood evaluations $y$,
* an initial Gaussian mixture for the variational distribution.

This stage establishes a minimal but diverse dataset and provides a starting point for the variational family. The mixture is intentionally over-dispersed to avoid early collapse.

---

## 2. Gaussian Process setup

After initial data is collected, GP hyperparameters are sampled via `gp_hp(...)`.

This step:

* defines uncertainty over kernel structure,
* generates multiple GP configurations,
* enables marginalisation over hyperparameters rather than committing to a single surrogate model.

Kernel matrices and inverses are then computed using `kinv(...)`, forming the computational backbone for repeated GP inference during optimisation.

---

## 3. ELBO construction

The core optimisation target is the ELBO, implemented in `elbo(...)`.

This combines:

* an expectation term, computed using Bayesian quadrature under the GP surrogate,
* an entropy term from the variational Gaussian mixture,
* averaging over GP hyperparameter samples to reduce sensitivity to kernel choice.

The result is a stochastic but stabilised objective that reflects both surrogate uncertainty and variational flexibility.

---

## 4. Warm-up phase

The first outer iterations operate in a warm-up regime.

During this phase:

* the mixture parameters are perturbed using `jitter(...)`,
* exploration is prioritised over refinement,
* acquisition behaviour is more aggressive and less constrained.

This prevents early convergence to poor local optima and encourages broad coverage of the domain.

---

## 5. Main optimisation loop

Each outer iteration consists of several coordinated updates:

### 5.1 Parameter refinement

The variational parameters are updated using gradients of the ELBO (`grad_elbo`).

This step gradually aligns the mixture with regions of high posterior mass under the surrogate model.

---

### 5.2 ELBO monitoring

At fixed intervals:

* ELBO is evaluated,
* a variance-adjusted version (ELCBO) is computed,
* both are recorded for convergence diagnostics.

This provides a stability signal and allows detection of diminishing returns.

---

### 5.3 Adaptive termination of warm-up

Warm-up ends automatically when improvement in ELCBO becomes consistently small.

At this point:

* data is pruned using `trim(...)`,
* low-relevance samples are removed,
* the system transitions into fully adaptive mode.

---

## 6. Active sampling (acquisition step)

New evaluation points for the expensive function are selected using `active_sample(...)`.

This is the key exploration mechanism.

The acquisition function combines:

* GP predictive variance (uncertainty),
* variational density (where probability mass lies),
* GP mean (promising regions of the landscape),

and is maximised using a derivative-free optimiser (CMA-ES).

The resulting points are added to the dataset, ensuring that subsequent GP updates are informed by regions that matter for both uncertainty reduction and posterior refinement.

---

## 7. Mixture structure adaptation

The variational family is not fixed.

Instead:

* `add(...)` introduces new Gaussian components by perturbing existing ones,
* `remove(...)` prunes low-weight components.

This maintains a balance between expressiveness and parsimony, preventing both underfitting and unnecessary complexity.

---

## 8. GP hyperparameter updates

After each iteration:

* new GP hyperparameter samples are drawn,
* kernel matrices are updated accordingly.

The frequency of resampling decreases over time, reflecting increasing stability in the surrogate model.

---

## 9. Data management and trimming

Once sufficient convergence behaviour is observed:

* dataset trimming is performed,
* low-likelihood or uninformative points are removed,
* kernel inverses are recomputed.

This improves numerical stability and focuses computation on relevant regions.

---

## 10. Diagnostics and visualisation

If enabled, the system periodically generates:

* mixture structure plots,
* data point overlays,
* likelihood surface visualisations,
* ELBO / ELCBO convergence curves.

These serve purely as monitoring tools for the evolution of the variational approximation and surrogate accuracy.

---

## 11. Output of the procedure

The function returns:

* final dataset $ (x, y) $,
* learned variational mixture parameters,
* ELBO and ELCBO history,
* parameter trajectory across iterations,
* data acquisition history,
* warm-up completion point.

---

## Summary

VBMC here operates as a feedback loop:

* the GP models the expensive function,
* the variational mixture approximates the target distribution,
* the ELBO couples both,
* and the acquisition function decides where new information is most valuable.

Each component is updated in turn, producing a self-correcting system that gradually concentrates computation where it is most informative.
