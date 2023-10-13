# Computational-Physics-Final-Project
 Final Project for the Computational Physics course at UT Austin



### Ideas

Here are the ideas we're thinking about so far:

Project ideas

- [ ]  Inspiral stochastic dynamics
- [x] Bayesian Inference
- [ ] Differentiable ODE solver 
- [x] Finding distribution samples are coming from
- [ ] Differentiable ODE solver with distribution of parameter estimation
- [ ] Stochastic stuff is common interest 
- [ ] Different cost functions corresponding to different measures of distance between distributions
- [ ] KPZ
- [x] Imaginary time evolution and LQFT



### What's common in the things we've selected:

- [ ] A class to define a probability distribution over the state space,
  - [ ] break up into Cost function $C(x)$ and prob distribution over the state space $\exp(-\beta C(x))$
- [ ] A class to sample using any/many methods (or use pre packaged samplers):
  - [ ] Metropolis Hastings
  - [ ] Maybe Gradient based? Hamiltonian Monte Carlo?
- [ ] Benchmark for a nice potential landscape:
  - [ ] e.g. $C(x) = \frac{1}{2}x^2$
