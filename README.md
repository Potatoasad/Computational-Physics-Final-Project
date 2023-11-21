# Computational-Physics-Final-Project
 Final Project for the Computational Physics course at UT Austin

### Ideas
Project ideas
- [x] Sampling + gradient descent + packing problems + glass phase transition
- [x] Statistical Mechanics steady state problems
- [x] Bayesian Inference
      
- [ ]  Inspiral stochastic dynamics
- [ ] Differentiable ODE solver 
- [ ] Finding distribution samples are coming from
- [ ] Differentiable ODE solver with distribution of parameter estimation
- [ ] Stochastic stuff is common interest 
- [ ] Different cost functions corresponding to different measures of distance between distributions
- [ ] KPZ
- [ ] Imaginary time evolution and LQFT

### What's common in the things we've selected:

- [ ] A class to sample using any/many methods (or use pre packaged samplers):
  - [x] Inverting CDF
  - [x] Metropolis Hastings
    - [ ] Benchmark for a nice potential landscape:
      - [ ] e.g. $C(x) = \frac{1}{2}x^2$
  - [ ] Maybe Gradient based? Hamiltonian Monte Carlo?
        
- [ ] A class to define a probability distribution over the state space,
  - [ ] break up into Cost function $C(x)$ and prob distribution over the state space $\exp(-\beta C(x))$

- [ ] Classes that inherit from the above but is specific to the applications
      
```python
#import jax.numpy as jnp

try:
  import jax.numpy as np
else:
  import numpy as np
  

class Distribution:
  - pdf
	- logpdf return -0.5*x**2

class Sampler:
  - sample
  - distribution (object)

class MCMCSampler:
  - proposal step (acceptance criteria)
  - compute pdf ratio
  
class SamplerVisualization:
  - Sampler
  - Plots
  
class CornerPlot:
  

class StateSpaceAnimation
 - samples
 - animate() 
```

