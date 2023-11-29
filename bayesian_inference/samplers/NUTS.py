## Before running this install:
## pip install jax
## pip install jaxlib
## pip install blackjax

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import pandas as pd

import blackjax

class NUTS:
    def __init__(self, likelihood, init_position, step_size=1e-3, inverse_mass_matrix=None, rng_key=None, warmup_steps=100):
        if rng_key is None:
            rng_key = jax.random.key(np.random.randint(2**32))

        self.likelihood = likelihood                                # likelihood object
        self.step_size = step_size                                  # stepsize (at the moment it doesn't matter)
        self.rng_key = rng_key                                      # key
        self.likelihood_func = lambda x: self.likelihood.logpdf(x)  # likelihood function
        self.init_position = init_position
        self.warmup_steps = warmup_steps

        ## Set up the warmup for the HMC sampler
        warmup = blackjax.window_adaptation(blackjax.nuts, self.likelihood_func)
        rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
        (self._state_init, parameters), _ = warmup.run(warmup_key, self.init_position, num_steps=warmup_steps)

        # the kernel performs one step
        self.kernel = blackjax.nuts(self.likelihood_func, **parameters).step
        self.states = []

    def step(self, rng_key, x):
        return self.kernel(rng_key, x)

    def inference_loop(self, num_samples, sample_key=None):
        if sample_key is None:
            self.rng_key, sample_key = jax.random.split(self.rng_key)

        print(f"Running the inference for {num_samples} samples")
        
        @jax.jit
        def one_step(state, rng_key):
            state, _ = self.kernel(rng_key, state)
            return state, state
    
        self.keys = jax.random.split(sample_key, num_samples)
        _, self.states = jax.lax.scan(one_step, self._state_init, self.keys)

        return self.states            

    def run(self, num_samples=100):
        self.states = []

        self.rng_key = jax.random.key(np.random.randint(2**16)) 

        ## Set up the warmup for the HMC sampler
        warmup = blackjax.window_adaptation(blackjax.nuts, self.likelihood_func)
        rng_key, warmup_key, sample_key = jax.random.split(self.rng_key, 3)
        (self._state_init, parameters), _ = warmup.run(warmup_key, self.init_position, num_steps=self.warmup_steps)

        self.kernel = blackjax.nuts(self.likelihood_func, **parameters).step
        self.states = []
        
        self.inference_loop(num_samples, sample_key=sample_key)
        return pd.DataFrame(self.states.position)
        