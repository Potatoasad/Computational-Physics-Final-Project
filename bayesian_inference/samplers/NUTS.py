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

from .DomainChanger import DomainChanger

def convert_to_jax_array(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, (float, int)):  # Check if the value is a float or int
            dictionary[key] = jnp.array([value])  # Replace with a JAX array of size 1
    return dictionary

class NUTS:
    """
    Solves a problem specified by a likelihood object using the NUTS sampler. 
    """
    def __init__(self, likelihood, init_position, limits=None,
                 step_size=1e-3, inverse_mass_matrix=None, rng_key=None, 
                 warmup_steps=100):
        if rng_key is None:
            rng_key = jax.random.key(np.random.randint(2**32))

        if limits is None:
            self.domain_changer = DomainChanger({key : 'infinite' for key in init_position.keys()}, backend='JAX')
        else:
            limit_dict = {}
            for key in init_position:
                if key in limits:
                    limit_dict[key] = limits[key]
                else:
                    limit_dict[key] = 'infinite'
            self.domain_changer = DomainChanger(limit_dict, backend='JAX')

        self.likelihood = likelihood                                # likelihood object
        self.step_size = step_size                                  # stepsize (at the moment it doesn't matter)
        self.rng_key = rng_key                                      # key

        self.likelihood_func = self.domain_changer.logprob_wrapped(self.likelihood.logpdf)#lambda x: likelihood_func(x)  # likelihood function
        #my_init = init_position.copy()
        my_init = self.domain_changer.transform(convert_to_jax_array(init_position))
        print(init_position, my_init)
        self.init_position = my_init
        self.warmup_steps = warmup_steps

        ## Set up the warmup for the HMC sampler
        warmup = blackjax.window_adaptation(blackjax.nuts, self.likelihood_func)
        rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
        (self._state_init, self.parameters), _ = warmup.run(warmup_key, self.init_position, num_steps=warmup_steps)

        # the kernel performs one step
        self.kernel = blackjax.nuts(self.likelihood_func, **self.parameters).step
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
        self.rng_key = jax.random.key(np.random.randint(2**16)) 
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        
        self.inference_loop(num_samples, sample_key=sample_key)

        positions = self.domain_changer.inverse_transform(self.states.position)
        data = {k: np.array(v).reshape(-1) for k, v in positions.items()}
        #df = pd.DataFrame(data)
        return positions #df  #pd.DataFrame(positions)
        