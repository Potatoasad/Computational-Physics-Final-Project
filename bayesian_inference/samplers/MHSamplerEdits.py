import numpy as np
from scipy.stats import norm
from tqdm import trange
from .DomainChanger import DomainChanger

import jax
import jax.numpy as jnp

def convert_to_jax_array(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, (float, int)):  # Check if the value is a float or int
            dictionary[key] = jnp.array([value])  # Replace with a JAX array of size 1
    return dictionary

def create_rng_key(backend):
    if backend == 'numpy':
        return np.random.seed(np.random.randint(2**32))
    elif backend == 'JAX':
        return jax.random.key(np.random.randint(2**32))
    else:
        raise ValueError(f"Do not recognize the {backend} backend")

class AbstractProposal:
    def __init__(self, step_size, backend='numpy', rng_key=None):
        self.step_size = step_size
        self.keys = None
        self.backend = backend
        self.rng_key = rng_key
        if rng_key is None:
            self.rng_key = create_rng_key(backend)

    def __call__(self, x):
        if self.keys is None:
            self.keys = list(x.keys())

        return {key: (value + self.next_step(key, value)) for key,value in x.items()}


class SphericalGaussianProposal(AbstractProposal):
    def __init__(self, step_size, backend='numpy', rng_key=None):
        super().__init__(step_size, backend=backend, rng_key=rng_key)

    def next_step(self, key, value):
        """
        Computes the next proposed value for for any variable inside the state dictionay x. 

        Within the state dictionary x (e.g. x = {'mu': [0.1, 0.2], 'kappa': 0.3}), each variable
        has a value that parameterizes its current state. This function will take in a value and 
        compute the next step in the proposal. 

        e.g. 
        >>> x = {'mu': [0.1, 0.2], 'kappa': 0.3}
        >>> SphericalGaussianProposal(0.1)
        >>> SphericalGaussianProposal.next_step('mu', x['mu']) # random normal with the right shape. 
        """
        if self.backend == 'numpy':
            if type(value) == np.ndarray:
                # Add a random gaussian of std step size to every entry of the matrix
                return np.random.randn(*value.shape)*self.step_size 
            elif (type(value) == type(2.0)) or (type(value) == np.float64):
                # Add a random gaussian of std step size to the float value
                return np.random.randn()*self.step_size
            else:
                raise ValueError(f"Got value of type {type(value)} having value {value} no idea what to do with this.")
        elif self.backend == 'JAX':
            new_key, self.rng_key = jax.random.split(self.rng_key)
            if type(value) == type(jnp.array([0.0, 1.0])):
                # Add a random gaussian of std step size to every entry of the matrix
                return jax.lax.stop_gradient(jax.random.normal(new_key , value.shape))*self.step_size 
            elif (type(value) == type(2.0)) or (type(value) == type(jnp.array([0.0, 10.0])[0])):
                # Add a random gaussian of std step size to the float value
                return jax.lax.stop_gradient(jax.random.normal(new_key))*self.step_size
            else:
                raise ValueError(f"Got value of type {type(value)} having value {value} no idea what to do with this. It isnt {type(jax.array([0.0, 10.0])[0])}")
        else:
            raise ValueError(f"Do not recognize the {self.backend} backend")




MAX_REJECTS_DEFAULT = 10000


class MHSampler:
    def __init__(self, likelihood, init_position, step_size=1, limits=None, rng_key=None, backend='numpy'):
        
        self.likelihood = likelihood
        self.backend = backend
        self.rng_key = rng_key
        if rng_key is None:
            self.rng_key = create_rng_key(backend)

        if limits is None:
            self.domain_changer = DomainChanger({key : 'infinite' for key in init_position.keys()}, backend=self.backend)
        else:
            limit_dict = {}
            for key in init_position:
                if key in limits:
                    limit_dict[key] = limits[key]
                else:
                    limit_dict[key] = 'infinite'
            self.domain_changer = DomainChanger(limit_dict, backend=self.backend)

        self.init_position = init_position

        if self.backend == 'JAX':
            self.init_position = jax.lax.stop_gradient(convert_to_jax_array(self.init_position))

        self.init_position_transformed = self.domain_changer.transform(self.init_position)
        self.likelihood_func = self.domain_changer.logprob_wrapped(self.likelihood.logpdf)

        #self.mu_sample = self.likelihood.mu_sample
        #self.var_sample = self.likelihood.var_sample
        #self.dim = self.mu_sample.shape[0]

        # should every model also have a data? or atleast mean and variance? 
        # it will be an array because each sample will have exactly same feature vectors

        self.step_size = step_size      # stepsize can be vector as long as dimensions match
        self.proposal = SphericalGaussianProposal(self.step_size, rng_key=self.rng_key, backend=self.backend)
        
        if rng_key is None:
            if self.backend == 'numpy':
                rng_key = np.random.seed(np.random.randint(2**32))
            elif self.backend == 'JAX':
                rng_key = jax.random.key(np.random.randint(2**32))
            else:
                raise ValueError(f"Do not recognize the {self.backend} backend")
        self.rng_key = rng_key

        self.history = []
        self.max_rejects_default = 10000 

        self.running_acceptances = 0
        self.total_steps = 0  
        self.num_samples = None                            
        
        #if proposal is None:
        #    proposal = lambda mu_trial: norm(mu_trial,self.step_size*np.ones(self.dim)).rvs()
        #self.proposal = proposal

        #if prior is None:
        #    prior = lambda mu_trial: norm(self.mu_sample,self.var_sample).logpdf(mu_trial)
        #self.prior = prior

    def accept_reject(self, p_accept):
        if self.backend == 'numpy':
            return (np.random.rand() < p_accept)
        elif self.backend == 'JAX':
            return bool(jax.random.uniform(self.rng_key) < p_accept)


    def step(self, x, max_rejects = MAX_REJECTS_DEFAULT): 
        accept = False
        rejects = 0

        while ((rejects < max_rejects) and (not accept)):
            x_proposal = self.proposal(x)

            log_likelihood_current = self.likelihood_func(x)
            log_likelihood_proposal = self.likelihood_func(x_proposal)

            #log_prior_current  = self.prior(mu_current)
            #log_prior_proposal = self.prior(mu_proposal)

            p_current = log_likelihood_current #+ log_prior_current
            p_proposal = log_likelihood_proposal #+ log_prior_proposal 

            if self.backend == 'JAX':
                p_accept = jnp.exp(p_proposal - p_current)
            elif self.backend == 'numpy':
                p_accept = np.exp(p_proposal - p_current)
            else:
                raise ValueError(f"Do not recognize the {self.backend} backend")
            
            accept = self.accept_reject(p_accept)

            self.total_steps += 1 

            if accept:
                self.running_acceptances += 1
                return x_proposal
            else:
                rejects += 1
                if rejects == max_rejects:
                    raise ValueError("The next proposal has been rejected {max_rejects} times! try changing something")       

    def run(self, n_steps = 1000, max_rejects=MAX_REJECTS_DEFAULT):
        self.num_samples = n_steps
        y = self.init_position_transformed
        self.history.append(self.domain_changer.inverse_transform(self.init_position_transformed))

        self.running_acceptances = 0
        self.total_steps = 0 

        print(f"Getting {n_steps} using Metropolis Hastings")

        for t in trange(n_steps):
            y = self.step(y, max_rejects=max_rejects)
            self.history.append(self.domain_changer.inverse_transform(y))

        
        acceptance_rate = self.running_acceptances/self.total_steps

        print(f"Sampling finished with an acceptance rate of {np.round(acceptance_rate*100, decimals=2)}")
        return self.result


    @property
    def result(self):
        if self.num_samples is None:
            return None

        def replace_array_with_float(val):
            if getattr(val, 'shape', None) == (1,):
                return float(val[0])
            elif type(val) == float:
                return val
            else:
                return np.array(val)

        return {key: ([replace_array_with_float(self.history[i][key]) for i in range(self.num_samples)]) for key in self.history[0].keys()}


