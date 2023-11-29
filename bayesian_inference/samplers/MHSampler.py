
import numpy as np
from scipy.stats import norm

class MHSampler:

    def __init__(self, likelihood, init_position, step_size=1, proposal=None, prior = None, rng_key=None):
        
        self.likelihood = likelihood
        self.init_position = init_position
        self.likelihood_func = lambda x: self.likelihood.logpdf(x)  # likelihood function

        self.mu_sample = self.likelihood.mu_sample
        self.var_sample = self.likelihood.var_sample
        self.dim = self.mu_sample.shape[0]

        # should every model also have a data? or atleast mean and variance? 
        # it will be an array because each sample will have exactly same feature vectors

        self.step_size = step_size      # stepsize can be vector as long as dimensions match
        
        if rng_key is None:
            rng_key = np.random.seed(np.random.randint(2**32))
        self.rng_key = rng_key                                
        
        if proposal is None:
            proposal = lambda mu_trial: norm(mu_trial,self.step_size*np.ones(self.dim)).rvs()
        self.proposal = proposal

        if prior is None:
            prior = lambda mu_trial: norm(self.mu_sample,self.var_sample).logpdf(mu_trial)
        self.prior = prior

    def step(self, mu_current, max_iter = 400): # x is subsequent init_positions
        accept = np.array([False])
        steps = 0
        if_1D = False
        while((steps < max_iter) and (not accept.any())):

            mu_proposal = self.proposal(mu_current)

            current_wrapper = {'mu_trial':mu_current}
            proposal_wrapper = {'mu_trial':mu_proposal}

            log_likelihood_current = self.likelihood_func(current_wrapper)
            log_likelihood_proposal = self.likelihood_func(proposal_wrapper)

            log_prior_current  = self.prior(mu_current)
            log_prior_proposal = self.prior(mu_proposal)

            p_current = log_likelihood_current + log_prior_current
            p_proposal = log_likelihood_proposal + log_prior_proposal 

            p_accept = np.exp(p_proposal - p_current)
            accept = np.random.rand(self.dim) < p_accept

            if accept.any():
                try:
                    mu_current[accept] = mu_proposal[accept]
                except IndexError:
                    mu_current = mu_proposal # if 1d scalar only
                    # posterior.append([np.array(np.copy(mu_current))])
                    if_1D = True
            steps += 1

        return np.copy(mu_current), accept.any(), if_1D
        

    def run(self, n_steps = 1000):
        mu_current = self.init_position['mu_trial']
        posterior = []
        posterior.append(np.copy(mu_current))
        acceptance_rate = 0
        if_accept = False

        for t in range(n_steps):
            mu_next, if_accept, if_1D = self.step(mu_current)
            mu_current = mu_next

            if (if_accept):
                if (not if_1D):
                    posterior.append(np.copy(mu_current))
                else:
                    posterior.append(np.array([np.copy(mu_current)])) # if 1d scalar only
                acceptance_rate +=1
        
        acceptance_rate /= n_steps
        return np.array(posterior), acceptance_rate