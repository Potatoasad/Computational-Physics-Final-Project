from .distribution import *

try:
	import jax.numpy as np
	from jax import grad
	JAX = True
except:
	import numpy as np
	JAX = False

class Normal(Distribution):
    """
    A class to represent a distribution
    ...

    Attributes
    ----------
    mu: float
        stores the location of the distirbution

    sigma: float
        stores the standard deviationo of the distirbution

    Methods
    -------
    pdf(x) : vector -> float
        returns the probability density function (PDF) at the point x

    logpdf(x) vector -> float: 
        returns the log of the probability density function (PDF) at point x

    dlogpdf(x) jax array -> float:
        returns the log of the probability density function (PDF) at point x 
        along with the vector gradient of the PDF.
    """
    def __init__(self, mu, sigma):
        super().__init__("Normal")
        self.mu = mu
        self.sigma = sigma

    def logpdf(self, x):
        z = (x - self.mu)/self.sigma
        return -(z**2)/2 - np.log(2*np.pi*self.sigma)/2

    def dlogpdf(self, x):
        if JAX:
            return self.logpdf(x), grad(self.logpdf)(x)
        else:
            raise NotImplementedError("No JAX to autodiff, and we haven't implemented this part")


