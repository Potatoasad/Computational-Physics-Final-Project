from .distribution import *

try:
	import jax.numpy as np
else:
	import numpy as np

class Normal(Distribution):
    """
    A class to represent a distribution
    ...

    Attributes
    ----------
    name: str
        stores the name of the distribution

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
    def __init__(self, name, mu, sigma):
        super().__init__(name)
        self.mu = mu
        self.sigma = sigma

    def logpdf(self, x):
        z = (x - self.mu)/self.sigma
        return -(z**2)/2 - np.log()

    def dlogpdf(self, x):
        raise NotImplementedError("This is the method from the abstract class, please inherit this class and implement dlogpdf(x)")


