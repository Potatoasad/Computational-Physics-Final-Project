try:
    import jax.numpy as np
    from jax import grad
    JAX = True
except:
    import numpy as np
    JAX = False


class Distribution:
    """
    A class to represent a distribution
    ...

    Attributes
    ----------
    name: str
        stores the name of the distribution

    Methods
    -------
    pdf(x) : vector -> float
        returns the probability density function (PDF) at the point x

    logpdf(x) vector -> float: 
        returns the log of the probability density function (PDF) at point x

    dlogpdf(x) jax array -> float:
        Note this function is computed automatically if you run compute_grad()
        returns the log of the probability density function (PDF) at point x 
        along with the vector gradient of the PDF.
    """
    def __init__(self, name):
        self.name = name
 
    def logpdf(self, x):
        raise NotImplementedError("This is the method from the abstract class, please inherit this class and implement logpdf(x)")

    def compute_grad(self):
        if JAX:
            self.dlogpdf = grad(self.logpdf)
        else:
            raise NotImplementedError("This method is not implemented when JAX is not available")


    def pdf(self, x):
        return np.exp(self.logpdf(x))

