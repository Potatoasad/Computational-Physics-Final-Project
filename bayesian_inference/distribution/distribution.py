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
        returns the log of the probability density function (PDF) at point x 
        along with the vector gradient of the PDF.
    """
    def __init__(self, name):
        self.name = name

    def pdf(self, x):
        raise NotImplementedError("This is the method from the abstract class, please inherit this class and implement pdf(x)")
 
    def logpdf(self, x):
        raise NotImplementedError("This is the method from the abstract class, please inherit this class and implement logpdf(x)")

    def dlogpdf(self, x):
        raise NotImplementedError("This is the method from the abstract class, please inherit this class and implement dlogpdf(x)")


