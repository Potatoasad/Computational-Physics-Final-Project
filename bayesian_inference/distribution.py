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
