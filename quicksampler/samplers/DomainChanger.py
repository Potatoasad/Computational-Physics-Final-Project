import jax
import jax.numpy as jnp
import numpy as np

class LogisticTransform:
    r"""Implementation of the Logistic Transformation between spaces

    If one wants to transform points using a logistic transformation from a
    closed interval :math:`[a,b]` to :math:`(-\infty, +\infty)`. The
    transformation is defined as:

    .. math::
        \begin{align*}
        f &: [a,b] \to (-\infty, \infty) \\
        f(x) &= \log\left(\frac{\frac{x-a}{b-a}}{1 - \frac{x-a}{b-a}}\right) \\
        f^{-1}(y) &= \frac{b-a}{1+e^{-y}} + a
        \end{align*}

    Attributes
    ----------
    epsilon : :obj:`float`, default = 1e-5
        Any input within :math:`\epsilon` of :math:`[a,b]` will be clipped onto the value 
        of the transformation at :math:`f(a + \epsilon)` or :math:`f(b - \epsilon)`
    backend : :obj:`str`, optional
        Will compute the transformation for JAX arrays if :obj:`'JAX'` and compute it for
        numpy arrays if :obj:`'numpy'`
    """

    def __init__(self, backend, epsilon=1e-5):
        """
        Initialize the LogisticTransform instance.

        Parameters
        ----------
        backend : str
            The backend for computation ('JAX' or 'numpy').
        epsilon : float, optional
            A small value for numerical stability, defaults to 1e-5.
        """
        self.backend = backend   # JAX or numpy
        if backend == 'JAX':
            self.jnp = jnp
        elif backend == 'numpy':
            self.jnp = np
        else:
            raise NotImplementedError(f"backend {backend} not recognized")

        self.epsilon = epsilon

    def transform_to_infinite(self, x, a, b):
        """Transforms from :math:`[a,b]` to :math:`(-\infty, +\infty)`

        Parameters
        ----------
        x : :obj:`array` or :obj:`float` of from either JAX or numpy
            The points to be transformed
        a : :obj:`array` or :obj:`float` of from either JAX or numpy
            The lower end of the bounded domain
        b : :obj:`array` or :obj:`float` of from either JAX or numpy
            The upper end of the bounded domain

        Returns
        -------
        y : :obj:`array` or :obj:`float` of size and type similar to x
            The transformed points that now live in the infinite domain
        """

        # Normalize x to [0, 1]
        normalized_x = (x - a) / (b - a)
    
        # Avoid division by zero or log of zero
        epsilon = self.epsilon
        normalized_x = self.jnp.clip(normalized_x, epsilon, 1 - epsilon)
    
        # Transform to (-∞, +∞)
        infinite_x = self.jnp.log(normalized_x / (1 - normalized_x))
        return infinite_x

    def inverse_transform_from_infinite(self, y, a, b):
        r"""Inverse transforms from :math:`(-\infty, +\infty)` to :math:`[a,b]`

        Parameters
        ----------
        y : :obj:`array` or :obj:`float` from either JAX or numpy
            The points in the infinite domain to be transformed back.
        a : :obj:`array` or :obj:`float` from either JAX or numpy
            The lower end of the bounded domain
        b : :obj:`array` or :obj:`float` from either JAX or numpy
            The upper end of the bounded domain

        Returns
        -------
        x : :obj:`array` or :obj:`float` of size and type similar to y
            The transformed points back into the closed interval [a, b].
        """

        # Transform from (-∞, +∞) to [0, 1]
        normalized_y = 1 / (1 + self.jnp.exp(-y))
        # Inverse normalize to [a, b]
        x = normalized_y * (b - a) + a
        return x

    def log_jacobian_determinant(self, variable_range, y):
        """Compute the logarithm of the Jacobian determinant of the transformation.

        Parameters
        ----------
        variable_range : str or list
            If a string, should be 'infinite' or a domain range represented as [a, b].
            If a list, it represents the domain range [a, b].
        y : :obj:`array` or :obj:`float` from either JAX or numpy
            The points in the infinite domain.

        Returns
        -------
        log_det : float
            The logarithm of the Jacobian determinant of the transformation.
        """
        if type(variable_range) == type([]):
            ## Is a domain range
            a,b = variable_range
            logistic_value = 1 / (1 + self.jnp.exp(-y))
            return self.jnp.log(b - a) + self.jnp.log(logistic_value) + self.jnp.log(1-logistic_value)
        else:
            if variable_range == 'infinite':
                return 0.0
            else:
                raise NotImplementedError(f'Do not understand the range {variable_range}')



class DomainChanger(LogisticTransform):
    r"""
    A class for transforming variables between different bounded domains using the Logistic Transformation.

    If one wants to transform points using a logistic transformation from a
    closed interval :math:`[a,b]` to :math:`(-\infty, +\infty)`. The
    transformation is defined as:

    .. math::
        \begin{align*}
        f &: [a,b] \to (-\infty, \infty) \\
        f(x) &= \log\left(\frac{\frac{x-a}{b-a}}{1 - \frac{x-a}{b-a}}\right) \\
        f^{-1}(y) &= \frac{b-a}{1+e^{-y}} + a
        \end{align*}

    Parameters
    ----------
    ranges : dict
        A dictionary specifying the variable ranges, where keys represent variable names and values
        can be either 'infinite' for unbounded variables or a list :obj:`[a, b]` representing the closed interval :obj`[a, b]`.
    backend : str, optional
        The backend for computation ('JAX' or 'numpy'). Defaults to 'numpy'.
    epsilon : float, optional
        A small value for numerical stability, defaults to 1e-5.
    """

    def __init__(self, ranges, backend='numpy', epsilon=1e-5):
        r"""
        Initialize the DomainChanger instance.

        Parameters
        ----------
        ranges : dict
            A dictionary specifying the variable ranges.
        backend : str, optional
            The backend for computation ('JAX' or 'numpy'). Defaults to 'numpy'.
        epsilon : float, optional
            A small value for numerical stability, defaults to 1e-5.
        """
        super().__init__(backend=backend, epsilon=epsilon)
        self.ranges = ranges
        self.transforms = None
        self.inverse_transforms = None

    def compute_transforms(self):
        """
        Compute the transformations and inverse transformations for each variable range.
        """
        if (self.transforms is None) or (self.inverse_transforms is None):
            self.transforms = {key : (lambda x, key=key: self.transform_to_infinite_from_range(x, self.ranges[key])) for key in self.ranges.keys()}
            self.inverse_transforms = {key : (lambda x, key=key: self.inverse_transform_from_infinite_from_range(x, self.ranges[key])) for key in self.ranges.keys()}

    def transform_to_infinite_from_range(self, x, ranges):
        r"""
        Transform variables from a specified range to (-∞, +∞).

        Parameters
        ----------
        x : float or array-like
            The input variable(s) to be transformed.
        ranges : str or list
            The specified variable range.

        Returns
        -------
        transformed_x : float or array-like
            The transformed variable(s).
        """
        if ranges == 'infinite':
            return x
        else:
            return self.transform_to_infinite(x, ranges[0], ranges[1])

    def inverse_transform_from_infinite_from_range(self, x, ranges):
        """
        Inverse transform variables from (-∞, +∞) to a specified range.

        Parameters
        ----------
        x : float or array-like
            The input variable(s) in the infinite domain.
        ranges : str or list
            The specified variable range.

        Returns
        -------
        transformed_x : float or array-like
            The inverse-transformed variable(s).
        """
        if ranges == 'infinite':
            return x
        else:
            return self.inverse_transform_from_infinite(x, ranges[0], ranges[1])


    def transform(self, x, suffix = ''):
        """
        Transform variables in the given dictionary to the infinite domain.

        Parameters
        ----------
        x : dict
            A dictionary containing variable names as keys and their corresponding values.
        suffix : str, optional
            A suffix to append to the transformed variable names. Defaults to an empty string.

        Returns
        -------
        new_x : dict
            A dictionary containing transformed variables with updated names.
        """
        self.compute_transforms()
        keys = list(self.ranges.keys())
        new_x = {}
        for key in keys:
            if self.ranges[key] == 'infinite':
                new_x[key + suffix] = x[key]
            else:
                new_x[key + suffix] = self.transforms[key](x[key])

        return new_x

    def inverse_transform(self, x, suffix=''):
        """
        Inverse transform variables in the given dictionary from the infinite domain.

        Parameters
        ----------
        x : dict
            A dictionary containing variable names as keys and their corresponding transformed values.
        suffix : str, optional
            A suffix to append to the original variable names. Defaults to an empty string.

        Returns
        -------
        new_x : dict
            A dictionary containing inverse transformed variables with original names.
        """
        self.compute_transforms()
        keys = list(self.ranges.keys())
        new_x = {}
        for key in keys:
            if self.ranges[key] == 'infinite':
                new_x[key] = x[key + suffix]
            else:
                new_x[key] = self.inverse_transforms[key](x[key + suffix])

        return new_x

    def inverse_log_jacobian(self, x):
        """
        Compute the sum of the logarithm of Jacobian determinants for each variable.

        Parameters
        ----------
        x : dict
            A dictionary containing variable names as keys and their corresponding transformed values.

        Returns
        -------
        log_det : float
            The sum of the logarithm of Jacobian determinants.
        """
        return self.jnp.sum(self.jnp.array([jnp.sum(self.log_jacobian_determinant(self.ranges[key], x[key])) for key in x.keys()]))

    def logprob_wrapped(self, logprob_func):
        """
        Create a wrapped likelihood function that includes the inverse log Jacobian term.

        Parameters
        ----------
        logprob_func : function
            A log probability function that takes a dictionary of variables as input.

        Returns
        -------
        likelihood_func : function
            A wrapped likelihood function that includes the inverse log Jacobian term.
        """
        def likelihood_func(y):
            x = self.inverse_transform(y)
            return  logprob_func(x) + self.inverse_log_jacobian(y)
        return likelihood_func

        