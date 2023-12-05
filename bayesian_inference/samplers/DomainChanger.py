import jax
import jax.numpy as jnp
import numpy as np

class LogisticTransform:
    def __init__(self, backend, epsilon=1e-5):
        self.backend = backend   # JAX or numpy
        if backend == 'JAX':
            self.jnp = jnp
        elif backend == 'numpy':
            self.jnp = np
        else:
            raise NotImplementedError(f"backend {backend} not recognized")

        self.epsilon = epsilon
    def transform_to_infinite(self, x, a, b):
        # Normalize x to [0, 1]
        normalized_x = (x - a) / (b - a)
    
        # Avoid division by zero or log of zero
        epsilon = self.epsilon
        normalized_x = self.jnp.clip(normalized_x, epsilon, 1 - epsilon)
    
        # Transform to (-∞, +∞)
        infinite_x = self.jnp.log(normalized_x / (1 - normalized_x))
        return infinite_x

    def inverse_transform_from_infinite(self, y, a, b):
        # Transform from (-∞, +∞) to [0, 1]
        normalized_y = 1 / (1 + self.jnp.exp(-y))
    
        # Inverse normalize to [a, b]
        x = normalized_y * (b - a) + a
        return x

    def log_jacobian_determinant(self, variable_range, y):
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
    def __init__(self, ranges, backend='numpy', epsilon=1e-5):
        super().__init__(backend=backend, epsilon=epsilon)
        self.ranges = ranges
        self.transforms = None
        self.inverse_transforms = None

    def compute_transforms(self):
        if (self.transforms is None) or (self.inverse_transforms is None):
            self.transforms = {key : (lambda x: self.transform_to_infinite(x, self.ranges[key][0], self.ranges[key][1])) for key in self.ranges.keys()}
            self.inverse_transforms = {key : (lambda x: self.inverse_transform_from_infinite(x, self.ranges[key][0], self.ranges[key][1])) for key in self.ranges.keys()}

    def transform(self, x, suffix = ''):
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
        return self.jnp.sum(self.jnp.array([self.log_jacobian_determinant(self.ranges[key], x[key]) for key in x.keys()]))

    def logprob_wrapped(self, logprob_func):
        def likelihood_func(y):
            x = self.inverse_transform(y)
            return  logprob_func(x) + self.inverse_log_jacobian(y)
        return likelihood_func

        