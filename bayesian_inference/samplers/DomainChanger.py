import jax
import jax.numpy as jnp

class DomainChanger:
    def __init__(self, ranges):
        self.ranges = ranges
    
    def transform_to_infinite(self, x, a, b):
        # Normalize x to [0, 1]
        normalized_x = (x - a) / (b - a)
    
        # Avoid division by zero or log of zero
        epsilon = 1e-5
        normalized_x = jnp.clip(normalized_x, epsilon, 1 - epsilon)
    
        # Transform to (-∞, +∞)
        infinite_x = jnp.log(normalized_x / (1 - normalized_x))
        return infinite_x

    def inverse_transform_from_infinite(self, y, a, b):
        # Transform from (-∞, +∞) to [0, 1]
        normalized_y = 1 / (1 + jnp.exp(-y))
    
        # Inverse normalize to [a, b]
        x = normalized_y * (b - a) + a
        return x

    def transform(self, x):
        keys = list(self.ranges.keys())
        new_x = {}
        for key in keys:
            if self.ranges[key] == 'infinite':
                new_x[key + '_transformed'] = x[key]
            else:
                new_x[key + '_transformed'] = self.transform_to_infinite(x[key], self.ranges[key][0], self.ranges[key][1])

        return new_x

    def inverse_transform(self, x):
        keys = list(self.ranges.keys())
        new_x = {}
        for key in keys:
            if self.ranges[key] == 'infinite':
                new_x[key] = x[key + '_transformed']
            else:
                new_x[key] = self.inverse_transform_from_infinite(x[key + '_transformed'], self.ranges[key][0], self.ranges[key][1])

        return new_x

    def transform_in_place(self, x):
        keys = list(self.ranges.keys())
        for key in keys:
            if self.ranges[key] == 'infinite':
                x[key] = x[key]
            else:
                x[key] = self.transform_to_infinite(x[key], self.ranges[key][0], self.ranges[key][1])

        return x

    def inverse_transform_in_place(self, x):
        keys = list(self.ranges.keys())
        for key in keys:
            if self.ranges[key] == 'infinite':
                x[key] = x[key]
            else:
                x[key] = self.inverse_transform_from_infinite(x[key], self.ranges[key][0], self.ranges[key][1])

        return x
        