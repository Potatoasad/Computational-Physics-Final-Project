import numpy as np
import scipy.sparse as ss
from scipy.spatial.distance import pdist

BOX_EDGE_LENGTH =                   1.0     # sets the length scale
DEFAULT_RELATIVE_BALL_DIAMETER =    0.02    # of BOX_EDGE_LENGTH
DEFAULT_RELATIVE_MAX_ADJUSTMENT =   0.02    # of BOX_EDGE_LENGTH
DEFAULT_DISTANCE_CUTOFF =           3       # of ball radii
DEFAULT_HYPERPARAMETER =            0.1     # inverse temperature

def augment_lattice(coordinates):
    repeated_box = list()
    repeated_box.append(coordinates.copy())
    shifts = [-1, 1]
    for i in shifts:
        for j in range(coordinates.shape[1]):
            temp = np.zeros(coordinates.shape)
            temp[:, j] = i
            repeated_box.append(coordinates.copy() + temp.copy())
    return np.vstack(repeated_box).copy()

class SoftSpherePacking:
    def __init__(self, 
                 number_balls, 
                 dimensions, 
                 relative_ball_diameter=DEFAULT_RELATIVE_BALL_DIAMETER, 
                 relative_distance_cutoff=DEFAULT_DISTANCE_CUTOFF,
                 hyperparameter=DEFAULT_HYPERPARAMETER):
        self.number_balls =     number_balls
        self.dimensions =       dimensions
        self.box_edge_length =  BOX_EDGE_LENGTH
        self.ball_radius =      self.box_edge_length * relative_ball_diameter / 2
        self.coordinates =      self.box_edge_length * np.random.rand(self.number_balls, self.dimensions)
        self.distance_cutoff =  relative_distance_cutoff * self.ball_radius
        self.hyperparameter =   hyperparameter

    def logpdf(self, configuration_data, periodic_boundary_conditions=False):
        self.coordinates = configuration_data["positions"].copy()
        if periodic_boundary_conditions:
            energy = 0
            for i in range(self.number_balls):
                ith_distances = pdist(np.vstack([self.coordinates[i], augment_lattice(np.delete(self.coordinates, i, axis=0))]))[:(2 ** self.dimensions + 1) * (self.number_balls - 1)]
                energy += - self.hyperparameter * np.sum(ith_distances < self.distance_cutoff)
            return energy
        else:
            energy = -self.hyperparameter * np.sum(pdist(self.coordinates) < self.distance_cutoff)
            return energy