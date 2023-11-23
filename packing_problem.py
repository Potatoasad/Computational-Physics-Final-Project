import numpy as np
import scipy.sparse as ss

BOX_EDGE_LENGTH =                   1.0     # sets the length scale
DEFAULT_RELATIVE_BALL_DIAMETER =    0.02    # of BOX_EDGE_LENGTH
DEFAULT_RELATIVE_MAX_ADJUSTMENT =   0.02    # of BOX_EDGE_LENGTH
DEFAULT_HYPERPARAMETER =            0.1     # inverse temperature
DEFAULT_STORE_HISTORY =             False
DEFAULT_NUMBER_SAMPLES =            1024    
RANDOM_SEED =                       10
np.random.seed(RANDOM_SEED)

class HardSpherePacking:
    def __init__(self, 
                 number_balls, 
                 dimensions, 
                 relative_ball_diameter=DEFAULT_RELATIVE_BALL_DIAMETER, 
                 relative_max_adjustment=DEFAULT_RELATIVE_MAX_ADJUSTMENT,
                 hyperparameter=DEFAULT_HYPERPARAMETER, 
                 store_history=DEFAULT_STORE_HISTORY, 
                 number_samples=DEFAULT_NUMBER_SAMPLES):

        self.number_balls =     number_balls
        self.dimensions =       dimensions
        self.box_edge_length =  BOX_EDGE_LENGTH
        self.ball_radius =      self.box_edge_length * relative_ball_diameter / 2
        self.max_adjustment =   self.box_edge_length * relative_max_adjustment
        self.coordinates =      self.ball_radius + (self.box_edge_length - self.ball_radius) * np.random.rand(self.number_balls, self.dimensions)
        self.hyperparameter =   hyperparameter
        self.store_history =    store_history
        self.number_samples =   number_samples
        self.acceptance_rate =  0

        if self.store_history:
            self.history = list()
            self.history.append(self.coordinates)

        self.A = np.zeros([int(self.number_balls * (self.number_balls - 1) / 2), self.number_balls])
        row = 0
        for i in range(self.number_balls - 1, 0, -1):
            self.A[row : row + i, self.number_balls - 1 - i] = np.ones(i)
            self.A[row : row + i, self.number_balls - i : self.number_balls] = -np.eye(i)
            row += i

    def all_distances(self, temporary_coordinates, squared=True):
        distances_squared = np.sum((self.A @ temporary_coordinates) ** 2, axis=1)
        
        if squared:
            return distances_squared
        else:
            return np.sqrt(distances_squared)

    def select_instances(self, temporary_coordinates, previous_energy):
        distances = self.all_distances(temporary_coordinates, squared=True)
        overlap = (distances - (2 * self.ball_radius)) ** 2
        
        if np.any(overlap < 0):
            return [0, previous_energy]
        else:
            energy = np.sum(distances)
            if previous_energy < energy:
                prob_accept = np.exp(-self.hyperparameter * (energy - previous_energy))
            else:
                prob_accept = 1
            return [np.random.choice([1, 0], p=[prob_accept, 1 - prob_accept]), energy]

    def fit(self):
        accepts = 0
        previous_energy = 0
        
        for i in range(self.number_samples):
            temporary_coordinates = np.mod(self.coordinates + self.max_adjustment * np.random.rand(self.number_balls, self.dimensions), 1)
            [to_accept, previous_energy] = self.select_instances(temporary_coordinates, previous_energy)
            
            if to_accept:
                self.coordinates = temporary_coordinates.copy()
                accepts += 1
        
        self.acceptance_rate = accepts / self.number_samples