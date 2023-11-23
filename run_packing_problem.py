from packing_problem import *
import time as t
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

hsp = HardSpherePacking(number_balls=512, relative_ball_diameter=0.1, dimensions=3, hyperparameter=0, number_samples=16)
start = t.time()
hsp.fit()
print(f"Time elapsed: {np.ceil(1e3 * (t.time() - start))} ms.")
print(f"Acceptance rate: {np.round(hsp.acceptance_rate, 6)}")
print(hsp.coordinates.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
def plot_sphere(ax, center, radius):
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 10)
    x = radius * np.outer(np.sin(theta), np.cos(phi))
    y = radius * np.outer(np.sin(theta), np.sin(phi))
    z = radius * np.outer(np.cos(theta), np.ones(len(phi)))
    for i in range(len(center)):
        ax.plot_surface(x + center[i, 0], y + center[i, 1], z + center[i, 2])
plot_sphere(ax, hsp.coordinates, hsp.ball_radius)
plt.show()
print(hsp.coordinates[-1])