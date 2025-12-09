import numpy as np
import matplotlib.pyplot as plt
from core import allocate_node, insert_particle, aggregate_particles, compute_all_forces
from structures import FlatQTree
from numba import njit, int64, float64

generator = np.random.default_rng(42)
particles = generator.random(size=(200,2))  # 10 random 2D particles
masses = np.random.lognormal(size=(200,))  # random masses
k = 5.0
radii = np.power(masses, 1/3) * k  # radii proportional to cube root of mass

plt.scatter(particles[:,0], particles[:,1], s=radii, color='blue')

qtree = FlatQTree(max_nodes=1024)
qtree.reset(0.5, 0.5, 1.0)
for i in range(particles.shape[0]):
    insert_particle(qtree, particles[:,0], particles[:,1], i)

for node in range(qtree.node_count):
    x, y, size = qtree.bbox[node]
    rect = plt.Rectangle((x-size/2, y-size/2), size, size, fill=False, edgecolor='red', linewidth=0.5)
    plt.gca().add_patch(rect)

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.gca().set_aspect('equal', adjustable='box')

for node in range(qtree.node_count):
    com_x, com_y, total_mass = aggregate_particles(qtree, particles[:,0], particles[:,1], masses, node)
    qtree.com_x[node] = com_x
    qtree.com_y[node] = com_y
    qtree.total_mass[node] = total_mass

fx_arr,fy_arr = compute_all_forces(qtree, particles[:,0], particles[:,1], masses,
                                   G=1.0, theta=0.5, softening=0.01)

for i in range(particles.shape[0]):
    plt.arrow(particles[i,0], particles[i,1], fx_arr[i]*1e-5, fy_arr[i]*1e-5,
              head_width=0.005, head_length=0.01, fc='green', ec='green')
    
plt.show()