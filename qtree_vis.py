import numpy as np
import matplotlib.pyplot as plt
from core import insert_particle
from structures import FlatQTree
import time

generator = np.random.default_rng(42)
particles = generator.random(size=(2000,2))  # 10 random 2D particles
masses = np.random.lognormal(size=(2000,))  # random masses
k = 5.0
radii = np.power(masses, 1/3) * k  # radii proportional to cube root of mass

plt.scatter(particles[:,0], particles[:,1], s=1, color='blue')

qtree = FlatQTree(max_nodes=2000)
s = time.time()
qtree.reset(0.5, 0.5, 1.0)
for i in range(particles.shape[0]):
    insert_particle(qtree, particles[:,0], particles[:,1], i)
print("Tree build time:", time.time() - s)
for node in range(qtree.node_count):
    x, y, size = qtree.bbox[node]
    rect = plt.Rectangle((x-size/2, y-size/2), size, size, fill=False, edgecolor='r', linewidth=0.5)
    plt.gca().add_patch(rect)

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.bbox_inches='tight'
plt.show()