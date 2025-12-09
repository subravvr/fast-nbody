import numpy as np
import matplotlib.pyplot as plt
from structures import FlatQTree, Orchestrator
from numba import njit, int64, float64
import time

generator = np.random.default_rng(42)
particles = generator.random(size=(1000000,2))  # 1000000 random 2D particles
masses = np.random.lognormal(size=(1000000,))  # random masses
k = 5.0
radii = np.power(masses, 1/3) * k  # radii proportional to cube root of mass
tsteps = 100
orchestrator = Orchestrator(particles, masses, G=1.0, theta=0.5, softening=0.01)

for _ in range(tsteps):
    s = time.time()
    particles = orchestrator.integrate(dt=0.01)
    e = time.time()
print("Average time per step:", (e - s)/tsteps)