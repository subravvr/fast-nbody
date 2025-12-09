import numpy as np
import matplotlib.pyplot as plt
from structures import FlatQTree, Orchestrator
from numba import njit, int64, float64
import time

generator = np.random.default_rng(42)
particles = generator.random(size=(50000,2))  # 1000000 random 2D particles
masses = np.random.lognormal(size=(50000,))  # random masses
k = 5.0
radii = np.power(masses, 1/3) * k  # radii proportional to cube root of mass
tsteps = 100
orchestrator = Orchestrator(particles, masses, G=1.0, theta=0.5, softening=0.01)

bavg, aavg, cavg = 0.0, 0.0, 0.0
for t in range(tsteps):
    s1 = time.time()
    orchestrator.build_tree()
    e1 = time.time()
    
    orchestrator.aggregate()
    e2 = time.time()


    particles = orchestrator.integrate(dt=0.01)
    e3 = time.time()

    if t>0:
        bavg += e1 - s1
        aavg += e2 - e1
        cavg += e3 - e2
print(f"Average build time: {bavg / (tsteps-1):.6f} s")
print(f"Average aggregate time: {aavg / (tsteps-1):.6f} s")
print(f"Average compute+integrate time: {cavg / (tsteps-1):.6f} s")
print(f"Total average time per step: {(bavg + aavg + cavg) / (tsteps-1):.6f} s")

