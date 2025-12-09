A fully parallel implementation of the Barnes-Hut algorithm for 2D N-body simulation.
Includes the following:
- Statically sized quadtree with arrays for child pointers
- Fast insertion / aggregation method through pointer usage
- Parallelized nodal force calculation and update

![Example Image](./images/qtree2000.png)

I got to roughly ~1ms / timestep for 1 million particles on a Macbook Air 2025 with an M4 and 16GB of RAM. Complexity is following the expected $\mathcal{O}(n\log n)$.

Next updates
- Post-simulation rendering. Right now, this is the easiest sanity check for the time integration.
- Real-time renderer. this proved to be nontrivial (I don't know OpenGL or other ways for super-fast rendering, so this will be a longer term project).
- Collision / merging. With 1000000 particles, the limit for cell size is definitely exceeded. Need to handle this better.

Long term updates
- Linking this with a fast attention-based ML model + a correction scheme (filtration) could be interesting to probabilistically predict local densities.


