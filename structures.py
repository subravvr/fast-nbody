import numpy as np
from numba.experimental import jitclass
from numba import float64, int64
from core import insert_particle, aggregate_particles, compute_all_forces

spec = [
    ('child_pointer', int64[:,:]),
    ('leaf_particle_indices', int64[:]),
    ('com_x', float64[:]),
    ('com_y', float64[:]),
    ('total_mass', float64[:]),
    ('bbox', float64[:,:]),
    ('node_count', int64),
    ('max_nodes', int64)]

@jitclass(spec)
class FlatQTree:
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes * 4  # Allow for subdivision
        self.child_pointer = -1 * np.ones((self.max_nodes, 4), dtype=np.int64)  # 4 children per node
        self.leaf_particle_indices = -1 * np.ones(self.max_nodes, dtype=np.int64) # 1 particle per leaf (complete subdivision.)
        self.bbox = np.zeros((self.max_nodes, 3), dtype=np.float64)  # xc, yc, size
        self.com_x = np.zeros(self.max_nodes, dtype=np.float64)
        self.com_y = np.zeros(self.max_nodes, dtype=np.float64)
        self.total_mass = np.zeros(self.max_nodes, dtype=np.float64)
        self.node_count = 0
        

    def reset(self, root_x, root_y, root_size):
        """Clears the tree and initializes the root node (index 0)."""
        
        # Reset the allocation cursor
        self.node_count = 1
        
        # Clear root node data (index 0)
        self.child_pointer[0, :] = int64(-1)
        self.leaf_particle_indices[0] = int64(-1)
        
        # Set the bounding box for the root node
        self.bbox[0, 0] = root_x
        self.bbox[0, 1] = root_y
        self.bbox[0, 2] = root_size

class Orchestrator:
    def __init__(self, particleVector, massVector, G=1.0, theta=0.5, softening=0.01):
        self.particles = particleVector
        self.masses = massVector
        self.G = G
        self.theta = theta
        self.softening = softening
        self.qtree = FlatQTree(max_nodes=len(particleVector))
        
    
    def build_tree(self):
        self.qtree.reset(0.5, 0.5, 1.0)
        for i in range(self.particles.shape[0]):
            insert_particle(self.qtree, self.particles[:,0], self.particles[:,1], i)
        
    def aggregate(self):
        for node in range(self.qtree.node_count):
            com_x, com_y, total_mass = aggregate_particles(self.qtree, self.particles[:,0], self.particles[:,1], self.masses, node)
            self.qtree.com_x[node] = com_x
            self.qtree.com_y[node] = com_y
            self.qtree.total_mass[node] = total_mass
    
    def compute_forces(self):
        fx_arr, fy_arr = compute_all_forces(self.qtree, self.particles[:,0], self.particles[:,1], self.masses,
                                           G=self.G, theta=self.theta, softening=self.softening)
        return fx_arr, fy_arr

    def integrate(self, dt):
        fx_arr, fy_arr = self.compute_forces()
        # Simple Euler integration
        self.particles[:,0] += fx_arr * dt / self.masses
        self.particles[:,1] += fy_arr * dt / self.masses
        return self.particles