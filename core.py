import numpy as np
from numba import njit, prange, float64, int64
# from structures import FlatQTree

# Helper functions
@njit(fastmath=True)
def dist2(x1: float64, y1: float64,
          x2: float64, y2: float64) -> float64:
    """Computes squared distance between two 2D points."""
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)

@njit
def allocate_node(tree,
                  bbox_x: float64, 
                  bbox_y: float64, 
                  bbox_size: float64) -> int64:
    """"
    Allocates a new node in the quadtree.
    Initializes the node's bounding box.
    Resets the node's child pointers and particle indices.
    Increments the tree's node cursor.

    Args:
        tree: FlatQTree instance
        bbox_x: float, x-coordinate of the bounding box's bottom-left corner
        bbox_y: float, y-coordinate of the bounding box's bottom-left corner
        bbox_size: float, size (width and height) of the bounding box
    Returns:
        idx_new: index of the newly allocated node
    """

    idx_new = tree.node_count
    if idx_new >= tree.max_nodes:
        raise RuntimeError("Exceeded maximum number of nodes in the quadtree.")
    
    # Claim new node, increment cursor
    tree.node_count += 1

    # Initialize child pointers to -1 (no children)
    tree.child_pointer[idx_new, :] = int64(-1)

    # Initialize particle indices to -1 (no particles)
    tree.leaf_particle_indices[idx_new] = int64(-1)

    # Set bounding box
    tree.bbox[idx_new, 0] = bbox_x
    tree.bbox[idx_new, 1] = bbox_y
    tree.bbox[idx_new, 2] = bbox_size

    return idx_new

@njit
def insert_particle(tree,
                    px_arr: np.ndarray,
                    py_arr: np.ndarray,
                    particle_idx: int64) -> None:
    """
    Inserts a particle into the quadtree.

    Args:
        tree: FlatQTree instance
        px_arr: x-coordinates of particles
        py_arr: y-coordinates of particles
        particle_idx: index of the particle to insert
    """

    px,py = px_arr[particle_idx], py_arr[particle_idx]
    node_idx = 0  # Start at root

    while True:
        # Get node bounding box
        node_x = tree.bbox[node_idx, 0]
        node_y = tree.bbox[node_idx, 1]
        node_size = tree.bbox[node_idx, 2]

        if node_size < 1e-10:
            if tree.leaf_particle_indices[node_idx] != -1:
                tree.leaf_particle_indices[node_idx] = particle_idx
                # Have to add aggregation here.
                return
            break
        
        # Check if leaf node (no children)
        if tree.child_pointer[node_idx, 0] == -1 and tree.leaf_particle_indices[node_idx] == -1:
                # Leaf is empty, insert particle here
                tree.leaf_particle_indices[node_idx] = particle_idx
                return
        elif tree.leaf_particle_indices[node_idx] != -1:
            # Leaf already has a particle, need to subdivide
            
            old_particle_idx = tree.leaf_particle_indices[node_idx] # Get existing particle

            tree.leaf_particle_indices[node_idx] = -1  # Clear leaf. This node is now internal.

            # Need to subdivide
            half_size = node_size / 2.0

            quadrants = [
                (node_x - half_size/2, node_y - half_size/2),  # SW
                (node_x + half_size/2, node_y - half_size/2),  # SE 
                (node_x - half_size/2, node_y + half_size/2),   # NW,
                (node_x + half_size/2, node_y + half_size/2)  # NE 
            ]


            # Allocate children
            for i, q in enumerate(quadrants):
                child_x, child_y = q
                child_idx = allocate_node(tree, child_x, child_y, half_size)
                tree.child_pointer[node_idx, i] = child_idx

            # Reinsert existing particles
            insert_particle(tree, px_arr, py_arr, old_particle_idx)


        # Determine which child to descend into
        quadrant = 0
        if px >= node_x:
            quadrant += 1
        if py >= node_y:
            quadrant += 2

        node_idx = tree.child_pointer[node_idx, quadrant]

@njit
def aggregate_particles(tree,
                        px_arr: np.ndarray,
                        py_arr: np.ndarray,
                        mass_arr: np.ndarray,
                        node_idx: int64) -> (float64, float64, float64):
    """
    Recursively aggregates particle properties in the quadtree.

    Args:
        tree: FlatQTree instance
        px_arr: x-coordinates of particles
        py_arr: y-coordinates of particles
        mass_arr: masses
        node_idx: index of the current node
    """

    # Base case - leaf node with a particle
    p_idx = tree.leaf_particle_indices[node_idx]
    if p_idx != -1:
        return px_arr[p_idx], py_arr[p_idx], mass_arr[p_idx]
    
    # Base case - empty leaf node
    if p_idx == -1 and tree.child_pointer[node_idx, 0] == -1:
        return 0.0, 0.0, 0.0
    
    # Recursive case - internal node
    total_mass = 0.0
    com_x = 0.0
    com_y = 0.0

    for i in range(4):
        child_idx = tree.child_pointer[node_idx, i]
        if child_idx != -1:
            cx, cy, cmass = aggregate_particles(tree, px_arr, py_arr, mass_arr, child_idx)
            total_mass += cmass
            com_x += cx * cmass
            com_y += cy * cmass
    
    if total_mass > 0.0:
        com_x /= total_mass
        com_y /= total_mass
    else:
        com_x = 0.0
        com_y = 0.0
    return com_x, com_y, total_mass


@njit(fastmath=True)
def compute_nodal_force(tree,
                        particle_idx: int64,
                        px_arr: float64,
                        py_arr: float64,
                        mass_arr: float64,
                        G: float64,
                        theta: float64,
                        softening: float64):
    """
    Recursively computes the gravitational force on a particle with tree traversal starting at the root node.

    Args:
        tree: FlatQTree instance
        particle_idx: index of the target particle
        px_arr: x-coordinates of particles
        py_arr: y-coordinates of particles
        mass_arr: masses
        G: gravitational constant
        theta: opening angle threshold
        softening: softening length to avoid singularities
    
    Returns:
        fx: x-component of the force
        fy: y-component of the force
    """
    stack = [0]  # Start with root node
    fx = 0.0
    fy = 0.0

    while stack:
        node_idx = stack.pop()

        # Get node bounding box
        node_x = tree.bbox[node_idx, 0]
        node_y = tree.bbox[node_idx, 1]
        node_size = tree.bbox[node_idx, 2]

        # Need center of mass here.
        com_x, com_y, total_mass = tree.com_x[node_idx], tree.com_y[node_idx], tree.total_mass[node_idx]

        if total_mass == 0.0:
            continue  # Empty node

        # Compute distance to center of mass
        dx, dy = com_x - px_arr[particle_idx], com_y - py_arr[particle_idx]
        d2 = dist2(px_arr[particle_idx], py_arr[particle_idx], com_x, com_y)
        r2 = d2 + softening * softening
        r = np.sqrt(r2)

        # Check opening criterion
        if (node_size / r) < theta or tree.child_pointer[node_idx, 0] == -1:
            # Treat as a single mass
            force_mag = G * mass_arr[particle_idx] * total_mass / r2
            fx += force_mag * dx / r
            fy += force_mag * dy / r
            continue
        
        elif tree.child_pointer[node_idx, 0] == -1 and tree.leaf_particle_indices[node_idx] != -1:
            # Leaf node with a single particle
            leaf_idx = tree.leaf_particle_indices[node_idx]
            if leaf_idx != particle_idx:
                target_mass = mass_arr[leaf_idx]
                dx, dy = px_arr[leaf_idx] - px_arr[particle_idx], py_arr[leaf_idx] - py_arr[particle_idx]
                d2 = dist2(px_arr[particle_idx], py_arr[particle_idx], px_arr[leaf_idx], py_arr[leaf_idx])
                r2 = d2 + softening * softening
                r = np.sqrt(r2)
                force_mag = G * mass_arr[particle_idx] * target_mass / r2
                fx += force_mag * dx / r
                fy += force_mag * dy / r
            continue
        
        else:
            # Internal node requiring recursion.
            for i in range(4):
                child_idx = tree.child_pointer[node_idx, i]
                if child_idx != -1:
                    stack.append(child_idx)
    return fx, fy

@njit(parallel=True, fastmath=True)
def compute_all_forces(tree,
                        px_arr: np.ndarray,
                        py_arr: np.ndarray,
                        mass_arr: np.ndarray,
                        G: float64,
                        theta: float64,
                        softening: float64) -> (np.ndarray, np.ndarray):
    """
    Computes gravitational forces on all particles using the quadtree.

    Args:
        tree: FlatQTree instance
        px_arr: x-coordinates of particles
        py_arr: y-coordinates of particles
        mass_arr: masses
        G: gravitational constant
        theta: opening angle threshold
        softening: softening length to avoid singularities
    
    Returns:
        fx_arr: x-components of the forces
        fy_arr: y-components of the forces
    """

    num_particles = px_arr.shape[0]
    fx_arr = np.zeros(num_particles, dtype=np.float64)
    fy_arr = np.zeros(num_particles, dtype=np.float64)

    for i in prange(num_particles):
        fx, fy = compute_nodal_force(tree, i, px_arr, py_arr, mass_arr, G, theta, softening)
        fx_arr[i] = fx
        fy_arr[i] = fy

    return fx_arr, fy_arr
