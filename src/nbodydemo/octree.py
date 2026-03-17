import numpy as np

class OctreeNode:
    def __init__(self, center, size):
        self.center = np.array(center, dtype=np.float64)
        self.size = size
        self.mass = 0.0
        self.com = np.zeros(3, dtype=np.float64)
        self.children = [None] * 8
        self.is_leaf = True
        self.body = None

    def _get_octant(self, position):
        octant_index = 0
        if position[0] >= self.center[0]: octant_index |= 1
        if position[1] >= self.center[1]: octant_index |= 2
        if position[2] >= self.center[2]: octant_index |= 4
        return octant_index

    def _subdivide(self):
        quarter = self.size / 4.0
        half_size = self.size / 2.0
        for octant_index in range(8):
            x_offset = quarter if (octant_index & 1) else -quarter
            y_offset = quarter if (octant_index & 2) else -quarter
            z_offset = quarter if (octant_index & 4) else -quarter
            child_center = self.center + np.array([x_offset, y_offset, z_offset], dtype=np.float64)
            self.children[octant_index] = OctreeNode(child_center, half_size)

    def _insert_to_child(self, body):
        octant_index = self._get_octant(body.pos)
        self.children[octant_index].insert(body)

    def insert(self, body):
        if self.mass == 0.0:
            self.body = body
            self.mass = body.mass
            self.com = body.pos.copy()
            return

        if self.is_leaf and self.body is not None:
            self._subdivide()
            old_body = self.body
            self.body = None
            self.is_leaf = False
            self._insert_to_child(old_body)

        if not self.is_leaf:
            self._insert_to_child(body)

        old_mass = self.mass
        self.mass += body.mass
        self.com = (self.com * old_mass + body.pos * body.mass) / self.mass

    def compute_force(self, body, theta, G):
        if self.mass == 0.0 or (self.is_leaf and self.body is body):
            return np.zeros(3, dtype=np.float64)

        com_to_body_vector = self.com - body.pos
        distance_squared = np.dot(com_to_body_vector, com_to_body_vector)
        if distance_squared < 1e6:
            return np.zeros(3, dtype=np.float64)
        distance = np.sqrt(distance_squared)

        if self.is_leaf or (self.size / distance < theta):
            inverse_distance_cubed = 1.0 / (distance_squared * distance)
            return G * self.mass * com_to_body_vector * inverse_distance_cubed
        else:
            total_acceleration = np.zeros(3, dtype=np.float64)
            for child in self.children:
                if child is not None:
                    total_acceleration += child.compute_force(body, theta, G)
            return total_acceleration