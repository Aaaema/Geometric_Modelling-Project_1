"""Half-edge data structure for triangle meshes.

Every triangle (a, b, c) is stored as three HALF-EDGES going round it:
    a -> b,  b -> c,  c -> a        (counter-clockwise, seen from outside)

For every half-edge h we keep four numbers (all in numpy arrays, O(1) lookup):
    origin[h]  vertex the half-edge starts at
    next[h]    next half-edge in the same triangle
    twin[h]    the same edge going the other way, in the neighbouring triangle
               (-1 if there is no neighbour -> boundary edge)
    face[h]    triangle the half-edge belongs to

and for every vertex v:
    out[v]     one half-edge that starts at v (a boundary one if v is on the boundary)

        c
       / ^
      v   \\        h = a -> b          next[h] = b -> c
     a ---> b       twin[h] = b -> a   (in the triangle on the other side)

Walking around a vertex (its "one-ring") only needs next and twin:
    h  ->  twin[prev[h]]    gives the next half-edge leaving the same vertex.
"""
import numpy as np


class HalfEdgeMesh:
    def __init__(self, n_vertices, faces):
        faces = [tuple(int(x) for x in f) for f in faces]
        self.faces = faces
        m = len(faces)

        # ---- 1. three half-edges per triangle: half-edge 3*f + i ----
        self.origin = np.empty(3 * m, dtype=int)
        self.next = np.empty(3 * m, dtype=int)
        self.face = np.repeat(np.arange(m), 3)
        lookup = {}                                   # (from, to) -> half-edge
        for f, tri in enumerate(faces):
            for i in range(3):
                h = 3 * f + i
                a, b = tri[i], tri[(i + 1) % 3]
                if (a, b) in lookup:
                    raise ValueError("Mesh is not manifold or not consistently oriented "
                                     f"(edge {a}->{b} used twice).")
                lookup[(a, b)] = h
                self.origin[h] = a
                self.next[h] = 3 * f + (i + 1) % 3
        self.lookup = lookup

        # ---- 2. twins: half-edge a->b is the twin of b->a ----
        self.twin = np.array([lookup.get((self.origin[self.next[h]], self.origin[h]), -1)
                              for h in range(3 * m)], dtype=int)

        # ---- 3. one outgoing half-edge per vertex (prefer a boundary one) ----
        self.out = np.full(n_vertices, -1, dtype=int)
        for h in range(3 * m):
            v = self.origin[h]
            if self.out[v] == -1 or self.twin[h] == -1:
                self.out[v] = h

        # ---- 4. cached one-rings (ordered neighbours) - used by every scheme ----
        self._rings = [self._walk_ring(v) for v in range(n_vertices)]

    # ---------------- basic moves ----------------
    def dest(self, h):
        return self.origin[self.next[h]]

    def prev(self, h):
        return self.next[self.next[h]]                # triangles: prev = next twice

    # ---------------- one-ring of a vertex ----------------
    def _walk_ring(self, v):
        """Neighbours of v in counter-clockwise order + whether v is on the boundary."""
        start = self.out[v]
        if start == -1:
            return [], False                          # vertex not used by any triangle
        ring, h = [], start
        while True:
            ring.append(self.dest(h))
            back = self.twin[self.prev(h)]
            if back == -1:                            # hit the boundary
                ring.append(self.origin[self.prev(h)])
                return ring, True
            h = back
            if h == start:                            # went all the way round
                return ring, False

    def neighbours(self, v):
        return self._rings[v][0]

    def valence(self, v):
        return len(self._rings[v][0])

    def on_boundary(self, v):
        return self._rings[v][1]

    def boundary_neighbours(self, v):
        """The two neighbours along the boundary (first and last of the open ring)."""
        ring, boundary = self._rings[v]
        return [ring[0], ring[-1]] if boundary else []

    def ring(self, v, start):
        """Closed ring of an interior vertex v, rotated so ring[0] = start (None at boundary)."""
        ring, boundary = self._rings[v]
        if boundary or start not in ring:
            return None
        i = ring.index(start)
        return ring[i:] + ring[:i]

    # ---------------- edges ----------------
    def edges(self):
        """Every undirected edge once, as (half-edge, a, b)."""
        for h in range(len(self.origin)):
            t = self.twin[h]
            if t == -1 or h < t:
                yield h, self.origin[h], self.dest(h)

    def half_edge(self, a, b):
        """Half-edge for edge {a, b} (either direction)."""
        h = self.lookup.get((a, b))
        return h if h is not None else self.lookup[(b, a)]

    def is_boundary_edge(self, a, b):
        h = self.half_edge(a, b)
        return self.twin[h] == -1

    def opposite(self, a, b):
        """Vertex opposite edge {a, b} in each triangle touching it (1 or 2 of them)."""
        h = self.half_edge(a, b)
        result = [self.origin[self.prev(h)]]
        if self.twin[h] != -1:
            result.append(self.origin[self.prev(self.twin[h])])
        return result
