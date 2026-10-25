"""Subdivision schemes for triangle meshes (weights from Zorin, "Subdivision Zoo").

A mesh is two numpy arrays:
    V - vertices,  shape (n, 3)
    F - triangles, shape (m, 3)  (vertex indices, counter-clockwise from outside)

Each scheme is one function:  new_V, new_F = scheme(V, F)
"""
import math
import numpy as np


# =====================================================================
#  Helper: neighbour information (who is next to whom)
# =====================================================================
def edge_key(a, b):
    """Same key for edge (a, b) and (b, a)."""
    return (a, b) if a < b else (b, a)


class Topology:
    def __init__(self, n_vertices, faces):
        self.faces = faces
        self.edge_faces = {}                                   # edge -> faces using it
        self.neighbours = [set() for _ in range(n_vertices)]   # vertex -> neighbour vertices
        self.vertex_faces = [[] for _ in range(n_vertices)]    # vertex -> faces using it
        self.boundary_neighbours = [[] for _ in range(n_vertices)]
        self.on_boundary = [False] * n_vertices

        for fi, (a, b, c) in enumerate(faces):
            for x, y in ((a, b), (b, c), (c, a)):
                self.edge_faces.setdefault(edge_key(x, y), []).append(fi)
                self.neighbours[x].add(y)
                self.neighbours[y].add(x)
                self.vertex_faces[x].append(fi)

        # an edge used by only ONE triangle is a boundary edge
        for (a, b), fs in self.edge_faces.items():
            if len(fs) == 1:
                self.on_boundary[a] = self.on_boundary[b] = True
                self.boundary_neighbours[a].append(b)
                self.boundary_neighbours[b].append(a)

    def valence(self, v):
        return len(self.neighbours[v])

    def is_boundary_edge(self, a, b):
        return len(self.edge_faces[edge_key(a, b)]) == 1

    def opposite(self, a, b):
        """Vertices opposite edge (a, b): one per triangle on that edge."""
        result = []
        for f in self.edge_faces.get(edge_key(a, b), []):
            for v in self.faces[f]:
                if v != a and v != b:
                    result.append(v)
        return result

    def ring(self, v, start):
        """Neighbours of v in order around it, starting at 'start' (None at boundary)."""
        nxt = {}
        for f in self.vertex_faces[v]:
            t = self.faces[f]
            i = t.index(v)
            nxt[t[(i + 1) % 3]] = t[(i + 2) % 3]
        ring, current = [], start
        for _ in range(self.valence(v)):
            if current is None:
                return None
            ring.append(current)
            current = nxt.get(current)
        return ring if current == start else None


def split_into_four(faces, edge_vertex):
    """Each triangle becomes 4 (used by Loop and Butterfly).

             c
            / \\
          ca---bc
          / \\ / \\
         a---ab--b
    """
    out = []
    for a, b, c in faces:
        ab = edge_vertex[edge_key(a, b)]
        bc = edge_vertex[edge_key(b, c)]
        ca = edge_vertex[edge_key(c, a)]
        out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return out


# =====================================================================
#  1) LOOP  (approximating - old points move, surface gets very smooth)
# =====================================================================
def loop(V, F):
    faces = F.tolist()
    topo = Topology(len(V), faces)
    new_V = []

    # Step 1: move every OLD vertex
    #   interior:  (1 - k*beta) * v + beta * (sum of k neighbours),  beta = 3/(8k)  (3/16 if k = 3)
    #   boundary:  3/4 * v + 1/8 * (each of the two boundary neighbours)
    for v in range(len(V)):
        if topo.on_boundary[v]:
            bn = topo.boundary_neighbours[v]
            if len(bn) == 2:
                new_V.append(3 / 4 * V[v] + 1 / 8 * (V[bn[0]] + V[bn[1]]))
            else:
                new_V.append(V[v].copy())
        else:
            k = topo.valence(v)
            beta = 3 / 16 if k == 3 else 3 / (8 * k)
            new_V.append((1 - k * beta) * V[v] + beta * V[list(topo.neighbours[v])].sum(axis=0))

    # Step 2: add a NEW vertex on every edge
    #   interior edge:  3/8 * (a + b) + 1/8 * (c + d)      c, d = opposite vertices
    #   boundary edge:  1/2 * (a + b)
    edge_vertex = {}
    for (a, b) in topo.edge_faces:
        edge_vertex[(a, b)] = len(new_V)
        if topo.is_boundary_edge(a, b):
            new_V.append((V[a] + V[b]) / 2)
        else:
            c, d = topo.opposite(a, b)
            new_V.append(3 / 8 * (V[a] + V[b]) + 1 / 8 * (V[c] + V[d]))

    # Step 3: split every triangle into 4
    return np.array(new_V), np.array(split_into_four(faces, edge_vertex))


# =====================================================================
#  2) MODIFIED BUTTERFLY  (interpolating - old points never move)
# =====================================================================
def butterfly(V, F):
    faces = F.tolist()
    topo = Topology(len(V), faces)

    # Step 1: old vertices stay exactly where they are
    new_V = [p.copy() for p in V]

    # Step 2: add a new vertex on every edge
    edge_vertex = {}
    for (a, b) in topo.edge_faces:
        edge_vertex[(a, b)] = len(new_V)
        new_V.append(butterfly_edge_point(V, topo, a, b))

    # Step 3: split every triangle into 4
    return np.array(new_V), np.array(split_into_four(faces, edge_vertex))


def butterfly_edge_point(V, topo, a, b):
    # Boundary edge -> 4-point rule:  -1/16, 9/16, 9/16, -1/16
    if topo.is_boundary_edge(a, b):
        a0 = [u for u in topo.boundary_neighbours[a] if u != b]
        b1 = [u for u in topo.boundary_neighbours[b] if u != a]
        if not a0 or not b1:
            return (V[a] + V[b]) / 2
        return 9 / 16 * (V[a] + V[b]) - 1 / 16 * (V[a0[0]] + V[b1[0]])

    # "Extraordinary" = interior vertex whose valence is not 6
    def extraordinary(v):
        return not topo.on_boundary[v] and topo.valence(v) != 6

    pa = special_rule(V, topo, a, b) if extraordinary(a) else None
    pb = special_rule(V, topo, b, a) if extraordinary(b) else None
    if pa is not None and pb is not None:
        return (pa + pb) / 2          # both ends extraordinary -> average
    if pa is not None:
        return pa
    if pb is not None:
        return pb
    return standard_rule(V, topo, a, b)


def standard_rule(V, topo, a, b):
    """8-point butterfly:  1/2 (a + b) + 1/8 (c + d) - 1/16 (four wing points)"""
    c, d = topo.opposite(a, b)

    def wing(x, y, away):
        # vertex on the other side of edge (x, y); if missing (boundary) -> reflect 'away'
        for w in topo.opposite(x, y):
            if w != away:
                return V[w]
        return V[x] + V[y] - V[away]

    wings = wing(a, c, b) + wing(b, c, a) + wing(a, d, b) + wing(b, d, a)
    return 1 / 2 * (V[a] + V[b]) + 1 / 8 * (V[c] + V[d]) - 1 / 16 * wings


def special_rule(V, topo, centre, other):
    """Edge next to a vertex of valence k != 6:  3/4 * centre + sum of s_j * ring_j"""
    ring = topo.ring(centre, other)       # ring[0] = other
    if ring is None:
        return None
    k = len(ring)
    if k == 3:
        s = [5 / 12, -1 / 12, -1 / 12]
    elif k == 4:
        s = [3 / 8, 0, -1 / 8, 0]
    else:
        s = [(1 / 4 + math.cos(2 * math.pi * j / k) + 1 / 2 * math.cos(4 * math.pi * j / k)) / k
             for j in range(k)]
    result = 3 / 4 * V[centre]
    for j in range(k):
        result = result + s[j] * V[ring[j]]
    return result


# =====================================================================
#  3) SQRT(3)  (Kobbelt) - extra scheme for teams of three
# =====================================================================
def sqrt3(V, F):
    faces = F.tolist()
    topo = Topology(len(V), faces)
    n = len(V)

    # Step 1: move old interior vertices:  (1 - alpha) v + alpha/k * sum(neighbours)
    #         alpha = (4 - 2 cos(2 pi / k)) / 9
    new_V = []
    for v in range(n):
        if topo.on_boundary[v]:
            new_V.append(V[v].copy())
        else:
            k = topo.valence(v)
            alpha = (4 - 2 * math.cos(2 * math.pi / k)) / 9
            new_V.append((1 - alpha) * V[v] + alpha / k * V[list(topo.neighbours[v])].sum(axis=0))

    # Step 2: add the centre point of every triangle (index n + face number)
    for a, b, c in faces:
        new_V.append((V[a] + V[b] + V[c]) / 3)

    # Step 3: "flip" every old edge - connect the two centre points instead
    new_F = []
    for (a, b), fs in topo.edge_faces.items():
        if len(fs) == 2:
            f1, f2 = fs
            if (a, b) not in directed_edges(faces[f1]):
                f1, f2 = f2, f1                       # f1 is the face with edge a -> b
            new_F += [(a, n + f2, n + f1), (b, n + f1, n + f2)]
        else:                                         # boundary edge: keep it
            f = fs[0]
            new_F.append((a, b, n + f) if (a, b) in directed_edges(faces[f]) else (b, a, n + f))

    return np.array(new_V), np.array(new_F)


def directed_edges(t):
    return [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]


# Schemes shown in the menu
SCHEMES = {"Loop": loop, "Butterfly": butterfly, "Sqrt(3)": sqrt3}
