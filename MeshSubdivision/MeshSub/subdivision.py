"""Subdivision schemes for triangle meshes (weights from Zorin, "Subdivision Zoo").

A mesh is two numpy arrays:
    V - vertices,  shape (n, 3)
    F - triangles, shape (m, 3)  (vertex indices, counter-clockwise from outside)
Neighbour questions (who is next to whom) are answered by the half-edge
structure in halfedge.py.

Each scheme is one function:  new_V, new_F = scheme(V, F)
"""
import math
import numpy as np


from halfedge import HalfEdgeMesh


def edge_key(a, b):
    """Same key for edge (a, b) and (b, a)."""
    return (a, b) if a < b else (b, a)


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
    mesh = HalfEdgeMesh(len(V), faces)
    new_V = []

    # Step 1: move every OLD vertex
    #   interior:  (1 - k*beta) * v + beta * (sum of k neighbours),  beta = 3/(8k)  (3/16 if k = 3)
    #   boundary:  3/4 * v + 1/8 * (each of the two boundary neighbours)
    for v in range(len(V)):
        if mesh.on_boundary(v):
            bn = mesh.boundary_neighbours(v)
            if len(bn) == 2:
                new_V.append(3 / 4 * V[v] + 1 / 8 * (V[bn[0]] + V[bn[1]]))
            else:
                new_V.append(V[v].copy())
        else:
            k = mesh.valence(v)
            beta = 3 / 16 if k == 3 else 3 / (8 * k)
            new_V.append((1 - k * beta) * V[v] + beta * V[mesh.neighbours(v)].sum(axis=0))

    # Step 2: add a NEW vertex on every edge
    #   interior edge:  3/8 * (a + b) + 1/8 * (c + d)      c, d = opposite vertices
    #   boundary edge:  1/2 * (a + b)
    edge_vertex = {}
    for _, a, b in mesh.edges():
        edge_vertex[edge_key(a, b)] = len(new_V)
        if mesh.is_boundary_edge(a, b):
            new_V.append((V[a] + V[b]) / 2)
        else:
            c, d = mesh.opposite(a, b)
            new_V.append(3 / 8 * (V[a] + V[b]) + 1 / 8 * (V[c] + V[d]))

    # Step 3: split every triangle into 4
    return np.array(new_V), np.array(split_into_four(faces, edge_vertex))


# =====================================================================
#  2) MODIFIED BUTTERFLY  (interpolating - old points never move)
# =====================================================================
def butterfly(V, F):
    faces = F.tolist()
    mesh = HalfEdgeMesh(len(V), faces)

    # Step 1: old vertices stay exactly where they are
    new_V = [p.copy() for p in V]

    # Step 2: add a new vertex on every edge
    edge_vertex = {}
    for _, a, b in mesh.edges():
        edge_vertex[edge_key(a, b)] = len(new_V)
        new_V.append(butterfly_edge_point(V, mesh, a, b))

    # Step 3: split every triangle into 4
    return np.array(new_V), np.array(split_into_four(faces, edge_vertex))


def butterfly_edge_point(V, mesh, a, b):
    # Boundary edge -> 4-point rule:  -1/16, 9/16, 9/16, -1/16
    if mesh.is_boundary_edge(a, b):
        a0 = [u for u in mesh.boundary_neighbours(a) if u != b]
        b1 = [u for u in mesh.boundary_neighbours(b) if u != a]
        if not a0 or not b1:
            return (V[a] + V[b]) / 2
        return 9 / 16 * (V[a] + V[b]) - 1 / 16 * (V[a0[0]] + V[b1[0]])

    # "Extraordinary" = interior vertex whose valence is not 6
    def extraordinary(v):
        return not mesh.on_boundary(v) and mesh.valence(v) != 6

    pa = special_rule(V, mesh, a, b) if extraordinary(a) else None
    pb = special_rule(V, mesh, b, a) if extraordinary(b) else None
    if pa is not None and pb is not None:
        return (pa + pb) / 2          # both ends extraordinary -> average
    if pa is not None:
        return pa
    if pb is not None:
        return pb
    return standard_rule(V, mesh, a, b)


def standard_rule(V, mesh, a, b):
    """8-point butterfly:  1/2 (a + b) + 1/8 (c + d) - 1/16 (four wing points)"""
    c, d = mesh.opposite(a, b)

    def wing(x, y, away):
        # vertex on the other side of edge (x, y); if missing (boundary) -> reflect 'away'
        for w in mesh.opposite(x, y):
            if w != away:
                return V[w]
        return V[x] + V[y] - V[away]

    wings = wing(a, c, b) + wing(b, c, a) + wing(a, d, b) + wing(b, d, a)
    return 1 / 2 * (V[a] + V[b]) + 1 / 8 * (V[c] + V[d]) - 1 / 16 * wings


def special_rule(V, mesh, centre, other):
    """Edge next to a vertex of valence k != 6:  3/4 * centre + sum of s_j * ring_j"""
    ring = mesh.ring(centre, other)       # ring[0] = other
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
    mesh = HalfEdgeMesh(len(V), faces)
    n = len(V)

    # Step 1: move old interior vertices:  (1 - alpha) v + alpha/k * sum(neighbours)
    #         alpha = (4 - 2 cos(2 pi / k)) / 9
    new_V = []
    for v in range(n):
        if mesh.on_boundary(v):
            new_V.append(V[v].copy())
        else:
            k = mesh.valence(v)
            alpha = (4 - 2 * math.cos(2 * math.pi / k)) / 9
            new_V.append((1 - alpha) * V[v] + alpha / k * V[mesh.neighbours(v)].sum(axis=0))

    # Step 2: add the centre point of every triangle (index n + face number)
    for a, b, c in faces:
        new_V.append((V[a] + V[b] + V[c]) / 3)

    # Step 3: "flip" every old edge - connect the two centre points instead.
    #   half-edge h = a -> b lies in face[h]; its twin b -> a lies in the other face.
    new_F = []
    for h, a, b in mesh.edges():
        f1 = mesh.face[h]
        if mesh.twin[h] != -1:
            f2 = mesh.face[mesh.twin[h]]
            new_F += [(a, n + f2, n + f1), (b, n + f1, n + f2)]
        else:                                         # boundary edge: keep it
            new_F.append((a, b, n + f1))

    return np.array(new_V), np.array(new_F)


# Schemes shown in the menu
SCHEMES = {"Loop": loop, "Butterfly": butterfly, "Sqrt(3)": sqrt3}
