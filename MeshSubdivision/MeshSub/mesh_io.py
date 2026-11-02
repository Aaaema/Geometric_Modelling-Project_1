"""Loading meshes from OBJ and STL (ASCII or binary) files.

A mesh is two numpy arrays:
    V - vertices, shape (n, 3), float
    F - triangles, shape (m, 3), int (indices into V, counter-clockwise from outside)
"""
import numpy as np


def load_mesh(path):
    name = path.lower()
    if name.endswith(".obj"):
        verts, faces = _load_obj(path)
    elif name.endswith(".stl"):
        verts, faces = _load_stl(path)
    else:
        raise ValueError("Unsupported file type: " + path)

    faces = [f for f in faces if len(set(f)) == 3]      # drop broken triangles
    if not verts or not faces:
        raise ValueError("No triangles found in file.")

    # drop vertices that no triangle uses (they have no neighbours to average)
    V, F = np.array(verts, dtype=float), np.array(faces, dtype=int)
    used, F = np.unique(F, return_inverse=True)
    return V[used], F.reshape(-1, 3)


def cube():
    """Built-in triangulated cube, shown when the program starts."""
    V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                  [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]], dtype=int)
    return V, F


# ---------------- OBJ ----------------
# Only "v x y z" and "f i j k ..." lines are used. Polygons are split into triangles (fan).
def _load_obj(path):
    verts, faces = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idx = []
                for token in parts[1:]:
                    k = int(token.split("/")[0])             # "5/2/7" -> 5
                    idx.append(k - 1 if k > 0 else len(verts) + k)
                for i in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[i], idx[i + 1]))
    return verts, faces


# ---------------- STL ----------------
# STL stores every triangle separately, so equal corners are merged ("welded").
def _load_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    corners = _read_binary_stl(data) if _is_binary_stl(data) else _read_ascii_stl(data)

    index, verts, faces = {}, [], []
    for i in range(0, len(corners) - 2, 3):
        tri = []
        for c in corners[i:i + 3]:
            key = tuple(c)
            if key not in index:
                index[key] = len(verts)
                verts.append(list(key))
            tri.append(index[key])
        faces.append(tuple(tri))
    return verts, faces


def _is_binary_stl(data):
    if len(data) < 84:
        return False
    count = int.from_bytes(data[80:84], "little")
    return 84 + count * 50 == len(data)


def _read_binary_stl(data):
    count = int.from_bytes(data[80:84], "little")
    record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    tris = np.frombuffer(data, dtype=record, count=count, offset=84)
    return [tuple(float(x) for x in p) for p in tris["v"].reshape(-1, 3)]


def _read_ascii_stl(data):
    corners = []
    for line in data.decode("ascii", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].lower() == "vertex":
            corners.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return corners
