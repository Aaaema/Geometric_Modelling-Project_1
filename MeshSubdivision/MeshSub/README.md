# MeshSub - Simple Subdivision Viewer (Task 1)

Only needs **numpy** and **matplotlib**.

## Run
```
pip install -r requirements.txt
python main.py
```
(or `python main.py models/cube.obj` to open a file directly)

## Controls
| Action | How |
|---|---|
| Choose scheme (menu) | Click Loop / Butterfly / Sqrt(3) on the left |
| Open OBJ or STL | "Open OBJ / STL" button |
| Next iteration | SPACE |
| Wireframe / flat shaded | W |
| Reset to original | R |
| Rotate / zoom | mouse drag / scroll wheel |

## Files
| File | What it does |
|---|---|
| `mesh_io.py` | Reads OBJ and STL files into vertices `V` and triangles `F` |
| `halfedge.py` | Half-edge mesh data structure (origin / next / twin / face per half-edge) used for all neighbour look-ups |
| `subdivision.py` | Loop, Butterfly and Sqrt(3) schemes (built on the half-edge mesh) |
| `main.py` | Window: 3D view, menu, keys, wireframe / flat shading |

## The three schemes in one line each
- **Loop** (approximating): old points move, new point on every edge (3/8, 3/8, 1/8, 1/8), each triangle -> 4. Very smooth, shrinks.
- **Butterfly** (interpolating): old points stay, new edge point uses 8 neighbours (1/2, 1/8, -1/16). Keeps the shape, less smooth.
- **Sqrt(3)**: add the centre of each triangle, move old points a little, flip old edges. Each triangle -> 3.
