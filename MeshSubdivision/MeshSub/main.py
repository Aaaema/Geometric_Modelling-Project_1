"""MeshSub - simple subdivision viewer (numpy + matplotlib only).

Mouse:  left-drag = rotate,  scroll = zoom
Keys:   SPACE = next subdivision iteration
        W     = switch wireframe / flat shaded
        R     = reset to the original mesh
Menu:   choose the scheme on the left, "Open file" button to load OBJ / STL
"""
import os
import sys

import matplotlib
if __name__ == "__main__":
    matplotlib.use("TkAgg")          # real window (also inside PyCharm)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mesh_io import load_mesh, cube
from subdivision import SCHEMES

MAX_FACES = 20_000                   # matplotlib gets slow above this
LIGHT = np.array([0.4, 0.5, 1.0]) / np.linalg.norm([0.4, 0.5, 1.0])

# free the keys we use from matplotlib's own shortcuts
for name in ("keymap.home", "keymap.save", "keymap.fullscreen", "keymap.quit"):
    plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in ("r", "w", " ")]


class Viewer:
    def __init__(self, V, F, name):
        self.scheme = "Loop"
        self.wireframe = False

        self.fig = plt.figure("MeshSub - Subdivision Viewer", figsize=(11, 7.5))
        self.ax = self.fig.add_axes([0.22, 0.05, 0.76, 0.88], projection="3d")

        # ----- "menu" on the left -----
        self.fig.text(0.02, 0.92, "Subdivision scheme", fontsize=11, weight="bold")
        radio_ax = self.fig.add_axes([0.02, 0.72, 0.17, 0.18])
        self.radio = RadioButtons(radio_ax, list(SCHEMES))
        self.radio.on_clicked(self.choose_scheme)

        open_ax = self.fig.add_axes([0.02, 0.62, 0.17, 0.06])
        self.open_btn = Button(open_ax, "Open OBJ / STL")
        self.open_btn.on_clicked(self.open_file)

        self.fig.text(0.02, 0.30,
                      "Keys\n"
                      "SPACE  next iteration\n"
                      "W        wireframe / flat\n"
                      "R        reset\n\n"
                      "Mouse\n"
                      "drag     rotate\n"
                      "scroll   zoom",
                      fontsize=9, family="monospace", va="bottom")
        self.info = self.fig.text(0.22, 0.95, "", fontsize=10)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.set_mesh(V, F, name)

    # ---------------- actions ----------------
    def set_mesh(self, V, F, name):
        self.V0, self.F0, self.name = V, F, name
        # fixed view box around the ORIGINAL mesh, so shrinking (Loop) is visible
        centre = (V.max(axis=0) + V.min(axis=0)) / 2
        radius = (V.max(axis=0) - V.min(axis=0)).max() / 2 * 1.3
        self.limits = [(c - radius, c + radius) for c in centre]
        self.reset()

    def reset(self):
        self.V, self.F, self.iteration = self.V0, self.F0, 0
        self.draw()

    def choose_scheme(self, label):
        self.scheme = label
        self.reset()                          # every scheme starts from the original

    def next_iteration(self):
        if len(self.F) * 4 > MAX_FACES:
            self.draw("Mesh is big enough - press R to reset")
            return
        self.V, self.F = SCHEMES[self.scheme](self.V, self.F)
        self.iteration += 1
        self.draw()

    def toggle_mode(self):
        self.wireframe = not self.wireframe
        self.draw()

    def open_file(self, _event=None):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Open mesh", initialdir="models" if os.path.isdir("models") else ".",
            filetypes=[("Meshes", "*.obj *.stl"), ("All files", "*.*")])
        if path:
            try:
                V, F = load_mesh(path)
                self.set_mesh(V, F, os.path.basename(path))
            except Exception as e:
                self.draw("Could not load file: " + str(e))

    # ---------------- events ----------------
    def on_key(self, event):
        if event.key == " ":
            self.next_iteration()
        elif event.key == "w":
            self.toggle_mode()
        elif event.key == "r":
            self.reset()

    def on_scroll(self, event):
        factor = 0.9 if event.button == "up" else 1.1    # up = zoom in
        self.limits = [((lo + hi) / 2 - (hi - lo) / 2 * factor,
                        (lo + hi) / 2 + (hi - lo) / 2 * factor) for lo, hi in self.limits]
        self.apply_limits()
        self.fig.canvas.draw_idle()

    # ---------------- drawing ----------------
    def draw(self, message=""):
        elev, azim = self.ax.elev, self.ax.azim          # keep the user's rotation
        self.ax.clear()
        triangles = self.V[self.F]                         # shape (m, 3, 3)

        if self.wireframe:
            # wireframe: transparent faces, only the edges are drawn
            poly = Poly3DCollection(triangles, facecolors=np.zeros((len(triangles), 4)),
                                    edgecolors="navy", linewidths=0.5)
        else:
            # flat shading: ONE colour per triangle, from the angle to the light
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
            brightness = 0.25 + 0.75 * np.abs(normals @ LIGHT)
            colours = np.outer(brightness, [0.55, 0.65, 0.85])
            poly = Poly3DCollection(triangles, facecolors=colours, edgecolors="none")
        self.ax.add_collection3d(poly)

        self.ax.view_init(elev=elev, azim=azim)
        self.apply_limits()
        self.ax.set_axis_off()
        self.info.set_text(
            f"{self.name}   |   Scheme: {self.scheme}   |   Iteration: {self.iteration}   |   "
            f"Vertices: {len(self.V)}   |   Faces: {len(self.F)}   |   "
            f"{'Wireframe' if self.wireframe else 'Flat shaded'}   {message}")
        self.fig.canvas.draw_idle()

    def apply_limits(self):
        self.ax.set_xlim(*self.limits[0])
        self.ax.set_ylim(*self.limits[1])
        self.ax.set_zlim(*self.limits[2])
        self.ax.set_box_aspect((1, 1, 1))


if __name__ == "__main__":
    if len(sys.argv) > 1:                                # optional: python main.py models/cube.obj
        V, F = load_mesh(sys.argv[1])
        viewer = Viewer(V, F, os.path.basename(sys.argv[1]))
    else:
        viewer = Viewer(*cube(), "built-in cube")
    plt.show()
