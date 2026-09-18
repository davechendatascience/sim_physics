"""Rendering with matplotlib 3D: CPU only, no OpenGL (docs/09 §3).

Headless mode draws off-screen (Agg) and writes a GIF plus a final PNG.
Windowed mode opens an interactive matplotlib window and draws while the
simulation runs. Shells are coloured by equivalent plastic strain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bodies import Shell


def display_split(verts, faces, max_edge):
    """Barycentric subdivision of large triangles, for drawing only.

    matplotlib's 3D painter sorts polygons by their mean depth, so one huge
    triangle (a floor) can be drawn over a small one that sits on it. Splitting
    faces to a similar size lets the depth sort come out right. Returns, per
    display triangle, its source face and the barycentric weights (3, 3) of its
    corners, so positions can be recomputed every frame.
    """
    out_face, out_w = [], []
    stack = [(f, np.eye(3)) for f in range(len(faces))]
    while stack:
        f, w = stack.pop()
        corners = w @ verts[faces[f]]
        edges = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)
        if edges.max() <= max_edge:
            out_face.append(f)
            out_w.append(w)
            continue
        m01, m12, m20 = (w[0] + w[1]) / 2, (w[1] + w[2]) / 2, (w[2] + w[0]) / 2
        for tri in ((w[0], m01, m20), (m01, w[1], m12), (m20, m12, w[2]), (m01, m12, m20)):
            stack.append((f, np.array(tri)))
    return np.array(out_face), np.array(out_w)


def _backend(mode):
    import matplotlib
    if mode == "headless":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class Renderer:
    def __init__(self, scene, mode="headless", view=(20, -60), title="", size=(6.4, 5.4)):
        if mode not in ("headless", "window"):
            raise ValueError("mode must be 'headless' or 'window'")
        self.plt = _backend(mode)
        import matplotlib
        from matplotlib import colors
        self.cmap = matplotlib.colormaps["inferno_r"]
        self.norm = colors.LogNorm(vmin=1e-4, vmax=5e-2)
        self.scene, self.mode, self.view, self.title = scene, mode, view, title
        self.frames = []
        if mode == "window":
            self.plt.ion()
        self.fig = self.plt.figure(figsize=size)
        self.ax = self.fig.add_subplot(111, projection="3d")
        x = scene.x()
        lo, hi = x.min(0), x.max(0)
        c, r = (lo + hi) / 2, (hi - lo).max() / 2 * 1.1
        self.limits = [(ci - r, ci + r) for ci in c]
        # display subdivision, computed once on each body's rest shape
        target = (hi - lo).max() / 16
        self.split = {}
        for b in scene.bodies:
            rest = b.rest if not hasattr(b, "X") else b.X
            self.split[b.name] = display_split(rest, b.faces, target)

    def _face_colors(self, b):
        from matplotlib import colors
        base = np.array(colors.to_rgba(b.color or "#bdc3c7"))
        n = len(b.faces)
        if isinstance(b, Shell):
            a = b.face_plastic_strain()
            out = np.tile(base, (n, 1))
            hot = a > 1e-5
            out[hot] = self.cmap(self.norm(np.clip(a[hot], 1e-4, 5e-2)))
            return out
        return np.tile(base, (n, 1))

    def draw(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax, s = self.ax, self.scene
        ax.cla()
        x = s.x()
        light = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
        tris, cols = [], []
        for b in s.bodies:
            face, w = self.split[b.name]
            src = x[b.faces + b.vert0][face]                        # (m, 3 corners, 3)
            tri = np.einsum("mij,mjk->mik", w, src)
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
            col = self._face_colors(b)[face]
            col[:, :3] *= (0.55 + 0.45 * np.abs(n @ light))[:, None]
            tris.append(tri)
            cols.append(col)
        # one collection: matplotlib depth-sorts polygons within a collection,
        # but draws separate collections in whole-object order
        ax.add_collection3d(Poly3DCollection(np.concatenate(tris), facecolors=np.concatenate(cols),
                                             edgecolors="none"))
        ax.set_xlim(*self.limits[0])
        ax.set_ylim(*self.limits[1])
        ax.set_zlim(*self.limits[2])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(*self.view)
        ax.set_axis_off()
        ax.set_title(f"{self.title}  t = {s.t:.3f} s", fontsize=10)
        self.fig.canvas.draw()
        if self.mode == "window":
            self.plt.pause(0.001)
        else:
            buf = np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()
            self.frames.append(buf)

    def save(self, gif_path, fps=20):
        """Write the captured frames as a GIF and the last frame as a PNG."""
        from PIL import Image
        gif_path = Path(gif_path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            self.draw()
        imgs = [Image.fromarray(f) for f in self.frames]
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
        png = gif_path.with_suffix(".png")
        imgs[-1].save(png)
        return gif_path, png

    def close(self):
        if self.mode == "window":
            self.plt.ioff()
            self.plt.show()
        self.plt.close(self.fig)
