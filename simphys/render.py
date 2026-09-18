"""Rendering with matplotlib 3D: CPU only, no OpenGL (docs/09 §3).

Headless mode draws off-screen (Agg) and writes a GIF plus a final PNG.
Windowed mode opens an interactive matplotlib window and draws while the
simulation runs. Shells are coloured by equivalent plastic strain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bodies import Shell


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
        for b in s.bodies:
            tri = x[b.faces + b.vert0]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
            shade = 0.55 + 0.45 * np.abs(n @ light)
            col = self._face_colors(b)
            col[:, :3] *= shade[:, None]
            ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolors=(0, 0, 0, 0.08),
                                                 linewidths=0.2))
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
