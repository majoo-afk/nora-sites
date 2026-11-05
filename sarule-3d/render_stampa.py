"""Anteprima 'plastica grigia' del modello stampato, dalla heightmap."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
z = np.load("output/heightmap_mm.npy"); step = 200 / (z.shape[1] - 1)
s = 2; zz = z[::s, ::s]; h, w = zz.shape
X, Y = np.meshgrid(np.arange(w) * step * s, (h - 1 - np.arange(h)) * step * s)
ls = LightSource(azdeg=300, altdeg=35)
col = ls.shade(zz, cmap=plt.cm.gray, vert_exag=1, blend_mode="hsv", vmin=-40, vmax=60)
fig = plt.figure(figsize=(12, 8), dpi=160); ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, Y, zz, facecolors=col, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
ax.set_box_aspect((X.max(), Y.max(), zz.max())); ax.view_init(elev=32, azim=-115)
ax.set_axis_off(); fig.savefig("output/anteprima_stampa.png", bbox_inches="tight", facecolor="white")
