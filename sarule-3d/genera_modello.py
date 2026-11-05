#!/usr/bin/env python3
"""Plastico 3D stampabile di Sarule e del Monte Gonare.

Dati gratuiti:
  - Altimetria: Copernicus DEM GLO-30 (30 m, ESA) letto via HTTP (COG)
  - Edifici e strade: OpenStreetMap (file .osm scaricato dall'API OSM)

Uso:
  python3 genera_modello.py --osm sarule.osm --out output/

Il modello e' una "heightmap" chiusa (solido unico, watertight): terreno,
centro abitato in rilievo e strade principali leggermente rialzate.
"""
import argparse
import os

import numpy as np
import rasterio
from lxml import etree
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, Polygon
from shapely.ops import transform as shp_transform, unary_union
import trimesh

DEM_URL = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_10_N40_00_E009_00_DEM/"
           "Copernicus_DSM_COG_10_N40_00_E009_00_DEM.tif")
UTM = "EPSG:32632"
ROAD_TYPES = {"primary", "secondary", "tertiary", "unclassified", "residential",
              "primary_link", "secondary_link", "tertiary_link"}

LUOGHI = {"Sarule": (40.2291, 9.1673), "Monte Gonare": (40.2276, 9.2028)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--osm", required=True)
    p.add_argument("--out", default="output")
    p.add_argument("--centro", default="40.2265,9.1850", help="lat,lon")
    p.add_argument("--larghezza-km", type=float, default=6.4)
    p.add_argument("--altezza-km", type=float, default=4.8)
    p.add_argument("--stampa-mm", type=float, default=200, help="lato lungo")
    p.add_argument("--esagerazione", type=float, default=1.5)
    p.add_argument("--passo-m", type=float, default=10)
    p.add_argument("--base-mm", type=float, default=3.0)
    p.add_argument("--edifici-mm", type=float, default=1.2)
    p.add_argument("--strade-mm", type=float, default=0.4)
    return p.parse_args()


def load_dem(transform, shape):
    dst = np.zeros(shape, dtype="float32")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(DEM_URL) as src:
            reproject(source=rasterio.band(src, 1), destination=dst,
                      dst_transform=transform, dst_crs=UTM,
                      resampling=Resampling.cubic)
    return dst


def load_osm(path, to_utm):
    root = etree.parse(path).getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat")))
             for n in root.iter("node")}
    buildings, roads = [], []
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
        pts = [nodes[nd.get("ref")] for nd in w.iter("nd") if nd.get("ref") in nodes]
        if len(pts) < 2:
            continue
        pts = [to_utm(*p) for p in pts]
        if "building" in tags and len(pts) >= 4:
            poly = Polygon(pts)
            if poly.is_valid and poly.area > 5:
                buildings.append(poly)
        elif tags.get("highway") in ROAD_TYPES:
            roads.append(LineString(pts))
    return buildings, roads


def heightmap_to_mesh(z, step_mm):
    """Solido chiuso: superficie superiore + pareti + fondo piatto (z=0)."""
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w]
    top = np.column_stack([xs.ravel() * step_mm, (h - 1 - ys.ravel()) * step_mm, z.ravel()])
    idx = np.arange(h * w).reshape(h, w)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = [np.column_stack([a.ravel(), c.ravel(), b.ravel()]),
             np.column_stack([b.ravel(), c.ravel(), d.ravel()])]

    # bordo in senso orario visto dall'alto
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    n = len(top)
    bottom = top[ring].copy()
    bottom[:, 2] = 0.0
    bidx = n + np.arange(len(ring))
    nxt = np.roll(np.arange(len(ring)), -1)
    faces.append(np.column_stack([ring, bidx, ring[nxt]]))
    faces.append(np.column_stack([ring[nxt], bidx, bidx[nxt]]))
    # fondo: ventaglio da un punto centrale
    center = np.array([[top[:, 0].mean(), top[:, 1].mean(), 0.0]])
    cidx = n + len(ring)
    faces.append(np.column_stack([np.full(len(ring), cidx), bidx[nxt], bidx]))
    verts = np.vstack([top, bottom, center])
    mesh = trimesh.Trimesh(verts, np.vstack(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform

    lat, lon = map(float, a.centro.split(","))
    cx, cy = to_utm(lon, lat)
    W, H = a.larghezza_km * 1000, a.altezza_km * 1000
    x0, y1 = cx - W / 2, cy + H / 2
    nx, ny = int(W / a.passo_m) + 1, int(H / a.passo_m) + 1
    transform = from_origin(x0, y1, a.passo_m, a.passo_m)

    print(f"Area {W/1000:.1f} x {H/1000:.1f} km, griglia {nx} x {ny}")
    dem = gaussian_filter(load_dem(transform, (ny, nx)), sigma=1.0)
    print(f"Quote: min {dem.min():.0f} m, max {dem.max():.0f} m")

    
    mm_per_m = a.stampa_mm / max(W, H)
    z = (dem - dem.min()) * mm_per_m * a.esagerazione + a.base_mm

    buildings, roads = load_osm(a.osm, to_utm)
    print(f"OSM: {len(buildings)} edifici, {len(roads)} strade")
    # Gli edifici singoli (10 m = 0,3 mm) sono troppo piccoli da stampare:
    # li fondo in isolati (dilatazione 6 m, unione, erosione 3 m).
    blocchi = unary_union([b.buffer(6) for b in buildings]).buffer(-3)
    strade = unary_union([r.buffer(7) for r in roads])
    m_ed = rasterize([blocchi], out_shape=(ny, nx), transform=transform) if buildings else 0
    m_st = rasterize([strade], out_shape=(ny, nx), transform=transform) if roads else 0
    z = z + np.where(m_ed > 0, a.edifici_mm, np.where(m_st > 0, a.strade_mm, 0))

    step_mm = a.passo_m * mm_per_m
    mesh = heightmap_to_mesh(z.astype("float64"), step_mm)
    print(f"Mesh: {len(mesh.faces)} triangoli, watertight={mesh.is_watertight}, "
          f"dimensioni {mesh.extents.round(1)} mm, scala 1:{1/mm_per_m*1000:.0f}")
    stl = os.path.join(a.out, "sarule_gonare.stl")
    mesh.export(stl)
    np.save(os.path.join(a.out, "heightmap_mm.npy"), z)

    # posizioni dei luoghi in pixel, per le anteprime
    inv = ~transform
    luoghi = {k: inv * to_utm(lo, la) for k, (la, lo) in LUOGHI.items()}
    np.save(os.path.join(a.out, "luoghi.npy"), luoghi, allow_pickle=True)
    render(dem, z, m_ed, m_st, luoghi, a.out, step_mm)
    print("Fatto:", stl)


def render(dem, z, m_ed, m_st, luoghi, out, step_mm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    ls = LightSource(azdeg=315, altdeg=40)
    rgb = ls.shade(dem, cmap=plt.cm.terrain, vert_exag=2, blend_mode="soft",
                   vmin=dem.min() - 150, vmax=dem.max() + 50)
    rgb[m_st > 0, :3] = rgb[m_st > 0, :3] * 0.4 + np.array([0.95, 0.9, 0.8]) * 0.6
    rgb[m_ed > 0, :3] = [0.75, 0.25, 0.2]
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=150)
    ax.imshow(rgb)
    for k, (px, py) in luoghi.items():
        ax.plot(px, py, "k^" if "Monte" in k else "ko", ms=6)
        ax.annotate(k, (px, py), xytext=(6, -10), textcoords="offset points",
                    fontsize=11, weight="bold")
    ax.set_axis_off()
    ax.set_title("Sarule e Monte Gonare - rilievo (Copernicus 30 m) + OSM")
    fig.savefig(os.path.join(out, "anteprima_mappa.png"), bbox_inches="tight")
    plt.close(fig)

    s = 3  # sottocampionamento per la vista 3D
    zz = z[::s, ::s]
    h, w = zz.shape
    X, Y = np.meshgrid(np.arange(w) * step_mm * s, (h - 1 - np.arange(h)) * step_mm * s)
    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(projection="3d")
    col = rgb[::s, ::s, :3]
    ax.plot_surface(X, Y, zz, facecolors=col, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((X.max(), Y.max(), zz.max() * 2.0))
    ax.view_init(elev=35, azim=-110)
    ax.set_axis_off()
    fig.savefig(os.path.join(out, "anteprima_3d.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
