#!/usr/bin/env python3
"""Plastico 3D A COLORI di Sarule e Monte Gonare (per stampa full-color).

Dati (gratuiti):
  - DTM LiDAR 1 m (dove disponibile) + DTM 10 m - Regione Sardegna, WCS
  - Ortofoto 2022 - Regione Sardegna, WMS (raster:Mosaico_2022_GB)
  - Edifici - OpenStreetMap (ODbL)

Output:
  output/colori/sarule_gonare_colori.glb   modello texturizzato, solido chiuso
  output/colori/ortofoto_2m.jpg            texture
  output/colori/viewer_data.json           dati per il visualizzatore web
"""
import base64
import io
import json
import os
import urllib.request

import numpy as np
import rasterio
from lxml import etree
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import Polygon
import trimesh

BASE = "https://webgis.regione.sardegna.it/geoserverraster/ows"
UTM = "EPSG:32632"  # ~ coincide con EPSG:7791 (RDN2008 / UTM 32N) dei dati regionali
X0, Y0, X1, Y1 = 512539, 4450513, 518939, 4455313   # 6,4 x 4,8 km
STAMPA_MM = 400           # lato lungo del plastico
ESAG = 1.5                # esagerazione verticale
BASE_MM = 4.0
OUT = "output/colori"
CACHE = os.environ.get("CACHE", "cache")
UA = {"User-Agent": "nora-sites-plastico/1.0"}


def get(url, path):
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def wcs(cov, scale, path):
    url = (f"{BASE}?service=WCS&version=2.0.1&request=GetCoverage&coverageId={cov}"
           f"&format=image/tiff&subset=E({X0-60},{X1+60})&subset=N({Y0-60},{Y1+60})")
    if scale != 1:
        url += f"&scalefactor={scale}"
    return get(url, path)


def to_grid(path, step):
    nx, ny = int((X1 - X0) / step) + 1, int((Y1 - Y0) / step) + 1
    dst = np.full((ny, nx), np.nan, dtype="float32")
    with rasterio.open(path) as s:
        src = s.read(1).astype("float32")
        src[src < -1000] = np.nan
        reproject(src, dst, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=from_origin(X0 - step / 2, Y1 + step / 2, step, step),
                  dst_crs=UTM, resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def dem_fuso(step):
    """1 m LiDAR dove c'e', 10 m altrove, con raccordo di 60 m."""
    d1 = to_grid(wcs("raster__DTM_1M_MOSAICO_ALTIMETRIA", 0.5, f"{CACHE}/dtm2m.tif"), step)
    d10 = to_grid(wcs("raster__DTM_10M_ALTIMETRIA_REV01", 1, f"{CACHE}/dtm10m.tif"), step)
    valid = ~np.isnan(d1)
    w = np.clip(distance_transform_edt(valid) * step / 60.0, 0, 1)
    d1f = np.where(valid, d1, d10)
    dem = d10 * (1 - w) + d1f * w
    print(f"DTM: {valid.mean()*100:.0f}% LiDAR 1 m, quote {np.nanmin(dem):.0f}-{np.nanmax(dem):.0f} m")
    return gaussian_filter(dem, 0.6)


def ortofoto(px_m=2.0, tiles=(2, 2)):
    W, H = int((X1 - X0) / px_m), int((Y1 - Y0) / px_m)
    img = Image.new("RGB", (W, H))
    tw, th = W // tiles[0], H // tiles[1]
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            bx0, bx1 = X0 + i * tw * px_m, X0 + (i + 1) * tw * px_m
            by1, by0 = Y1 - j * th * px_m, Y1 - (j + 1) * th * px_m
            url = (f"{BASE}?service=WMS&version=1.1.1&request=GetMap"
                   f"&layers=raster:Mosaico_2022_GB&styles=&srs=EPSG:32632"
                   f"&bbox={bx0},{by0},{bx1},{by1}&width={tw}&height={th}&format=image/jpeg")
            p = get(url, f"{CACHE}/orto_{i}_{j}.jpg")
            img.paste(Image.open(p).convert("RGB"), (i * tw, j * th))
    return img


def edifici(path, step, shape):
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform
    root = etree.parse(path).getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    shapes = []
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
        if "building" not in tags:
            continue
        pts = [to_utm(*nodes[nd.get("ref")]) for nd in w.iter("nd") if nd.get("ref") in nodes]
        if len(pts) < 4:
            continue
        poly = Polygon(pts)
        if not poly.is_valid or poly.area < 5:
            continue
        try:
            h = float(tags.get("building:levels", 2)) * 3.2 + 1
        except ValueError:
            h = 7.4
        if tags.get("building") in ("church", "chapel") or tags.get("amenity") == "place_of_worship":
            h = 12
        shapes.append((poly, h))
    tr = from_origin(X0 - step / 2, Y1 + step / 2, step, step)
    hb = rasterize(shapes, out_shape=shape, transform=tr, fill=0, dtype="float32",
                   all_touched=False)
    print(f"Edifici OSM: {len(shapes)}")
    return hb


def mesh_colori(z, step_mm, tex_h_frac):
    """Solido chiuso texturizzato. La texture ha in fondo una fascia marrone
    (frazione tex_h_frac) usata per pareti e fondo."""
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w]
    top = np.column_stack([xs.ravel() * step_mm, (h - 1 - ys.ravel()) * step_mm, z.ravel()])
    uv_top = np.column_stack([xs.ravel() / (w - 1),
                              tex_h_frac + (1 - tex_h_frac) * (1 - ys.ravel() / (h - 1))])
    idx = np.arange(h * w).reshape(h, w)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = [np.column_stack([a.ravel(), c.ravel(), b.ravel()]),
             np.column_stack([b.ravel(), c.ravel(), d.ravel()])]
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    n = len(top)
    # pareti con vertici propri (UV nella fascia marrone)
    wt = top[ring].copy()
    wb = wt.copy(); wb[:, 2] = 0
    k = len(ring)
    wi_t, wi_b = n + np.arange(k), n + k + np.arange(k)
    nxt = np.roll(np.arange(k), -1)
    faces.append(np.column_stack([wi_t, wi_b, wi_t[nxt]]))
    faces.append(np.column_stack([wi_t[nxt], wi_b, wi_b[nxt]]))
    center = np.array([[top[:, 0].mean(), top[:, 1].mean(), 0.0]])
    ci = n + 2 * k
    faces.append(np.column_stack([np.full(k, ci), wi_b[nxt], wi_b]))
    verts = np.vstack([top, wt, wb, center])
    uv_brown = np.full((2 * k + 1, 2), [0.5, tex_h_frac / 2])
    uv = np.vstack([uv_top, uv_brown])
    faces = np.vstack(faces)
    # chiusura: unisci pareti al bordo superiore per avere un solido watertight
    mesh = trimesh.Trimesh(verts, faces, process=False)
    return mesh, uv


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    W = X1 - X0
    mm_per_m = STAMPA_MM / W

    # --- modello per la stampa: griglia 5 m
    step = 5.0
    dem = dem_fuso(step)
    hb = edifici("sarule.osm", step, dem.shape)
    zmin = dem.min()
    z = (dem - zmin + hb) * mm_per_m * ESAG + BASE_MM

    img = ortofoto()
    img.save(f"{OUT}/ortofoto_2m.jpg", quality=90)
    # texture: ortofoto + fascia marrone per i bordi
    band = 64
    tex = Image.new("RGB", (img.width, img.height + band), (92, 64, 44))
    tex.paste(img, (0, 0))
    frac = band / tex.height

    mesh, uv = mesh_colori(z, step * mm_per_m, frac)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=0.9))
    glb = f"{OUT}/sarule_gonare_colori.glb"
    mesh.export(glb)
    chk = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
    print(f"GLB: {len(mesh.faces)} triangoli, {mesh.extents.round(1)} mm, "
          f"scala 1:{1000/mm_per_m:.0f}, watertight (dopo merge)={chk.is_watertight}")

    # --- dati per il visualizzatore web: griglia 10 m, quote in mm*100 (uint16)
    step_v = 10.0
    dem_v = dem_fuso(step_v)
    hb_v = edifici("sarule.osm", step_v, dem_v.shape)
    zv = (dem_v - zmin + hb_v) * mm_per_m * ESAG + BASE_MM
    q = np.round(zv * 100).astype("<u2")
    small = img.resize((3200, 2400), Image.LANCZOS)
    buf = io.BytesIO(); small.save(buf, "JPEG", quality=82)
    json.dump({"w": zv.shape[1], "h": zv.shape[0], "step_mm": step_v * mm_per_m,
               "z_b64": base64.b64encode(q.tobytes()).decode(),
               "tex_b64": base64.b64encode(buf.getvalue()).decode()},
              open(f"{OUT}/viewer_data.json", "w"))
    print("viewer data:", os.path.getsize(f"{OUT}/viewer_data.json") // 1024, "KB")


if __name__ == "__main__":
    main()
