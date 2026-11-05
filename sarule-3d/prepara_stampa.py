#!/usr/bin/env python3
"""Prepara lo ZIP per i service di stampa a colori (Craftcloud, JLC3DP...).

- Plastico 380 x 285 mm (rientra nei 390 x 340 mm della stampa full-color JLC3DP)
- Guscio cavo, spessore >= 3,5 mm, aperto sotto (meno materiale = prezzo piu' basso)
- Texture "cartografica" libera da diritti: tinta per quota + ombreggiatura dal DTM
  regionale, torrenti / laghetti / strade / edifici da OpenStreetMap
- Uscita: output/stampa/sarule_gonare_stampa.zip  (OBJ + MTL + JPG alla radice)
"""
import os
import zipfile

import numpy as np
from lxml import etree
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import minimum_filter
from shapely.geometry import LineString, Polygon
import trimesh

import genera_colori as gc

LATO_MM = 380.0
GUSCIO_MM = 3.5
BASE_MM = 5.0
ESAG = 1.5
STEP_M = 10.0          # griglia geometria (0,6 mm sul modello)
TEX_PX_M = 1.5625      # 4096 x 3072 px di texture
OUT = "output/stampa"
NOME = "sarule_gonare_stampa"


def osm_layers(path):
    to_utm = Transformer.from_crs("EPSG:4326", gc.UTM, always_xy=True).transform
    root = etree.parse(path).getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    L = {"edifici": [], "strade": [], "piste": [], "torrenti": [], "acqua": []}
    for w in root.iter("way"):
        t = {x.get("k"): x.get("v") for x in w.iter("tag")}
        pts = [to_utm(*nodes[n.get("ref")]) for n in w.iter("nd") if n.get("ref") in nodes]
        if len(pts) < 2:
            continue
        hw = t.get("highway")
        if "building" in t and len(pts) >= 4:
            L["edifici"].append(Polygon(pts).buffer(0))
        elif hw in gc.__dict__.get("ROADS", ()) or hw in (
                "primary", "secondary", "tertiary", "unclassified", "residential",
                "living_street", "service"):
            L["strade"].append(LineString(pts).buffer(6 if hw in ("primary", "secondary", "tertiary") else 3.5))
        elif hw in ("track", "path", "footway"):
            L["piste"].append(LineString(pts).buffer(2))
        elif t.get("waterway") == "stream":
            L["torrenti"].append(LineString(pts).buffer(4))
        elif t.get("natural") == "water" and len(pts) >= 4:
            L["acqua"].append(Polygon(pts).buffer(0))
    return L


def tinta(dem):
    """Palette ipsometrica sobria: verde salvia -> oliva -> ocra -> pietra."""
    stops = [(390, (86, 112, 74)), (520, (112, 132, 82)), (650, (148, 150, 96)),
             (780, (176, 160, 112)), (900, (166, 146, 118)), (1000, (172, 166, 156)),
             (1085, (214, 210, 202))]
    q = np.array([s[0] for s in stops], float)
    c = np.array([s[1] for s in stops], float)
    out = np.stack([np.interp(dem, q, c[:, i]) for i in range(3)], -1)
    return out / 255.0


def hillshade(dem, px, az=315, alt=40, ex=2.0):
    gy, gx = np.gradient(dem * ex, px)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, z = np.radians(az), np.radians(alt)
    hs = np.sin(z) * np.sin(slope) + np.cos(z) * np.cos(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)


def texture(L):
    px = TEX_PX_M
    W, H = int((gc.X1 - gc.X0) / px), int((gc.Y1 - gc.Y0) / px)
    dem = gc.to_grid(gc.wcs("raster__DTM_10M_ALTIMETRIA_REV01", 1, f"{gc.CACHE}/dtm10m.tif"), px)
    d1 = gc.to_grid(gc.wcs("raster__DTM_1M_MOSAICO_ALTIMETRIA", 0.5, f"{gc.CACHE}/dtm2m.tif"), px)
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import gaussian_filter
    dem = gaussian_filter(dem, 3)               # ammorbidisce il DTM 10 m
    v = ~np.isnan(d1)
    wgt = np.clip(distance_transform_edt(v) * px / 80.0, 0, 1)
    dem = (dem * (1 - wgt) + np.where(v, d1, dem) * wgt)[:H, :W]
    rgb = tinta(dem)
    hs = hillshade(dem, px)[..., None]
    rgb = rgb * (0.45 + 0.75 * hs)
    tr = from_origin(gc.X0, gc.Y1, px, px)

    def paint(geoms, color, alpha=1.0):
        if not geoms:
            return
        m = rasterize(geoms, out_shape=(H, W), transform=tr, all_touched=True).astype(bool)
        rgb[m] = rgb[m] * (1 - alpha) + np.array(color) / 255.0 * alpha

    paint(L["piste"], (226, 214, 186), 0.55)
    paint(L["torrenti"], (78, 128, 160), 0.85)
    paint(L["acqua"], (70, 120, 156), 1.0)
    paint(L["strade"], (246, 240, 226), 0.95)
    paint(L["edifici"], (178, 82, 56), 1.0)
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8"))
    return img


def guscio(z, step_mm):
    """Superficie esterna (z), pareti, anello di base, superficie interna
    (z ridotta di GUSCIO_MM, rimpicciolita in pianta di GUSCIO_MM per lato)."""
    h, w = z.shape
    LX, LY = (w - 1) * step_mm, (h - 1) * step_mm
    win = int(np.ceil(GUSCIO_MM / step_mm)) * 2 + 1
    zin = minimum_filter(z, size=win) - GUSCIO_MM
    assert zin.min() > 0.3, zin.min()
    ys, xs = np.mgrid[0:h, 0:w]
    xo, yo = xs * step_mm, (h - 1 - ys) * step_mm
    sx, sy = (LX - 2 * GUSCIO_MM) / LX, (LY - 2 * GUSCIO_MM) / LY
    xi, yi = GUSCIO_MM + xo * sx, GUSCIO_MM + yo * sy
    top = np.column_stack([xo.ravel(), yo.ravel(), z.ravel()])
    inn = np.column_stack([xi.ravel(), yi.ravel(), zin.ravel()])
    uv = np.column_stack([xs.ravel() / (w - 1), 1 - ys.ravel() / (h - 1)])
    n = h * w
    idx = np.arange(n).reshape(h, w)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    f_top = np.vstack([np.column_stack([a.ravel(), c.ravel(), b.ravel()]),
                       np.column_stack([b.ravel(), c.ravel(), d.ravel()])])
    f_inn = f_top[:, ::-1] + n                    # normali verso il basso/interno
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    k = len(ring)
    ob = top[ring].copy(); ob[:, 2] = 0           # bordo esterno a terra
    ib = inn[ring].copy(); ib[:, 2] = 0           # bordo interno a terra
    OB, IB = 2 * n + np.arange(k), 2 * n + k + np.arange(k)
    nx = np.roll(np.arange(k), -1)
    r, rn = ring, ring[nx]
    f_wall_o = np.vstack([np.column_stack([r, OB, rn]), np.column_stack([rn, OB, OB[nx]])])
    f_wall_i = np.vstack([np.column_stack([r + n, rn + n, IB]), np.column_stack([rn + n, IB[nx], IB])])
    f_rim = np.vstack([np.column_stack([OB, IB, OB[nx]]), np.column_stack([OB[nx], IB, IB[nx]])])
    V = np.vstack([top, inn, ob, ib])
    return V, uv, f_top, np.vstack([f_inn, f_wall_o, f_wall_i, f_rim])


def write_obj(path, V, uv, f_tex, f_plain):
    base = os.path.splitext(os.path.basename(path))[0]
    with open(path, "w") as f:
        f.write(f"# Plastico Sarule - Monte Gonare, unita' mm\nmtllib {base}.mtl\n")
        np.savetxt(f, V, fmt="v %.3f %.3f %.3f")
        np.savetxt(f, uv, fmt="vt %.5f %.5f")
        f.write("usemtl terreno\n")
        ft = f_tex + 1
        np.savetxt(f, np.column_stack([ft[:, 0], ft[:, 0], ft[:, 1], ft[:, 1], ft[:, 2], ft[:, 2]]),
                   fmt="f %d/%d %d/%d %d/%d")
        f.write("usemtl base\n")
        np.savetxt(f, f_plain + 1, fmt="f %d %d %d")
    with open(os.path.join(os.path.dirname(path), base + ".mtl"), "w") as f:
        f.write("newmtl terreno\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n"
                f"map_Kd {base}.jpg\n\nnewmtl base\nKa 1 1 1\nKd 0.42 0.30 0.21\nKs 0 0 0\nd 1\nillum 1\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(gc.CACHE, exist_ok=True)
    mm_per_m = LATO_MM / (gc.X1 - gc.X0)
    dem = gc.dem_fuso(STEP_M)
    hb = gc.edifici("sarule.osm", STEP_M, dem.shape)
    z = (dem - dem.min() + hb) * mm_per_m * ESAG + BASE_MM
    V, uv, ft, fp = guscio(z, STEP_M * mm_per_m)
    nf = len(ft)                                # normali verso l'esterno:
    fp = np.vstack([fp[:nf], fp[nf:, ::-1]])    # pareti e anello di base invertiti

    m = trimesh.Trimesh(V, np.vstack([ft, fp]), process=True)
    print(f"Guscio: {len(m.faces)} triangoli, {m.extents.round(1)} mm, watertight={m.is_watertight}, winding={m.is_winding_consistent}, "
          f"volume {m.volume/1000:.0f} cm3 (pieno sarebbe ~{(z.mean()*m.extents[0]*m.extents[1])/1000:.0f} cm3)")

    img = texture(osm_layers("sarule.osm"))
    img.save(f"{OUT}/{NOME}.jpg", quality=92)
    write_obj(f"{OUT}/{NOME}.obj", V, uv, ft, fp)
    zp = f"{OUT}/{NOME}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for ext in ("obj", "mtl", "jpg"):
            zf.write(f"{OUT}/{NOME}.{ext}", f"{NOME}.{ext}")   # file alla radice dello zip
    for ext in ("obj", "jpg", "zip"):
        print(f"{NOME}.{ext}: {os.path.getsize(f'{OUT}/{NOME}.{ext}')/1e6:.1f} MB")


if __name__ == "__main__":
    main()
