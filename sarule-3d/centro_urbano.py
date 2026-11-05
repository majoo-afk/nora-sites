#!/usr/bin/env python3
"""Modello 3D dettagliato del CENTRO URBANO di Sarule (LoD1/LoD2).

- Terreno: DTM LiDAR 1 m Regione Sardegna (WCS)
- Edifici: sagome DBGT 10k Regione Sardegna (WFS) + edifici solo-OSM
- Altezze: stimate da uso/dimensione (+ building:levels OSM) - correggibili in ALTEZZE_MANUALI
- Tetti a padiglione generati dalla sagoma (altezza = distanza dal bordo x pendenza)

Uscite (output/centro/): sarule_centro.stl, viewer_centro.json (+ html con build in fondo)
"""
import base64
import json
import os
import urllib.request

import numpy as np
from lxml import etree
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import LineString, Polygon, shape
import trimesh

import genera_colori as gc
from genera_modello import heightmap_to_mesh

WFS = "https://webgis.regione.sardegna.it/geoserver/ows"
CX, CY = 514260, 4453200
W_M, H_M = 1100, 900                 # area reale in metri
X0, X1, Y0, Y1 = CX - W_M / 2, CX + W_M / 2, CY - H_M / 2, CY + H_M / 2
STAMPA_MM = 360                      # lato lungo
STEP = 0.6                           # m, griglia
BASE_MM = 4.0
OUT = "output/centro"
FACCE_MAX = 1_600_000

# correzioni a mano: (lat, lon) -> altezza gronda in metri
ALTEZZE_MANUALI = {}


def wfs(layer, path):
    if not os.path.exists(path):
        url = (f"{WFS}?service=WFS&version=2.0.0&request=GetFeature&typeNames=dbu:dbgt_10k_22_v05_02_{layer}"
               f"&srsName=EPSG:32632&bbox={X0-50},{Y0-50},{X1+50},{Y1+50},EPSG:32632&outputFormat=application/json")
        req = urllib.request.Request(url, headers={"User-Agent": "nora-sites-plastico/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
            f.write(r.read())
    return json.load(open(path))["features"]


def edifici():
    """Lista di (poligono, gronda_m, pendenza, tipo)."""
    out = []
    for f in wfs("edificio_ingombro_suolo", f"{gc.CACHE}/dbgt_edifici.json"):
        g = shape(f["geometry"]).buffer(0)
        if g.is_empty:
            continue
        p = f["properties"]; a = g.area
        uso, tip = p.get("uso"), p.get("tipologia")
        if tip == "chiesa" or uso == "luogo di culto":
            out.append((g, 10.5, 0.50, "chiesa")); continue
        if uso in ("stalla", "industriale"):
            out.append((g, 5.0, 0.15, "rurale")); continue
        if uso == "sede di scuola":
            out.append((g, 7.5, 0.25, "pubblico")); continue
        h = 3.2 if a < 35 else (8.8 if a > 350 else 6.4)
        out.append((g, h, 0.35, "casa"))
    for f in wfs("edificio_minore_ingombro_suolo", f"{gc.CACHE}/dbgt_minori.json"):
        g = shape(f["geometry"]).buffer(0)
        tip = f["properties"].get("tipologia")
        h = {"tomba cimiteriale": 2.2, "cappella": 4.0}.get(tip, 2.6)
        out.append((g, h, 0.2, "minore"))
    # edifici presenti solo in OSM
    to_utm = Transformer.from_crs("EPSG:4326", gc.UTM, always_xy=True).transform
    root = etree.parse("sarule.osm").getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    from shapely.strtree import STRtree
    tree = STRtree([o[0] for o in out])
    geoms = [o[0] for o in out]
    nuovi = 0
    for w in root.iter("way"):
        t = {x.get("k"): x.get("v") for x in w.iter("tag")}
        if "building" not in t:
            continue
        pts = [to_utm(*nodes[n.get("ref")]) for n in w.iter("nd") if n.get("ref") in nodes]
        if len(pts) < 4:
            continue
        g = Polygon(pts).buffer(0)
        if g.is_empty or g.area < 6:
            continue
        cov = sum(geoms[i].intersection(g).area for i in tree.query(g))
        lv = t.get("building:levels")
        if cov / g.area < 0.3:
            h = float(lv) * 3.1 + 0.3 if lv and lv.replace(".", "").isdigit() else (3.2 if g.area < 35 else 6.4)
            out.append((g, h, 0.35, "casa")); nuovi += 1
    print(f"Edifici: {len(out)} ({nuovi} aggiunti da OSM)")
    return out


def strade():
    to_utm = Transformer.from_crs("EPSG:4326", gc.UTM, always_xy=True).transform
    root = etree.parse("sarule.osm").getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    asf, ped = [], []
    larg = {"primary": 4.0, "secondary": 3.5, "tertiary": 3.2, "residential": 2.6, "unclassified": 2.6,
            "living_street": 2.2, "service": 1.8, "pedestrian": 2.5, "footway": 1.2, "steps": 1.2,
            "track": 1.5, "path": 0.8}
    for w in root.iter("way"):
        t = {x.get("k"): x.get("v") for x in w.iter("tag")}
        hw = t.get("highway")
        pts = [to_utm(*nodes[n.get("ref")]) for n in w.iter("nd") if n.get("ref") in nodes]
        if len(pts) < 2:
            continue
        if t.get("place") == "square" or (hw == "pedestrian" and t.get("area") == "yes"):
            if len(pts) >= 4:
                ped.append(Polygon(pts).buffer(0))
            continue
        if hw in larg:
            g = LineString(pts).buffer(larg[hw])
            (ped if hw in ("pedestrian", "footway", "steps", "path", "track") else asf).append(g)
    return asf, ped


def main():
    os.makedirs(OUT, exist_ok=True)
    nx, ny = int(W_M / STEP), int(H_M / STEP)
    tr = from_origin(X0, Y1, STEP, STEP)
    # --- terreno
    url = (f"{gc.BASE}?service=WCS&version=2.0.1&request=GetCoverage"
           f"&coverageId=raster__DTM_1M_MOSAICO_ALTIMETRIA&format=image/tiff"
           f"&subset=E({X0-20},{X1+20})&subset=N({Y0-20},{Y1+20})")
    gc.X0, gc.Y1 = X0 + STEP / 2, Y1 - STEP / 2        # to_grid usa X0/Y1 del modulo
    gc.X1, gc.Y0 = X1 - STEP / 2, Y0 + STEP / 2
    dem = gc.to_grid(gc.get(url, f"{gc.CACHE}/dtm1m_centro.tif"), STEP)[:ny, :nx]
    assert not np.isnan(dem).any(), "buchi nel DTM"
    dem = gaussian_filter(dem, 0.7)

    # --- edifici
    B = edifici()
    ids = rasterize([(g, i + 1) for i, (g, *_) in enumerate(B)], out_shape=(ny, nx),
                    transform=tr, fill=0, dtype="int32")
    m = ids > 0
    edge = np.zeros_like(m)
    for sh in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        edge |= np.roll(ids, sh, axis=(0, 1)) != ids
    inner = m & ~edge
    dist = distance_transform_edt(inner) * STEP + STEP / 2
    gronda = np.array([0] + [b[1] for b in B], float)
    pend = np.array([0] + [b[2] for b in B], float)
    # quota di base per edificio = 30 percentile del terreno sotto la sagoma
    base = np.zeros(len(B) + 1)
    flat_ids, flat_dem = ids[m], dem[m]
    order = np.argsort(flat_ids)
    sid, sdem = flat_ids[order], flat_dem[order]
    cuts = np.flatnonzero(np.diff(sid)) + 1
    for grp_id, grp in zip(sid[np.r_[0, cuts]], np.split(sdem, cuts)):
        base[grp_id] = np.percentile(grp, 30)
    tetto = base[ids] + gronda[ids] + np.minimum(dist * pend[ids], 5.5)
    z_m = np.where(m, np.maximum(dem, tetto), dem)

    mm_per_m = STAMPA_MM / W_M
    z = (z_m - dem.min()) * mm_per_m + BASE_MM
    print(f"Scala 1:{1000/mm_per_m:.0f}, quote {dem.min():.0f}-{dem.max():.0f} m, "
          f"plastico {STAMPA_MM} x {H_M*mm_per_m:.0f} x {z.max():.0f} mm")

    # --- STL
    mesh = heightmap_to_mesh(z.astype("float64"), STEP * mm_per_m)
    n0 = len(mesh.faces)
    if n0 > FACCE_MAX:
        mesh = mesh.simplify_quadric_decimation(face_count=FACCE_MAX)
        mesh.merge_vertices(); trimesh.repair.fix_normals(mesh)
    print(f"STL: {n0} -> {len(mesh.faces)} triangoli, watertight={mesh.is_watertight}, "
          f"winding={mesh.is_winding_consistent}, volume={mesh.volume/1000:.0f} cm3")
    mesh.export(f"{OUT}/sarule_centro.stl")

    # --- colori per il visualizzatore (per vertice)
    asf, ped = strade()
    hs = __import__("prepara_stampa").hillshade(dem, STEP, ex=1.0)
    terra = np.array([132, 140, 98]) / 255 * (0.55 + 0.55 * hs[..., None])
    col = terra.copy()
    if ped:
        mp = rasterize(ped, out_shape=(ny, nx), transform=tr).astype(bool)
        col[mp] = np.array([206, 198, 180]) / 255 * (0.8 + 0.25 * hs[mp][:, None])
    if asf:
        ma = rasterize(asf, out_shape=(ny, nx), transform=tr).astype(bool)
        col[ma] = np.array([118, 116, 112]) / 255 * (0.85 + 0.2 * hs[ma][:, None])
    rng = np.random.default_rng(7)
    tinta = {"casa": (176, 86, 58), "chiesa": (150, 138, 120), "rurale": (150, 128, 108),
             "pubblico": (168, 104, 74), "minore": (160, 150, 140)}
    roof = np.array([[0, 0, 0]] + [np.clip(np.array(tinta[b[3]]) * rng.uniform(0.85, 1.12), 0, 255)
                                    for b in B]) / 255
    # luce sulle falde del tetto (pendenza dalla mappa delle quote)
    gy, gx = np.gradient(z_m, STEP)
    lum = np.clip(1.0 - 0.6 * gx - 0.6 * gy, 0.6, 1.25)
    col[m] = roof[ids[m]] * lum[m][:, None]
    col[m & edge] = np.array([232, 224, 206]) / 255         # filo muro / gronda chiara
    rgb = (np.clip(col, 0, 1) * 255).astype("uint8")

    # visualizzatore: griglia piu rada per restare leggero sul telefono
    s = 3 if STEP < 0.5 else 2 if STEP < 0.8 else 1
    zs = z[::s, ::s]; cs = rgb[::s, ::s].copy()
    ids_s = ids[::s, ::s]; edge_s = np.zeros(ids_s.shape, bool)
    for sh in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        edge_s |= np.roll(ids_s, sh, axis=(0, 1)) != ids_s
    cs[(ids_s > 0) & edge_s] = (226, 216, 196)      # muri chiari anche sulla griglia rada
    near = np.zeros_like(edge_s)
    for sh in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        near |= np.roll(ids_s, sh, axis=(0, 1)) > 0
    piede = (ids_s == 0) & near                     # piede del muro: leggermente scuro
    cs[piede] = (cs[piede] * 0.8).astype('uint8')
    q = np.round(zs * 100).astype("<u2")
    json.dump({"w": zs.shape[1], "h": zs.shape[0], "step_mm": STEP * s * mm_per_m,
               "z_b64": base64.b64encode(q.tobytes()).decode(),
               "c_b64": base64.b64encode(cs.tobytes()).decode()},
              open(f"{OUT}/viewer_centro.json", "w"))
    print("viewer:", zs.shape, os.path.getsize(f"{OUT}/viewer_centro.json") // 1024, "KB")


if __name__ == "__main__":
    main()
