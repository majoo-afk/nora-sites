#!/usr/bin/env python3
"""Tavola di riferimento per dipingere a mano il plastico (38 x 28,5 cm).

Uscite in output/pittura/:
  tavola_pittura_A3.pdf  pag.1 mappa a zone di colore in scala 1:1 (stampare A3 "dimensioni reali")
                         pag.2 legenda colori, ricette e passaggi
  mappa_zone.png         la stessa mappa per il telefono
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from lxml import etree
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import LineString

import genera_colori as gc
import prepara_stampa as ps

OUT = "output/pittura"
LX, LY = 380.0, 285.0          # mm
PX = 3.125                     # m per pixel (2048 x 1536)

ZONE = [  # (nome, colore RGB, ricetta acrilici)
    ("Fondovalle (sotto 550 m)", (74, 98, 62), "verde vescica + punta di terra d'ombra bruciata"),
    ("Colline (550–750 m)", (116, 128, 76), "verde oliva (verde vescica + ocra gialla)"),
    ("Pascoli alti (750–950 m)", (170, 150, 98), "ocra gialla + bianco + punta di terra di Siena"),
    ("Roccia e cime (sopra 950 m, pendii ripidi)", (150, 144, 134), "grigio caldo: bianco + nero + punta di terra d'ombra"),
    ("Paese ed edifici", (172, 78, 52), "rosso terracotta (rosso ossido + ocra)"),
    ("Strade asfaltate", (238, 232, 216), "avorio (bianco + punta di ocra)"),
    ("Torrenti e laghetti", (58, 92, 104), "blu-verde scuro (blu oltremare + verde + terra d'ombra)"),
]


def dem_ref():
    d10 = gc.to_grid(gc.wcs("raster__DTM_10M_ALTIMETRIA_REV01", 1, f"{gc.CACHE}/dtm10m.tif"), PX)
    d1 = gc.to_grid(gc.wcs("raster__DTM_1M_MOSAICO_ALTIMETRIA", 0.5, f"{gc.CACHE}/dtm2m.tif"), PX)
    d10 = gaussian_filter(d10, 2)
    v = ~np.isnan(d1)
    w = np.clip(distance_transform_edt(v) * PX / 80.0, 0, 1)
    dem = d10 * (1 - w) + np.where(v, d1, d10) * w
    W, H = int((gc.X1 - gc.X0) / PX), int((gc.Y1 - gc.Y0) / PX)
    return dem[:H, :W]


def mappa_zone():
    dem = dem_ref()
    H, W = dem.shape
    gy, gx = np.gradient(dem, PX)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    zone = np.zeros(dem.shape, int)
    zone[dem >= 550] = 1
    zone[dem >= 750] = 2
    zone[(dem >= 950) | ((dem >= 800) & (slope > 32))] = 3
    zone = np.round(gaussian_filter(zone.astype(float), 1.5)).astype(int)
    pal = np.array([z[1] for z in ZONE[:4]]) / 255.0
    rgb = pal[zone]
    hs = ps.hillshade(dem, PX, ex=1.5)[..., None]
    rgb = rgb * (0.62 + 0.5 * hs)
    L = ps.osm_layers("sarule.osm")
    tr = from_origin(gc.X0, gc.Y1, PX, PX)
    for key, zi in (("torrenti", 6), ("acqua", 6), ("strade", 5), ("edifici", 4)):
        if L[key]:
            m = rasterize(L[key], out_shape=(H, W), transform=tr, all_touched=True).astype(bool)
            rgb[m] = np.array(ZONE[zi][1]) / 255.0
    return np.clip(rgb, 0, 1), dem


def etichette():
    to_utm = Transformer.from_crs("EPSG:4326", gc.UTM, always_xy=True).transform
    root = etree.parse("sarule.osm").getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    def mm(x, y):
        return (x - gc.X0) / (gc.X1 - gc.X0) * LX, (gc.Y1 - y) / (gc.Y1 - gc.Y0) * LY
    lab = [("SARULE", *mm(*to_utm(9.1673, 40.2291)), 11, "k"),
           ("Monte Gonare 1083 m\n(Santuario)", *mm(*to_utm(9.2028, 40.2276)), 9, "k")]
    fiumi = {}
    for w in root.iter("way"):
        t = {x.get("k"): x.get("v") for x in w.iter("tag")}
        pts = [to_utm(*nodes[n.get("ref")]) for n in w.iter("nd") if n.get("ref") in nodes]
        if len(pts) < 2:
            continue
        nm = t.get("name")
        if t.get("waterway") == "stream" and nm:
            fiumi.setdefault(nm, []).append(LineString(pts))
        if t.get("place") == "town" or nm == "Cimitero di Orani":
            c = LineString(pts).centroid
            lab.append(("verso Orani", *mm(c.x, c.y), 8, "#333"))
    for nm in ("Riu S'Iscala", "Riu Marre", "Riu Binzas", "Riu Bade Orani", "Riu Sa Padde", "Riu Su Cantaru"):
        if nm in fiumi:
            ln = max(fiumi[nm], key=lambda g: g.length)
            p = ln.interpolate(0.5, normalized=True)
            x, y = mm(p.x, p.y)
            if 8 < x < LX - 8 and 8 < y < LY - 8:
                lab.append((nm, x, y, 7, "#1d4a5a"))
    return lab


def pagina_mappa(pdf, rgb):
    fig = plt.figure(figsize=(420 / 25.4, 297 / 25.4))          # A3 orizzontale
    ox, oy = 20, 6
    ax = fig.add_axes([ox / 420, oy / 297, LX / 420, LY / 297])   # 1:1 in mm
    ax.imshow(rgb, extent=[0, LX, LY, 0], interpolation="lanczos")
    ax.set_xlim(0, LX); ax.set_ylim(LY, 0); ax.set_axis_off()
    for i, x in enumerate(np.arange(0, LX + 1, 50)):
        ax.axvline(x, color="k", lw=0.3, alpha=0.35)
    for j, y in enumerate(np.arange(0, LY + 1, 50)):
        ax.axhline(y, color="k", lw=0.3, alpha=0.35)
    for i, x in enumerate(np.arange(25, LX, 50)):
        for j, y in enumerate(np.arange(25, LY, 50)):
            ax.text(x - 23, y - 22, f"{'ABCDEFGH'[i]}{j+1}", fontsize=6, color="k", alpha=0.55)
    for txt, x, y, fs, col in etichette():
        ax.text(x, y, txt, fontsize=fs, color=col, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.65), ec="none"))
    ax.add_patch(Rectangle((0, 0), LX, LY, fill=False, lw=0.8, ec="k"))
    ax.annotate("N", xy=(LX - 12, 14), xytext=(LX - 12, 30), ha="center", fontsize=10, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color="k"))
    fig.text(ox / 420, (oy + LY + 2) / 297,
             "Sarule – Monte Gonare · mappa colori in SCALA 1:1 (38 × 28,5 cm) · quadretti da 5 cm · "
             "stampare in A3 a 'dimensioni reali / 100%'", fontsize=9)
    pdf.savefig(fig); plt.close(fig)


PASSI = [
    ("0. Preparazione", "Lava il pezzo con acqua e sapone, asciuga. Carteggia leggermente i bordi (grana 240). "
     "Dai una mano di primer spray grigio o bianco opaco (es. per modellismo/miniature) e lascia asciugare 2 ore."),
    ("1. Base per quote", "Con pennello piatto stendi le 4 tinte di fondo seguendo la mappa: fondovalle, colline, "
     "pascoli alti, roccia. Usa i quadretti A1…H6 per orientarti. Sfuma i confini tra una fascia e l'altra mescolando "
     "le due tinte sul pennello finché sono ancora umide: in natura non ci sono linee nette."),
    ("2. Velatura (wash)", "Diluisci terra d'ombra bruciata con molta acqua (1 parte colore, 6-8 acqua, una goccia di sapone). "
     "Passala su tutto il plastico e tampona l'eccesso con carta: il colore scuro resta nelle valli e nei solchi e il rilievo 'esce'."),
    ("3. Pennello asciutto (drybrush)", "Pennello vecchio, poco colore chiaro (ocra + bianco), scaricalo quasi tutto sulla carta "
     "e sfiora le creste e le cime: si illuminano gli spigoli. Sul Gonare usa grigio chiarissimo quasi bianco."),
    ("4. Boschi e macchia", "Con pennello tondo o spugnetta, tamponi di verde scuro (verde vescica + nero) dove nella foto aerea "
     "(visualizzatore 3D che ti ho mandato) vedi boschi: versanti nord e valli."),
    ("5. Dettagli", "Pennello sottile (n. 0 o 00): torrenti blu-verde scuro nei solchi, strade avorio sulle creste rialzate, "
     "paese in terracotta con qualche puntino più chiaro. Per le strade puoi usare anche un pennarello a vernice avorio a punta fine."),
    ("6. Protezione", "Spray trasparente OPACO (vernice finale per acrilici). Due mani leggere a 20 minuti di distanza."),
]


def pagina_legenda(pdf):
    fig = plt.figure(figsize=(420 / 25.4, 297 / 25.4))
    fig.text(0.04, 0.93, "Come dipingere il plastico – legenda e passaggi", fontsize=20, weight="bold")
    y = 0.86
    for nome, col, ric in ZONE:
        fig.patches.append(Rectangle((0.04, y - 0.012), 0.035, 0.035, transform=fig.transFigure,
                                     fc=np.array(col) / 255, ec="k", lw=0.4))
        fig.text(0.085, y + 0.009, nome, fontsize=11, weight="bold")
        fig.text(0.085, y - 0.012, ric, fontsize=9.5, color="#444")
        y -= 0.065
    fig.text(0.04, y - 0.01, "Materiale (~20-30 €): acrilici da belle arti o modellismo (bianco, nero, ocra gialla, terra di Siena,\n"
             "terra d'ombra bruciata, verde vescica, blu oltremare, rosso ossido), primer spray opaco, vernice\n"
             "spray opaca, pennelli piatto n.8, tondo n.4, n.0, un pennello vecchio per drybrush, carta da cucina.",
             fontsize=9.5, va="top", color="#333")
    yy = 0.86
    for tit, txt in PASSI:
        fig.text(0.50, yy, tit, fontsize=11.5, weight="bold")
        import textwrap
        body = "\n".join(textwrap.wrap(txt, 88))
        fig.text(0.50, yy - 0.012, body, fontsize=9.5, va="top", color="#333")
        yy -= 0.028 + 0.0195 * (body.count("\n") + 1)
    fig.text(0.50, max(yy - 0.01, 0.04), "Consiglio: prova i colori prima sul retro/bordo del pezzo. "
             "Meglio tinte spente e terrose che colori brillanti.", fontsize=9.5, style="italic")
    pdf.savefig(fig); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rgb, _ = mappa_zone()
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(f"{OUT}/tavola_pittura_A3.pdf") as pdf:
        pagina_mappa(pdf, rgb)
        pagina_legenda(pdf)
    # versione telefono
    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(rgb, extent=[0, LX, LY, 0]); ax.set_axis_off()
    for i, x in enumerate(np.arange(0, LX + 1, 50)):
        ax.axvline(x, color="k", lw=0.4, alpha=0.35)
    for j, y in enumerate(np.arange(0, LY + 1, 50)):
        ax.axhline(y, color="k", lw=0.4, alpha=0.35)
    for i, x in enumerate(np.arange(25, LX, 50)):
        for j, y in enumerate(np.arange(25, LY, 50)):
            ax.text(x - 23, y - 22, f"{'ABCDEFGH'[i]}{j+1}", fontsize=8, alpha=0.6)
    for txt, x, y, fs, col in etichette():
        ax.text(x, y, txt, fontsize=fs + 2, color=col, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.65), ec="none"))
    fig.savefig(f"{OUT}/mappa_zone.png"); plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
