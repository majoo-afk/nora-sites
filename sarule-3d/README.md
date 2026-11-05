# Plastico 3D di Sarule e Monte Gonare

Modello stampabile in 3D del paese di Sarule (NU) e delle campagne fino al
Monte Gonare, generato da dati aperti.

| | |
|---|---|
| Area reale | 6,4 × 4,8 km (centro 40.2265 N, 9.1850 E) |
| Dimensioni stampa | 200 × 150 × 36 mm, scala 1:32.000 |
| Esagerazione verticale | 1,5× |
| Quote | 392 – 1079 m s.l.m. |
| Mesh | ~621k triangoli, solido chiuso (watertight) |
| Filamento stimato | ~190 g PLA (2 pareti, riempimento 15%) |

File: `output/sarule_gonare_stl.zip` (STL), anteprime in `output/*.png`.

## Rigenerare / cambiare area

```bash
pip install rasterio trimesh shapely scipy pyproj networkx matplotlib lxml
# dati OSM dell'area (bbox lon_min,lat_min,lon_max,lat_max)
curl -o sarule.osm "https://api.openstreetmap.org/api/0.6/map?bbox=9.145,40.205,9.225,40.248"
python3 genera_modello.py --osm sarule.osm --out output \
  --centro 40.2265,9.1850 --larghezza-km 6.4 --altezza-km 4.8 \
  --stampa-mm 200 --esagerazione 1.5
python3 render_stampa.py
```

Note: gli edifici singoli a questa scala (10 m = 0,3 mm) non sono stampabili,
quindi sono fusi in isolati rialzati di 1,2 mm; le strade principali sono
rialzate di 0,4 mm.

## Fonti e licenze
- Altimetria: Copernicus DEM GLO-30 © DLR e.V. 2010-2014 / © Airbus 2014-2018, fornito da ESA (Copernicus).
- Edifici e strade: © OpenStreetMap contributors, licenza ODbL.

## Versione a colori (anteprima)

`genera_colori.py` crea un plastico 40 × 30 cm (scala 1:16.000) con:
- DTM LiDAR 1 m della Regione Sardegna (copre il paese, ~31% dell'area) fuso con il DTM 10 m regionale;
- ortofoto 2022 (WMS Regione Sardegna, `raster:Mosaico_2022_GB`) come texture;
- edifici OSM in rilievo con altezza stimata.

```bash
python3 genera_colori.py     # -> output/colori/sarule_gonare_colori.glb (+ texture, dati viewer)
python3 build_viewer.py      # -> output/colori/sarule_gonare_3d.html (visualizzatore three.js)
```

**Attenzione:** le ortofoto 2019/2022 sono © MASAF-AGEA, *tutti i diritti riservati*
(filigrana visibile). Vanno usate solo come anteprima privata: per stampare o
pubblicare serve l'autorizzazione AGEA, oppure un'immagine con licenza libera
(es. Sentinel-2 cloudless 2016, CC BY 4.0, 10 m) o foto da drone proprie.
Per questo `output/colori/` non è versionato.
