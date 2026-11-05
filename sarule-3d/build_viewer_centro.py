"""Visualizzatore 3D del centro urbano (colori per vertice)."""
import json
from lxml import etree
from pyproj import Transformer
from shapely.geometry import Polygon
import centro_urbano as cu

t = Transformer.from_crs(4326, 32632, always_xy=True).transform
root = etree.parse('sarule.osm').getroot()
nodes = {n.get('id'): t(float(n.get('lon')), float(n.get('lat'))) for n in root.iter('node')}
def uv(x, y): return [round((x - cu.X0) / cu.W_M, 4), round((cu.Y1 - y) / cu.H_M, 4)]
labels = []
for w in root.iter('way'):
    tg = {x.get('k'): x.get('v') for x in w.iter('tag')}
    nm = tg.get('name')
    if nm in ('San Michele Arcangelo', 'Comune di Sarule', 'Santa Croce', 'Chiesa di Santa Croce', 'San Bernardino',
              'Nostra Signora del Rosario', "Sant'Antonio di Padova", 'Parco Comunale "Su Ponte"'):
        p = [nodes[n.get('ref')] for n in w.iter('nd') if n.get('ref') in nodes]
        if len(p) >= 3:
            c = Polygon(p).centroid
            u = uv(c.x, c.y)
            if 0 < u[0] < 1 and 0 < u[1] < 1:
                labels.append([nm.replace('Comune di Sarule', 'Municipio'), u])
h = open('viewer/template.html').read()
d = open('output/centro/viewer_centro.json').read()
# colori per vertice al posto della texture
h = h.replace("""const img = new Image();
img.src = 'data:image/jpeg;base64,' + D.tex_b64;
await img.decode();
const tex = new THREE.Texture(img); tex.colorSpace = THREE.SRGBColorSpace;
tex.anisotropy = renderer.capabilities.getMaxAnisotropy(); tex.needsUpdate = true;
const terrain = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: tex, roughness: 0.92, metalness: 0 }));""",
"""const crgb = Uint8Array.from(atob(D.c_b64), c => c.charCodeAt(0));
const col = new Float32Array(W * H * 3);
for (let k = 0; k < W * H * 3; k++) { const v = crgb[k] / 255; col[k] = v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }
geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
const terrain = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0 }));""")
assert 'vertexColors' in h
# etichette dinamiche
h = h.replace('<div class="tag" id="t1">Sarule</div>\n<div class="tag" id="t2">Monte Gonare · 1083 m</div>', '<div id="tags"></div>')
h = h.replace("const labels = [[document.getElementById('t1'), __P1__], [document.getElementById('t2'), __P2__]];",
 "const LBL = __LBL__; const labels = LBL.map(([t, p]) => { const e = document.createElement('div'); e.className = 'tag'; e.textContent = t; document.getElementById('tags').appendChild(e); return [e, p]; });")
h = h.replace('<h1>Sarule e Monte Gonare</h1>', '<h1>Sarule · centro urbano</h1>')
h = h.replace('Anteprima del plastico a colori · 40 × 30 cm · scala 1:16.000', '971 edifici · 36 × 29,5 cm · scala 1:3.000 · rilievo reale')
h = h.replace('min="1" max="3" step="0.1" value="1.5"', 'min="1" max="2.5" step="0.1" value="1"').replace('>1,5×<', '>1,0×<')
h = h.replace(' · Ortofoto © MASAF-AGEA, tutti i diritti riservati (solo anteprima) · Edifici © OpenStreetMap',
              ' · Edifici DBGT © Regione Sardegna e © OpenStreetMap · altezze stimate')
h = h.replace('v3d: [new THREE.Vector3(-70, 340, 470)', 'vzoom: [new THREE.Vector3(-60, 95, 120), new THREE.Vector3(-30, 35, 15)],\n  v3d: [new THREE.Vector3(-40, 330, 420)')
h = h.replace("vside: [new THREE.Vector3(30, 95, 480)", "vside: [new THREE.Vector3(20, 90, 420)")
h = h.replace("v.set(pos[3*k], pos[3*k+1] + 4, pos[3*k+2])", "v.set(pos[3*k], pos[3*k+1] + 6, pos[3*k+2])")
h = (h.replace('__DATA__', d).replace('__BASE__', '4.0').replace('__EX__', '1.0')
      .replace('__LBL__', json.dumps(labels, ensure_ascii=False)))
open('output/centro/sarule_centro_3d.html', 'w').write(h)
print(len(labels), 'etichette', labels)
