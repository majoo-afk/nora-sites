"""Inserisce i dati nel template del visualizzatore web."""
import json
from pyproj import Transformer
X0, Y0, X1, Y1 = 512539, 4450513, 518939, 4455313
t = Transformer.from_crs(4326, 32632, always_xy=True).transform
def uvp(lat, lon):
    x, y = t(lon, lat); return [round((x - X0) / (X1 - X0), 4), round((Y1 - y) / (Y1 - Y0), 4)]
h = open('viewer/template.html').read()
h = (h.replace('__DATA__', open('output/colori/viewer_data.json').read())
      .replace('__BASE__', '4.0').replace('__EX__', '1.5')
      .replace('__P1__', json.dumps(uvp(40.2291, 9.1673)))
      .replace('__P2__', json.dumps(uvp(40.2276, 9.2028))))
open('output/colori/sarule_gonare_3d.html', 'w').write(h)
