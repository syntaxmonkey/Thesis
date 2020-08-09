from scipy.spatial import distance
import numpy as np
coords = np.array([(35.0456, -85.2672),
          (35.1174, -89.9711),
          (35.9728, -83.9422),
          (36.1667, -86.7833)])


coordsA = coords.copy()
coordsA[:,0] = coordsA[:,0] + 11
coordsA[:,1] = coordsA[:,1] + 5

b = np.vstack( ( coords, coordsA ) )
print(b)