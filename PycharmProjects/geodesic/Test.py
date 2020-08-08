from scipy.spatial import distance
import numpy as np
coords = np.array([(35.0456, -85.2672),
          (35.1174, -89.9711),
          (35.9728, -83.9422),
          (36.1667, -86.7833)])

print(coords)
# coordsA = np.array([(35.0456+11, -85.2672+12),
#           (35.1174+9, -89.9711+5),
#           (35.9728+8, -83.9422+8),
#           (36.1667+10, -86.7833+13)])
coordsA = coords.copy()
coordsA[:,0] = coordsA[:,0] + 11
coordsA[:,1] = coordsA[:,1] + 5

a=distance.cdist(coords, coordsA, 'euclidean')
print(a)

print("Min Value:", a.min())
result = np.where( a == np.amin(a))
print("Result:", result)
listOfCordinates = list(zip(result[0], result[1]))
print("Min Index:", listOfCordinates )

minIndex = listOfCordinates[0]
print(a[minIndex[0], minIndex[1]])

# The x coordinate indicates the point in the first array while the y coordinate indicates the point in the second array.
# Once we have the coordinates, can we determine which points are the closest?  Does it matter?