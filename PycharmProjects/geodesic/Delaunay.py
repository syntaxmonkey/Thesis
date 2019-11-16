import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
tri = Delaunay(points)


print(points)
print(points[tri.simplices])
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')

#print(points[:,0])

plt.show()