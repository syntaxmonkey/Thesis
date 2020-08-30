# # https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.structure_tensor
#
# # Tensor Structure testg
#
# from skimage.feature import structure_tensor
# import numpy as np
#
# square = np.zeros((5, 5))
# square[2, 2] = 1
#
# Arr, Arc, Acc = structure_tensor(square, sigma=0.1)
#
#
# print(square)
# print(Arr)
# print(Arc)
# print(Acc)
#
# # array([[0., 0., 0., 0., 0.],
# #        [0., 1., 0., 1., 0.],
# #        [0., 4., 0., 4., 0.],
# #        [0., 1., 0., 1., 0.],
# #        [0., 0., 0., 0., 0.]])

import numpy as np

points = np.array([[0, 0], [0, 1.1],
                   [1, 0], [1, 1]])
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation
tri = Delaunay(points)

print("Points:", tri.points)
print("Vertex neighbours:", tri.vertex_neighbor_vertices)
print("vertex_to_simple:", tri.vertex_to_simplex[3])
print("Neighbours:", tri.neighbors)
print("Neighbor of point 0:", tri.neighbors[0], tri.points[0])
print("Neighbor of point 1:", tri.neighbors[1])

