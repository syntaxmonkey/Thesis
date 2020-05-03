from scipy.spatial import Delaunay # For generating Delaunay - https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.html
import matplotlib.pyplot as plt
import Bridson_Common
import numpy as np
import matplotlib.tri as mtri
import math

def generateDelaunay(points, radius=0):
    tri = Delaunay(points)  # Generate the triangles from the vertices.
    # removeLongTriangles(tri, radius)

    plt.figure()
    plt.subplot(1, 1, 1, aspect=1)
    plt.title('Display Delaunay')
    plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
    # plt.triplot(points[:, 0], points[:, 1], tri.triangles)
    plt.plot(points[:, 0], points[:, 1], 'o')

    tri = removeLongTriangles(points, tri, radius*radius)

    return tri


def removeLongTriangles(points, tri, radius=0):
	triangles = tri.simplices.copy()
	newTriangles = []

	for triangle in triangles:
		Keep = True
		# Iterate through all 3 edges.  Ensure their euclidean distance is less than or equal to radius.
		# print("Triangle:", triangle[0])
		for i in range(3):
			currentIndex = i
			nextIndex = (currentIndex + 1) % 3
			# print("Indeces:", currentIndex, nextIndex)
			distance = Bridson_Common.euclidean_distance(points[triangle[currentIndex]], points[triangle[nextIndex]])
			# print("distance:", radius, distance)
			if distance > radius:
				Keep = False
				break
		if Keep:
			newTriangles.append(triangle)

	# print("New Triangle Shape:", np.shape(newTriangles))
	newTriangles = np.array(newTriangles)

	newTri = mtri.Triangulation(points[:,0], points[:,1], newTriangles)

	return newTri




def displayDelaunayMesh(points, radius=0):
	tri = generateDelaunay(points, radius)

	# triangles = tri.simplices
	# for triangle in triangles:
	# 	print("Triangle:", triangle)

	print("tri", tri)
	plt.figure()
	plt.subplot(1, 1, 1, aspect=1)
	plt.title('Display Triangulation')
	# plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
	plt.triplot(points[:, 0], points[:, 1], tri.triangles)
	plt.plot(points[:,0], points[:,1], 'o')
	return tri
	# plt.show()

