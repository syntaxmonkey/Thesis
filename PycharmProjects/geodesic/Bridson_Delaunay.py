from scipy.spatial import Delaunay # For generating Delaunay - https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.html
import matplotlib.pyplot as plt
import Bridson_Common
import numpy as np
import matplotlib.tri as mtri
import math

def generateDelaunay(points, radius, mask):
    tri = Delaunay(points)  # Generate the triangles from the vertices.

    plt.figure()
    plt.subplot(1, 1, 1, aspect=1)
    plt.title('Display Delaunay')
    plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
    # plt.triplot(points[:, 0], points[:, 1], tri.triangles)
    plt.plot(points[:, 0], points[:, 1], 'o')

    tri = removeLongTriangles(points, tri, radius*radius, mask)

    return tri



def removeLongTriangles(points, tri, radius, mask):
	# Find Average Area.
	averageArea = Bridson_Common.findAverageArea(tri.simplices.copy(), points)
	print("Average Area:", averageArea)

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
			# if distance > radius:
			if distance > radius or isExteriorTriangle(points[triangle[currentIndex]], points[triangle[nextIndex]], mask):
			# if isExteriorTriangle(points[triangle[currentIndex]], points[triangle[nextIndex]], mask):
				Keep = False
				break
		area = Bridson_Common.findArea(points[triangle[0]], points[triangle[1]], points[triangle[2]])
		if area < averageArea / 3.0 or area > 3.0*averageArea:
			Keep = False
		if Keep:
			newTriangles.append(triangle)

	# print("New Triangle Shape:", np.shape(newTriangles))
	newTriangles = np.array(newTriangles)

	newTri = mtri.Triangulation(points[:,0], points[:,1], newTriangles)

	return newTri


def isExteriorTriangle(p1, p2, mask):
	# return False
	# Find the mid point of the lines of the triangles.  See if the midpoint of the line intersects with the mask.
	midx, midy = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
	# If the value is 255, then the line intersects with the mask and should be removed.
	if mask[midx, midy] == 255:
		return True
	return False


def displayDelaunayMesh(points, radius, mask, xrange):
	tri = generateDelaunay(points, radius, mask)

	# triangles = tri.simplices
	# for triangle in triangles:
	# 	print("Triangle:", triangle)

	print("tri", tri)
	plt.figure()
	plt.subplot(1, 1, 1, aspect=1)
	plt.title('Display Triangulation')
	# plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
	# Plot the lines representing the mesh.
	plt.triplot(points[:, 1], xrange-points[:, 0], tri.triangles)
	# Plot the points on the border.
	plt.plot(points[:, 1], xrange-points[:, 0], 'o')
	return tri
	# plt.show()

