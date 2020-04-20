from scipy.spatial import Delaunay # For generating Delaunay - https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.html
import matplotlib.pyplot as plt

def generateDelaunay(points):
    tri = Delaunay(points)  # Generate the triangles from the vertices.
    return tri


def displayDelaunayMesh(points):
	tri = generateDelaunay(points)
	print("tri", tri)
	plt.figure()
	plt.subplot(1, 1, 1, aspect=1)
	plt.title('Display Delaunay')
	plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
	plt.plot(points[:,0], points[:,1], 'o')
	return tri
	# plt.show()

