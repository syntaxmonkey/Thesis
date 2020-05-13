import Bridson_ChainCode, Bridson_CreateMask, Bridson_sampling, Bridson_Delaunay
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri


'''
For the constructor, pass in the points.

Do we want to generate the Delaunay internally or receive from the outside?


'''
class MeshObject:
	# def __init__(self, mask, dradius, pointCount=0):
	def __init__(self, *args, **kwargs):

		if 'mask' in kwargs and 'dradius' in kwargs:
			mask = kwargs.get('mask')
			dradius = kwargs.get('dradius')
			self.GenMeshFromMask(mask, dradius)
		elif 'flatvertices' in kwargs and 'flatfaces' in kwargs and 'xrange' in kwargs and 'yrange' in kwargs:
			flatvertices = kwargs.get('flatvertices')
			flatfaces = kwargs.get('flatfaces')
			xrange = kwargs.get('xrange')
			yrange = kwargs.get('yrange')
			self.GenTriangulation(flatvertices, flatfaces, xrange, yrange)
		else:
			print("No enough parameters")


	def GenTriangulation(self, flatvertices, flatfaces, xrange, yrange):
		print("Flat Triangulation")
		flatvertices[:, 0] *= xrange
		flatvertices[:, 1] *= yrange
		self.flatvertices = flatvertices
		self.flatfaces = flatfaces
		self.triangulation = mtri.Triangulation(flatvertices[:, 0], flatvertices[:, 1], flatfaces)

		print("tri", self.triangulation)
		plt.figure()
		plt.subplot(1, 1, 1, aspect=1)
		plt.title('Display Triangulation')
		# plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
		# Plot the lines representing the mesh.
		plt.triplot(flatvertices[:, 1], xrange - flatvertices[:, 0], self.triangulation.triangles)
		# Plot the points on the border.
		plt.plot(flatvertices[:, 1], xrange - flatvertices[:, 0], 'o')

		return


	def GenMeshFromMask(self, mask, dradius, pointCount=0):
		print("GenMeshFromMask")
		xrange, yrange = np.shape(mask)
		self.count, self.chain, self.chainDirection, border = Bridson_ChainCode.generateChainCode(mask, rotate=False)
		self.border = Bridson_ChainCode.generateBorder(border, dradius)

		self.invertedMask = Bridson_CreateMask.InvertMask(mask)

		plt.figure()
		plt.subplot(1, 1, 1, aspect=1)
		plt.title('Inverted Mask')
		plt.imshow(self.invertedMask)

		radius, pointCount = Bridson_sampling.calculateParameters(xrange, yrange, dradius, pointCount)

		points = Bridson_sampling.genSquarePerimeterPoints(xrange, yrange, radius=radius, pointCount=pointCount)
		print(np.shape(points))

		# Merge border with square perimeter.
		points = np.append(points, self.border, axis=0)

		# Generate all the sample points.
		points = Bridson_sampling.Bridson_sampling(width=xrange, height=yrange, radius=radius, existingPoints=points, mask=self.invertedMask)
		print(np.shape(points))

		if len(self.invertedMask) > 0:
			points = self.filterOutPoints(points, self.invertedMask)

		self.points = points
		self.triangulation = Bridson_Delaunay.displayDelaunayMesh(points, radius, self.invertedMask, xrange)

		print("tri", self.triangulation)
		plt.figure()
		plt.subplot(1, 1, 1, aspect=1)
		plt.title('Generated Mesh Triangulation')
		# plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
		# Plot the lines representing the mesh.
		plt.triplot(points[:, 1], xrange - points[:, 0], self.triangulation.triangles)
		# Plot the points on the border.
		plt.plot(points[:, 1], xrange - points[:, 0], 'o')




	def filterOutPoints(self, points, mask):
		# expects inverted mask.
		# Remove points that intersect with the mesh.
		newPoints = []
		for point in points:
			# Points that are '0' should be retained.
			if mask[int(point[0]), int(point[1])] == 0:
				newPoints.append(point)

		return np.array(newPoints)





if __name__ == "__main__":
	dradius = 1.5
	xrange, yrange = 100, 100
	mask = Bridson_CreateMask.genLetter(xrange, yrange, character='Y')

	meshObject = MeshObject(mask, dradius)

	plt.show()
	print("Done")
