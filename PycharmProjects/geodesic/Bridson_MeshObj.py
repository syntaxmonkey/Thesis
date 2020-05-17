import Bridson_ChainCode, Bridson_CreateMask, Bridson_sampling, Bridson_Delaunay
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
import math
import pylab


'''
For the constructor, pass in the points.

Do we want to generate the Delaunay internally or receive from the outside?


'''
class MeshObject:
	# def __init__(self, mask, dradius, pointCount=0, indexLabel=0):
	def __init__(self, *args, **kwargs):

		self.indexLabel = str(kwargs.get('indexLabel'))
		if 'mask' in kwargs and 'dradius' in kwargs:
			# Case we are provided with a Mask.
			mask = kwargs.get('mask')
			dradius = kwargs.get('dradius')
			self.GenMeshFromMask(mask, dradius)
		elif 'flatvertices' in kwargs and 'flatfaces' in kwargs and 'xrange' in kwargs and 'yrange' in kwargs:
			# Case we are provided the flattened vertices and triangles
			flatvertices = kwargs.get('flatvertices')
			flatfaces = kwargs.get('flatfaces')
			xrange = kwargs.get('xrange')
			yrange = kwargs.get('yrange')
			self.GenTriangulation(flatvertices, flatfaces, xrange, yrange)
		else:
			print("No enough parameters")


	def DrawVerticalLines(self):
		return

	def GenTriangulation(self, flatvertices, flatfaces, xrange, yrange):
		print("Flat Triangulation from Mesh" + self.indexLabel)
		flatvertices[:, 0] *= xrange
		flatvertices[:, 1] *= yrange
		self.flatvertices = flatvertices
		self.flatfaces = flatfaces
		self.triangulation = mtri.Triangulation(flatvertices[:, 0], flatvertices[:, 1], flatfaces)

		print("tri", self.triangulation)
		self.fig = plt.figure()
		plt.subplot(1, 1, 1, aspect=1)
		plt.title('Flat Triangulation ' + self.indexLabel)
		# plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
		# Plot the lines representing the mesh.
		plt.triplot(flatvertices[:, 1], xrange - flatvertices[:, 0], self.triangulation.triangles)
		thismanager = pylab.get_current_fig_manager()
		thismanager.window.wm_geometry("+1300+560")

		# Plot the points on the border.
		# plt.plot(flatvertices[:, 1], xrange - flatvertices[:, 0], 'o')

		return


	def GenMeshFromMask(self, mask, dradius, pointCount=0):
		print("GenMeshFromMask")
		xrange, yrange = np.shape(mask)
		self.count, self.chain, self.chainDirection, border = Bridson_ChainCode.generateChainCode(mask, rotate=False)
		self.border = Bridson_ChainCode.generateBorder(border, dradius)

		self.invertedMask = Bridson_CreateMask.InvertMask(mask)

		# Display figure of Inverted Mask
		# plt.figure()
		# plt.subplot(1, 1, 1, aspect=1)
		# plt.title('Inverted Mask')
		# plt.imshow(self.invertedMask)

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
		self.fig = plt.figure()
		plt.subplot(1, 1, 1, aspect=1)
		plt.title('Generated Mesh Triangulation' + self.indexLabel)
		# plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
		# Plot the lines representing the mesh.
		plt.triplot(points[:, 1], xrange - points[:, 0], self.triangulation.triangles)
		thismanager = pylab.get_current_fig_manager()
		thismanager.window.wm_geometry("+1300+0")
		# Plot the points on the border.
		# plt.plot(points[:, 1], xrange - points[:, 0], 'o')
		self.generateSquareChainCode()


	def generateSquareChainCode(self, corners=4):
		edges = len(self.chainDirection)
		shortEdges = edges - math.floor(edges/2)

		shortEdges = shortEdges if shortEdges % 2 == 0 else shortEdges - 1
		shortSideLength = int(shortEdges / 2)

		longEdges = edges - shortEdges
		if longEdges % 2 == 0:
			longSideA = longSideB = int(longEdges / 2)
		else:
			longSideA = math.floor( longEdges )
			longSideB = int(longEdges - longSideA)

		squareChain = []
		for i in range(longSideA - 1):
			squareChain.append(0)
		squareChain.append(90)

		for i in range(shortSideLength - 1):
			squareChain.append(0)
		squareChain.append(90)

		for i in range(longSideB - 1):
			squareChain.append(0)
		squareChain.append(90)

		for i in range(shortSideLength - 1):
			squareChain.append(0)
		squareChain.append(90)

		self.squareChain = squareChain
		Bridson_ChainCode.writeChainCodeFile('./', 'chaincode.txt', self.squareChain)

		print('Chain code length: ', len(self.squareChain))


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
