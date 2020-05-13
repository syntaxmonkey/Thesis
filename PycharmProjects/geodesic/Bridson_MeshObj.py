import Bridson_ChainCode, Bridson_CreateMask, Bridson_sampling, Bridson_Delaunay
import matplotlib.pyplot as plt
import numpy as np



'''
For the constructor, pass in the points.

Do we want to generate the Delaunay internally or receive from the outside?


'''
class MeshObject:
	def __init__(self, mask, dradius, pointCount=0):
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
		self.tri = Bridson_Delaunay.displayDelaunayMesh(points, radius, self.invertedMask, xrange)


	def filterOutPoints(self, points, mask):
		# expects inverted mask.
		# Remove points that intersect with the mesh.
		newPoints = []
		for point in points:
			# Points that are '0' should be retained.
			if mask[int(point[0]), int(point[1])] == 0:
				newPoints.append(point)

		return np.array(newPoints)

	def PopulateMesh(self):
		return





if __name__ == "__main__":
	dradius = 1.5
	xrange, yrange = 100, 100
	mask = Bridson_CreateMask.genLetter(xrange, yrange, character='Y')

	meshObject = MeshObject(mask, dradius)

	plt.show()
	print("Done")
