import Bridson_ChainCode, Bridson_CreateMask, Bridson_sampling, Bridson_Delaunay
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
import math
import pylab
import Bridson_Common
import scipy
import Bridson_TriangulationDualGraph

'''
For the constructor, pass in the points.

Do we want to generate the Delaunay internally or receive from the outside?


'''
class MeshObject:
	# def __init__(self, mask, dradius, pointCount=0, indexLabel=0):
	def __init__(self, *args, **kwargs):

		self.indexLabel = str(kwargs.get('indexLabel'))
		if 'mask' in kwargs and 'dradius' in kwargs:
			Bridson_Common.logDebug(__name__, "Creating Mesh from Mask.")
			# Case we are provided with a Mask.
			mask = kwargs.get('mask')
			dradius = kwargs.get('dradius')
			self.GenMeshFromMask(mask, dradius)
		elif 'flatvertices' in kwargs and 'flatfaces' in kwargs and 'xrange' in kwargs and 'yrange' in kwargs:
			Bridson_Common.logDebug(__name__, "Creating Mesh from vertices and faces.")
			# Case we are provided the flattened vertices and triangles
			flatvertices = kwargs.get('flatvertices')
			flatfaces = kwargs.get('flatfaces')
			xrange = kwargs.get('xrange')
			yrange = kwargs.get('yrange')
			self.GenTriangulationFromOBJ(flatvertices, flatfaces, xrange, yrange)
		else:
			Bridson_Common.logDebug(__name__, "No enough parameters")


	def generateDualGraph(self):
		self.DualGraph = Bridson_TriangulationDualGraph.TriangulationDualGraph(self.points, self.triangulation.edges, self.triangulation.triangles, self.triangulation.neighbors)


	def colourTriangleCluster(self, index):
		# Find the triangle.
		centerTriangle = self.triangulation.triangles[index]
		# print("Center Triangle:", centerTriangle)
		v1, v2, v3 = centerTriangle

		edge1 = (v1,v2)
		edge2 = (v2, v3)
		edge3 = (v3, v1)
		Edge1 = self.DualGraph.EdgeHashMap.get(edge1)
		Edge2 = self.DualGraph.EdgeHashMap.get(edge2)
		Edge3 = self.DualGraph.EdgeHashMap.get(edge3)

		triangle1Index = Edge1.getNeighbour(index)
		self.colourTriangle(triangle1Index )
		print("Triangle1:", triangle1Index)

		triangle2Index = Edge2.getNeighbour(index)
		self.colourTriangle(triangle2Index )
		print("Triangle2:", triangle2Index)

		triangle3Index = Edge3.getNeighbour(index)
		self.colourTriangle(triangle3Index )
		print("Triangle3:", triangle3Index)

		self.colourTriangle(index, colour='w')


	def colourTriangle(self, triangleIndex, colour='magenta'):
		# print("Value of triangleIndex:", triangleIndex)
		if triangleIndex == None:
			return
		singleTriangle = np.array([self.triangulation.triangles[triangleIndex]])
		# singleTriangle = np.vstack((singleTriangle, self.triangulation.triangles[triangleIndex+1]))
		Bridson_Common.logDebug(__name__, singleTriangle)
		if Bridson_Common.invert:
			self.ax.triplot(self.points[:, 1], xrange - self.points[:, 0], singleTriangle, color=colour, linestyle='-', lw=1)
		else:
			self.ax.triplot(self.points[:, 0], self.points[:, 1], singleTriangle, color=colour, linestyle='-', lw=1)


	def DrawVerticalLines(self, density=Bridson_Common.density, linedensity=Bridson_Common.lineDotDensity):

		Bridson_Common.logDebug(__name__, "********** XLimit ***********", self.ax.get_xlim())
		Bridson_Common.logDebug(__name__, "********** YLimit ***********", self.ax.get_ylim())
		xlower, xupper = self.ax.get_xlim()
		ylower, yupper = self.ax.get_ylim()

		xincrement = abs(xupper - xlower) * density
		yincrement = abs(yupper - ylower) * linedensity

		pointCount = math.ceil(abs(yupper - ylower) / yincrement)
		columnCount = math.ceil(abs(xupper - xlower) / xincrement)
		Bridson_Common.logDebug(__name__, "************** PointCount ***********", pointCount)
		self.ax.set_ylim(ylower - 1, yupper + 1)

		dotPoints = []

		notFound = 0
		for j in range(columnCount+1):
			rowPoints = []
			for i in range(pointCount + 1):
				pointx, pointy = xlower + j * xincrement, ylower + yincrement * i
				# pointy, pointx = ylower + yincrement * i, xlower + j * xincrement
				if self.trifinder(pointx, pointy) > -1:  # How do we guarantee that the dots on the line are in order?
					rowPoints.append((pointx, pointy))
				else:
					notFound += 1
					#Bridson_Common.logDebug(__name__, "**** Point not found in trifinder *****", (pointx, pointy))
			if len(rowPoints) > 0:
				rowPoints = np.array(rowPoints)
				dotPoints.append(rowPoints)
		# Bridson_Common.logDebug(__name__, dotPoints)
		dotPoints = np.array(dotPoints)
		Bridson_Common.logDebug(__name__, "**** Points Not FOUND *****", notFound)


		colourArray = ['r', 'w', 'm']
		markerArray = ['o', '*', 's']
		index = 0

		for line in dotPoints:
			# https://kite.com/python/answers/how-to-plot-a-smooth-line-with-matplotlib-in-python
			# a_BSpline = np.array(scipy.interpolate.make_interp_spline(line[:, 0], line[:, 1]))
			# self.ax.plot(line[:, 0], a_BSpline, color='r')
			# self.ax.plot(line[:, 0], line[:, 1], color='r')
			colour = colourArray[ (index % 3) ]
			if Bridson_Common.drawDots:
				marker = markerArray[ (index % 3) ]
			else:
				marker = None
			# self.ax.plot(line[:, 0], line[:, 1], color='r', marker='o')
			self.ax.plot(line[:, 0], line[:, 1], color=colour , marker=marker )
			index += 1

		self.linePoints = dotPoints

	def DrawVerticalLinesSeededFrom(self, LineSeedObj, SourceMesh, density=Bridson_Common.density, linedensity=Bridson_Common.lineDotDensity):
		# The seedPoints were generated by Poisson Distribution using the original Mask.

		seedPoints = LineSeedObj.points
		# Convert all the seedPoints to this our coordinate system.
		seedPoints = self.TranslateBarycentricPoints( LineSeedObj, SourceMesh )

		Bridson_Common.logDebug(__name__, "********** XLimit ***********", self.ax.get_xlim())
		Bridson_Common.logDebug(__name__, "********** YLimit ***********", self.ax.get_ylim())
		xlower, xupper = self.ax.get_xlim()
		ylower, yupper = self.ax.get_ylim()

		xincrement = abs(xupper - xlower) * density
		yincrement = abs(yupper - ylower) * linedensity

		pointCount = math.ceil(abs(yupper - ylower) / yincrement)
		columnCount = math.ceil(abs(xupper - xlower) / xincrement)
		Bridson_Common.logDebug(__name__, "************** PointCount ***********", pointCount)
		self.ax.set_ylim(ylower - 1, yupper + 1)

		dotPoints = []
		bucketCount = int((xupper - xlower) / Bridson_Common.dradius )
		# table = [[range(bucketCount)], [[0]*bucketCount]]
		# print("************ Table: ", table)
		# buckets = {table[0][i]: table[1][i] for i in range(bucketCount)}
		buckets = {}
		for i in range(bucketCount):
			buckets[i] = 0

		print("************* Buckets: ", buckets)
		print("************* Bucket Count:", len(buckets))
		# Filter out the seedPoints and only allow one point per bucket point.
		filteredPoints = []

		for point in seedPoints:
			x, y = point
			bucketIndex = int((x - xlower) / Bridson_Common.dradius )
			try:
				if buckets[bucketIndex] == 0:
					if bucketIndex == 0 and buckets[bucketIndex+1] == 0:
						# Handle the case bucketIndex is zero.
						buckets[bucketIndex] = 1
						filteredPoints.append(point)
					elif bucketIndex == len(buckets) - 1 and buckets[bucketIndex - 2] == 0:
						# Handle the case bucketIndex is zero.
						buckets[bucketIndex] = 1
						filteredPoints.append(point)
					elif buckets[bucketIndex -1] == 0 and buckets[bucketIndex + 1] == 0:
						buckets[bucketIndex] = 1
						filteredPoints.append(point)
			except:
				print("BucketIndex Error")
				print("X:", x)
				print("xlower:", xlower)
				print("BucketIndex: ", bucketIndex)
				print("buckets:", buckets)
				exit(1)

		print("***************** FilteredPoints: ", filteredPoints)
		notFound = 0
		for point in filteredPoints:
			# print("DrawVertical Line seed Point: ", point)
			x, y = point
			j = x # Obtain the x value and use it for the line.
		# for j in range(columnCount+1):
			rowPoints = []
			for i in range(pointCount + 1):
				# pointx, pointy = xlower + j * xincrement, ylower + yincrement * i
				pointx, pointy = j, ylower + yincrement * i
				# pointy, pointx = ylower + yincrement * i, xlower + j * xincrement
				if self.trifinder(pointx, pointy) > -1:  # How do we guarantee that the dots on the line are in order?
					rowPoints.append((pointx, pointy))
				else:
					notFound += 1
					#Bridson_Common.logDebug(__name__, "**** Point not found in trifinder *****", (pointx, pointy))
			if len(rowPoints) > 0:
				rowPoints = np.array(rowPoints)
				dotPoints.append(rowPoints)
		# Bridson_Common.logDebug(__name__, dotPoints)
		dotPoints = np.array(dotPoints)
		Bridson_Common.logDebug(__name__, "**** Points Not FOUND *****", notFound)


		colourArray = ['r', 'w', 'm']
		markerArray = ['o', '*', 's']
		index = 0

		for line in dotPoints:
			# https://kite.com/python/answers/how-to-plot-a-smooth-line-with-matplotlib-in-python
			# a_BSpline = np.array(scipy.interpolate.make_interp_spline(line[:, 0], line[:, 1]))
			# self.ax.plot(line[:, 0], a_BSpline, color='r')
			# self.ax.plot(line[:, 0], line[:, 1], color='r')
			colour = colourArray[ (index % 3) ]
			if Bridson_Common.drawDots:
				marker = markerArray[ (index % 3) ]
			else:
				marker = None
			# self.ax.plot(line[:, 0], line[:, 1], color='r', marker='o')
			self.ax.plot(line[:, 0], line[:, 1], color=colour , marker=marker )
			index += 1

		self.linePoints = dotPoints




	def DrawHorizontalLines(self, density=Bridson_Common.density, linedensity=Bridson_Common.lineDotDensity):

		Bridson_Common.logDebug(__name__, "********** XLimit ***********", self.ax.get_xlim())
		Bridson_Common.logDebug(__name__, "********** YLimit ***********", self.ax.get_ylim())
		xlower, xupper = self.ax.get_xlim()
		ylower, yupper = self.ax.get_ylim()

		xincrement = abs(xupper - xlower) * linedensity
		yincrement = abs(yupper - ylower) * density
		pointCount = math.ceil(abs(xupper - xlower) / xincrement)
		rowCount = math.ceil(abs(yupper - ylower) / yincrement)
		Bridson_Common.logDebug(__name__, "************** PointCount ***********", pointCount)
		self.ax.set_ylim(ylower - 1, yupper + 1)

		dotPoints = []

		for j in range(rowCount+1):
			rowPoints = []
			for i in range(pointCount + 1):
				pointx, pointy = xlower + xincrement * i, ylower + j * yincrement
				if self.trifinder(pointx, pointy) > -1:
					rowPoints.append((pointx, pointy))
			if len(rowPoints) > 0:
				rowPoints = np.array(rowPoints)
				dotPoints.append(rowPoints)
		# Bridson_Common.logDebug(__name__, dotPoints)
		dotPoints = np.array(dotPoints)

		colourArray = ['r', 'w', 'm']
		markerArray = ['o', '*', 's']
		index = 0

		for line in dotPoints:
			# self.ax.plot(line[:, 0], line[:, 1], color='r')
			colour = colourArray[ (index % 3) ]
			if Bridson_Common.drawDots:
				marker = markerArray[ (index % 3) ]
			else:
				marker = None
			# self.ax.plot(line[:, 0], line[:, 1], color='r', marker='o')
			self.ax.plot(line[:, 0], line[:, 1], color=colour , marker=marker )
			index += 1

		self.linePoints = dotPoints


	def TranslateBarycentricPoints(self, otherMeshObj, originalMesh):
		newPoints = []
		newLine = []
		for point in otherMeshObj.points:
			x, y = point
			cartesian = Bridson_Common.convertAxesBarycentric(x, y, originalMesh.triangulation, self.triangulation,
			                                                  originalMesh.trifinder, originalMesh.points, self.points)
			newPoints.append( cartesian ) # Append points to create new line.

		newPoints = np.array( newPoints ) # Add to list of existing lines
		return newPoints

	def TransferLinePointsFromTarget(self, otherMeshObj):
		self.linePoints = []
		for otherLinePoints in otherMeshObj.linePoints:
			# Iterate through each line row.
			newLine = []
			for point in otherLinePoints:
				x, y = point
				cartesian = Bridson_Common.convertAxesBarycentric(x, y, otherMeshObj.triangulation, self.triangulation,
				                                                  otherMeshObj.trifinder, otherMeshObj.points, self.points)
				newLine.append( cartesian ) # Append points to create new line.

			self.linePoints.append( np.array(newLine) ) # Add to list of existing lines
		self.linePoints = np.array( self.linePoints ) # Convert line into numpy array.

		colourArray = ['r', 'w', 'm']
		markerArray = ['o', '*', 's']
		index = 0

		for line in self.linePoints:
			# self.ax.plot(line[:, 0], line[:, 1], color='ro')
			colour = colourArray[ (index % 3) ]
			if Bridson_Common.drawDots:
				marker = markerArray[ (index % 3) ]
			else:
				marker = None
			# self.ax.plot(line[:, 0], line[:, 1], color='r', marker='o')
			self.ax.plot(line[:, 0], line[:, 1], color=colour , marker=marker )
			index += 1
		return




	def GenTriangulationFromOBJ(self, flatvertices, flatfaces, xrange, yrange):
		# Bridson_Common.logDebug(__name__, flatvertices)
		# Bridson_Common.logDebug(__name__, flatfaces)
		Bridson_Common.logDebug(__name__, "Flat Triangulation from Mesh" + self.indexLabel)
		# if np.max(flatvertices < 10):

		# If normalizedUV is used.
		if Bridson_Common.normalizeUV:
			flatvertices[:, 0] *= xrange
			flatvertices[:, 1] *= yrange

		self.points = flatvertices[:]
		self.flatfaces = flatfaces[:]
		# This code adds a new face.  Trying to cause trifinder failure.
		# self.flatfaces = np.vstack((self.flatfaces, [640, 532, 532]))

		# Bridson_Common.logDebug(__name__, self.flatfaces)
		self.triangulation = mtri.Triangulation(self.points[:, 0], self.points[:, 1], flatfaces)
		try:
			self.trifinder = self.triangulation.get_trifinder()
			Bridson_Common.logDebug(__name__, "** Found trifinder ", self.indexLabel)
			self.trifinderGenerated = True
		except:
			Bridson_Common.logDebug(__name__, "Cannot trifinder ", self.indexLabel)
			self.trifinderGenerated = False
		# Bridson_Common.logDebug(__name__, "tri", self.triangulation)

		if Bridson_Common.displayMesh:
			self.fig = plt.figure()
			# self.ax = plt.axes()
			self.ax = plt.subplot(1, 1, 1, aspect=1)
			plt.title('Flat Triangulation ' + self.indexLabel)
			# plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
			# Plot the lines representing the mesh.
			if Bridson_Common.colourCodeMesh:
				# Based on code from https://stackoverflow.com/questions/28245327/filling-triangles-in-matplotlib-triplot-with-individual-colors
				print("Length of triangles", len(self.triangulation.triangles))
				colors = np.array( [ i % Bridson_Common.colourCount for i in range(len(self.triangulation.triangles)) ] )
				print("Length of colors", len(colors))
				# plt.tripcolor(self.points[:, 1], self.points[:, 0], self.triangulation.triangles.copy() , facecolors=colors, lw=0.5)
				plt.tripcolor(self.points[:, 0], self.points[:, 1], self.triangulation.triangles.copy(), facecolors=colors,
				              lw=0.5)
			else:
				if Bridson_Common.invert:
					plt.triplot(self.points[:, 1], xrange - self.points[:, 0], self.triangulation.triangles, 'b-', lw=0.5)
				else:
					plt.triplot(self.points[:, 1], self.points[:, 0], self.triangulation.triangles, 'b-', lw=0.5)

			# Display reference red triangles
			singleTriangle = np.array([self.triangulation.triangles[0]])
			singleTriangle = np.vstack((singleTriangle, self.triangulation.triangles[1]))
			# Bridson_Common.logDebug(__name__, singleTriangle)
			if Bridson_Common.invert:
				plt.triplot(self.points[:, 1], xrange - self.points[:, 0], singleTriangle, 'r-', lw=1)
			else:
				plt.triplot(self.points[:, 0], self.points[:, 1], singleTriangle, 'r-', lw=1)

			# Display reference green triangles
			singleTriangle = np.array([self.triangulation.triangles[-1]])
			singleTriangle = np.vstack((singleTriangle, self.triangulation.triangles[-2]))
			# Bridson_Common.logDebug(__name__, singleTriangle)
			if Bridson_Common.invert:
				plt.triplot(self.points[:, 1], xrange - self.points[:, 0], singleTriangle, 'g-', lw=1)
			else:
				plt.triplot(self.points[:, 0], self.points[:, 1], singleTriangle, 'g-', lw=1)

			# plt.triplot(self.points[:, 0], xrange - self.points[:, 1], self.triangulation.triangles)
			thismanager = pylab.get_current_fig_manager()
			thismanager.window.wm_geometry("+1300+560")

		# Plot the points on the border.
		# plt.plot(flatvertices[:, 1], xrange - flatvertices[:, 0], 'o')

		return




	def GenMeshFromMask(self, mask, dradius, pointCount=0):
		Bridson_Common.logDebug(__name__, "GenMeshFromMask")
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
		Bridson_Common.logDebug(__name__, np.shape(points))

		# Merge border with square perimeter.
		Bridson_Common.logDebug(__name__, "Points Shape: " ,  np.shape(points))
		Bridson_Common.logDebug(__name__, "Border Shape: " , np.shape(self.border))
		points = np.append(points, self.border, axis=0)

		# Generate all the sample points.
		points = Bridson_sampling.Bridson_sampling(width=xrange, height=yrange, radius=radius, existingPoints=points, mask=self.invertedMask)
		Bridson_Common.logDebug(__name__, np.shape(points))

		if len(self.invertedMask) > 0:
			points = self.filterOutPoints(points, self.invertedMask)

		self.points = points
		self.triangulation, self.points = Bridson_Delaunay.displayDelaunayMesh(points, radius, self.invertedMask, xrange)
		self.trifinder = self.triangulation.get_trifinder()

		Bridson_Common.logDebug(__name__, "tri" , self.triangulation)
		if Bridson_Common.displayMesh:
			self.fig = plt.figure()
			# self.ax = plt.axes()

			# Plot the mesh.
			self.ax = plt.subplot(1, 1, 1, aspect=1)
			plt.title('Generated Mesh Triangulation' + self.indexLabel)
			# plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
			# Plot the lines representing the mesh.
			# plt.triplot(points[:, 0], xrange - points[:, 1], self.triangulation.triangles)
			if Bridson_Common.colourCodeMesh:
				# Based on code from https://stackoverflow.com/questions/28245327/filling-triangles-in-matplotlib-triplot-with-individual-colors
				print("Length of triangles", len(self.triangulation.triangles))
				colors = np.array( [ i % Bridson_Common.colourCount for i in range(len(self.triangulation.triangles)) ] )
				print("Length of colors", len(colors))
				# plt.tripcolor(self.points[:, 1], self.points[:, 0], self.triangulation.triangles.copy() , facecolors=colors, lw=0.5)
				plt.tripcolor(self.points[:, 0], self.points[:, 1], self.triangulation.triangles.copy(), facecolors=colors,
				              lw=0.5)
			else:
				if Bridson_Common.invert:
					plt.triplot(points[:, 1], xrange - points[:, 0], self.triangulation.triangles, lw=0.5)
				else:
					plt.triplot(points[:, 1],  points[:, 0], self.triangulation.triangles, lw=0.5)

			# Plot red reference triangles
			singleTriangle = np.array([self.triangulation.triangles[0]])
			singleTriangle = np.vstack((singleTriangle, self.triangulation.triangles[1]))
			Bridson_Common.logDebug(__name__, singleTriangle)
			if Bridson_Common.invert:
				plt.triplot(self.points[:, 1], xrange - self.points[:, 0], singleTriangle, 'r-', lw=1)
			else:
				plt.triplot(self.points[:, 0], self.points[:, 1], singleTriangle, 'r-', lw=1)

			# Plot green reference triangles
			singleTriangle = np.array([self.triangulation.triangles[-1]])
			singleTriangle = np.vstack((singleTriangle, self.triangulation.triangles[-2]))
			Bridson_Common.logDebug(__name__, singleTriangle)
			if Bridson_Common.invert:
				plt.triplot(self.points[:, 1], xrange - self.points[:, 0], singleTriangle, 'g-', lw=1)
			else:
				plt.triplot(self.points[:, 0], self.points[:, 1], singleTriangle, 'g-', lw=1)



			thismanager = pylab.get_current_fig_manager()
			thismanager.window.wm_geometry("+1300+0")
		# Plot the points on the border.
		# plt.plot(points[:, 1], xrange - points[:, 0], 'o')
		self.generateSquareChainCode()
		self.trifinderGenerated = True
		return


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

		Bridson_Common.logDebug(__name__, 'Chain code length: ', len(self.squareChain))


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
	Bridson_Common.logDebug(__name__, "Done")
