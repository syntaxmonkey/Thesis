import Bridson_ChainCode, Bridson_CreateMask, Bridson_sampling, Bridson_Delaunay
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
import math
import pylab
import Bridson_Common
import scipy
import Bridson_TriangulationDualGraph
import sys
import SLIC
import pickle
import datetime
import Bridson_LineCalculations
from numba import jit, njit, jit_module
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
		# self.generateDualGraph() # Causes pickle issue
		self.linePoints = None


	def diagnosticExterior(self):
		# Runs the diagnostics for displaying exterior points and exterior edges.
		# Draw exterior edges
		for edge in self.DualGraph.exteriorEdges:
			exteriorEdges = []
			start, end = edge
			startPoint = self.DualGraph.points[start]
			endPoint = self.DualGraph.points[end]
			exteriorEdges.append(startPoint)
			exteriorEdges.append(endPoint)
			exteriorEdges = np.array(exteriorEdges)
			self.ax.plot(exteriorEdges[:, 0], exteriorEdges[:, 1], linestyle = '-' ,color='m', linewidth=2)

		# Draw exterior dots
		for pointIndex in self.DualGraph.exteriorPoints:
			point = self.DualGraph.points[ pointIndex ]
			self.ax.plot(point[0], point[1], color='b', markersize=8, marker='*')




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
			self.ax.triplot(self.points[:, 1], xrange - self.points[:, 0], singleTriangle, color=colour, linestyle='-', lw=3)
		else:
			self.ax.triplot(self.points[:, 0], self.points[:, 1], singleTriangle, color=colour, linestyle='-', lw=3)



	def findPointsMatchingAngle(self, angle=0):
		# Given the angle, find two points that fulfill the angle requirements.
		# Determine the dx, dy based on the angle.
		# print("Using angle: ", angle)
		startingSize = 50
		if math.isnan(angle): angle = 0
		dx, dy = Bridson_Common.calculateDirection( int( angle ) )

		dx, dy = dx * Bridson_Common.dradius * startingSize, dy * Bridson_Common.dradius * startingSize
		# print("dx, dy:", dx, dy)
		pointsFound = False

		# We need to start with dx, dy as large as possible.

		while pointsFound == False:
			exteriorPoints = self.DualGraph.exteriorPoints
			pointIndex = self.DualGraph.exteriorPoints[ np.random.randint(len(exteriorPoints)) ]
			x, y = self.DualGraph.points[pointIndex]
			# print("Testing x,y:", x, y)

			x2, y2 = x+dx, y+dy
			triangleIndex = self.trifinder(x2,y2)
			# print("Second trifinder index:", triangleIndex)
			if triangleIndex != -1:
				# have found valid point pair.
				# print("Valid point pair, p1 and p2:", (x,y), (x2,y2))
				return (x,y), (x2,y2)

			x3, y3 = x-dx, y-dy
			triangleIndex = self.trifinder(x3, y3)
			# print("Second trifinder index:", triangleIndex)

			if triangleIndex != -1:
				# valid point pair
				# print("Valid point pair, p1 and p3:", (x,y), (x3,y3))
				return (x,y), (x3,y3)

			dx, dy = dx*0.9, dy*0.9
			# print("Trying again.")


	def calculateAngle(self, sourceMeshObj, desiredAngle=0):
		# print("Initial angle:", desiredAngle)
		# Convert points to using Barycentric.
		p1, p2 = sourceMeshObj.findPointsMatchingAngle( angle=desiredAngle)
		# print("Source points:", p1, p2)
		cartesian1 = Bridson_Common.convertAxesBarycentric(p1[0], p1[1], sourceMeshObj.triangulation, self.triangulation,
		                                                  sourceMeshObj.trifinder, sourceMeshObj.points, self.points)
		cartesian2 = Bridson_Common.convertAxesBarycentric(p2[0], p2[1], sourceMeshObj.triangulation, self.triangulation,
		                                                  sourceMeshObj.trifinder, sourceMeshObj.points, self.points)
		# print("Target points:", cartesian1, cartesian2)
		if cartesian1 == None or cartesian2 == None:
			recoveredAngle = 90
		else:
			dx, dy = cartesian1[0] - cartesian2[0], cartesian1[1] - cartesian2[1]
			# print("New dx, dy:", dx, dy)
			recoveredAngle = Bridson_Common.determineAngle(dx, dy)
		# print("Recovered Angle:", recoveredAngle)
		return recoveredAngle


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


		# colourArray = ['r', 'w', 'm']
		markerArray = ['o', '*', 's']
		index = 0

		for line in dotPoints:
			# https://kite.com/python/answers/how-to-plot-a-smooth-line-with-matplotlib-in-python
			# a_BSpline = np.array(scipy.interpolate.make_interp_spline(line[:, 0], line[:, 1]))
			# self.ax.plot(line[:, 0], a_BSpline, color='r')
			# self.ax.plot(line[:, 0], line[:, 1], color='r')
			colour = Bridson_Common.colourArray[ (index % 3) ]
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


	# Draw vertical lines.  Use exterior points as line seeds.
	def DrawVerticalLinesExteriorSeed(self, density=Bridson_Common.density, linedensity=Bridson_Common.lineDotDensity):
		seedPoints = self.DualGraph.exteriorPoints.copy()

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

		notFound = 0

		for pointIndex in seedPoints:
			# print("DrawVertical Line seed Point: ", point)
			x, y = point = self.DualGraph.points[pointIndex]
			j = x # Obtain the x value and use it for the line.
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

	# Draw vertical lines.  Use exterior points as line seeds.
	def DrawVerticalLinesExteriorSeed2(self, density=Bridson_Common.density, linedensity=Bridson_Common.lineDotDensity):
		seedPoints = self.DualGraph.exteriorPoints.copy()

		notFound = 0
		dotPoints = []
		print("DrawVerticalLinesExteriorSeed2 seedPoints:", seedPoints)
		# for pointIndex in seedPoints[28:29]: # Interesting one.  Use Attempt = 17 and Attempt = 18.  Falls off trifinder at Attempt=18.
		# for pointIndex in seedPoints[23:24]: # Interesting one.  Stuck half way through.
		for pointIndex in seedPoints:
			triangleTraversal = []
			attempt = 0
		# for pointIndex in seedPoints:
			# print("DrawVertical Line seed Point: ", point)
			x, y = point = self.DualGraph.points[pointIndex]
			j = x # Obtain the x value and use it for the line.
			rowPoints = []
			rowPoints.append((x,y)) # Add the exterior point to the line.

			if False: # Display the first cluster of triangles
				triangleList = self.DualGraph.GetPointTriangleMembership(pointIndex)
				print("TriangleList:", triangleList)
				for triangleIndex in triangleList:
					self.colourTriangle(triangleIndex)


			intersection, edge, triangleIndex, direction = self.DualGraph.FindFirstIntersection( pointIndex, self.trifinder )
			triangleTraversal.append( triangleIndex )
			if intersection != None:
				rowPoints.append(intersection)
				# self.colourTriangle(triangleIndex)
				if Bridson_Common.highlightEdgeTriangle:
					self.colourTriangle(triangleIndex, colour='y')
				# Find next intersection.

				while True:
					nextIntersection, edge, triangleIndex, isFinalIntersection = self.DualGraph.FindNextIntersection( intersection, edge, triangleIndex, direction )
					# self.colourTriangle(triangleIndex)
					triangleTraversal.append(triangleIndex)
					if nextIntersection != None:
						if isFinalIntersection:
							Bridson_Common.logDebug(__name__,"Last Intersection:", nextIntersection)
							# Reduce the length of the final segment to ensure final intersection is in trifinder.
							nextIntersection = self.finalIntersectionReduction(rowPoints[-1], nextIntersection)
							Bridson_Common.logDebug(__name__,"Final Last Intersection adjusted:", nextIntersection)
						rowPoints.append( nextIntersection )
						intersection = nextIntersection
						# self.colourTriangle(triangleIndex)
					else:
						break

					if isFinalIntersection:
						break

					# 18 Attempts is where the graph flies into space.
					if attempt > 1000:
						break # Exit the while loop.
					attempt += 1

			Bridson_Common.logDebug(__name__,">>>>>>>>>>>>>>>>>>> Triangle Traversal:", triangleTraversal)
			if len(rowPoints) > 0:
				rowPoints = np.array(rowPoints)
				dotPoints.append(rowPoints)

		# Bridson_Common.logDebug(__name__, dotPoints)
		dotPoints = np.array(dotPoints)
		Bridson_Common.logDebug(__name__, "**** Points Not FOUND *****", notFound)

		colourArray = ['r', 'w', 'm']
		markerArray = ['o', '*', 's']
		index = 0

		if Bridson_Common.displayMesh:
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


	def rotateSeedPoints(self, angle=Bridson_Common.lineAngle):
		# rotates lines points by some angle.
		pass

	def findMiddleLineIndex(self, middle ):
		# This method will find the index of the middle line.
		minIndex = -1
		minDistance = 999999
		middleSet = [ middle ]

		for index in range(len(self.linePoints)):
			line = self.linePoints[index]
			# print("Bridson_MeshObj:findMiddleLineIndex Line:", line)
			location = Bridson_Common.findClosestIndex( line, middleSet)
			print("Location: ", location)
			foundPoint = line[ location[0] ]
			# print("Found point:", foundPoint[0])
			currentDistance = Bridson_Common.euclidean_distance(middle, foundPoint[0])
			if currentDistance < minDistance:
				minDistance = currentDistance
				minIndex = index
		print("MinDistance:", minDistance, "minIndex:", minIndex)
		self.middleLine = minIndex


	# Draw vertical lines.  Use exterior points as line seeds.
	def DrawAngleLinesExteriorSeed2(self, raster, density=Bridson_Common.density, linedensity=Bridson_Common.lineDotDensity, angle=Bridson_Common.lineAngle ):
		# Find the middle point of the raster.
		rasterShape = np.shape( raster )
		middle = [ int(rasterShape[0] / 2), int(rasterShape[1]/2) ]

		# print("Middle Point:", middle)
		# angle: 0 degrees goes north.  90 degrees goes east.  180 degrees goes south.  270 degrees goes west.
		dx, dy = Bridson_Common.calculateDirection(angle)
		# print("DrawAngleLinesExteriorSeed2 dx, dy:", dx, dy)

		seedPoints = self.DualGraph.exteriorPoints.copy()


		# print("DrawAngleLinesExteriorSeed2 seedPoints:", len(seedPoints))
		notFound = 0
		dotPoints = []
		# print("DrawAngleLinesExteriorSeed2 seedPoints:", seedPoints)
		# for pointIndex in seedPoints[28:29]: # Interesting one.  Use Attempt = 17 and Attempt = 18.  Falls off trifinder at Attempt=18.
		# for pointIndex in seedPoints[23:24]: # Interesting one.  Stuck half way through.
		# print("DrawAngleLinesExteriorSeed2 seedPoints:", seedPoints)
		seedPointCount = len(seedPoints)

		# Start a this point.
		indexOffset = int(seedPointCount / 3)

		# for pointIndex in seedPoints:
		for index in range(seedPointCount):
			pointIndex = seedPoints[ (index + indexOffset) % seedPointCount ]
			triangleTraversal = []
			attempt = 0
		# for pointIndex in seedPoints:
			# print("DrawVertical Line seed Point: ", point)
			x, y = point = self.DualGraph.points[pointIndex]
			# j = x # Obtain the x value and use it for the line.
			rowPoints = []
			rowPoints.append((x,y)) # Add the exterior point to the line.

			if False: # Display the first cluster of triangles
				triangleList = self.DualGraph.GetPointTriangleMembership(pointIndex)
				# print("TriangleList:", triangleList)
				for triangleIndex in triangleList:
					self.colourTriangle(triangleIndex)


			intersection, edge, triangleIndex, direction = self.DualGraph.FindFirstIntersection( pointIndex, self.trifinder, dx=dx, dy=dy )
			triangleTraversal.append( triangleIndex )
			if intersection != None:
				rowPoints.append(intersection)
				# print("DrawAngleLinesExteriorSeed2 found interscection:", intersection, triangleIndex)
				# self.colourTriangle(triangleIndex)
				if Bridson_Common.highlightEdgeTriangle:
					self.colourTriangle(triangleIndex, colour='y')
				# Find next intersection.

				while True:
					nextIntersection, edge, triangleIndex, isFinalIntersection = self.DualGraph.FindNextIntersection( intersection, edge, triangleIndex, direction, dx=dx, dy=dy )
					# self.colourTriangle(triangleIndex)
					triangleTraversal.append(triangleIndex)
					if nextIntersection != None:
						if isFinalIntersection:
							Bridson_Common.logDebug(__name__,"Last Intersection:", nextIntersection)
							# Reduce the length of the final segment to ensure final intersection is in trifinder.
							nextIntersection = self.finalIntersectionReduction(rowPoints[-1], nextIntersection)
							Bridson_Common.logDebug(__name__,"Final Last Intersection adjusted:", nextIntersection)
						rowPoints.append( nextIntersection )
						intersection = nextIntersection
						# self.colourTriangle(triangleIndex)
					else:
						break

					if isFinalIntersection:
						break

					# 18 Attempts is where the graph flies into space.
					if attempt > 1000:
						break # Exit the while loop.
					attempt += 1

			Bridson_Common.logDebug(__name__,">>>>>>>>>>>>>>>>>>> Triangle Traversal:", triangleTraversal)
			if len(rowPoints) > 0:
				# print("DrawAngleLinesExteriorSeed2 rowPoints not empty:", rowPoints)
				rowPoints = np.array(rowPoints)
				dotPoints.append(rowPoints)

		# Bridson_Common.logDebug(__name__, dotPoints)
		dotPoints = np.array(dotPoints)
		Bridson_Common.logDebug(__name__, "**** Points Not FOUND *****", notFound)

		colourArray = ['r', 'w', 'm']
		markerArray = ['o', '*', 's']
		index = 0
		# if Bridson_Common.displayMesh:
		if False:
			Bridson_Common.logDebug(__name__, "**** About to display lines *****", notFound)

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

		Bridson_Common.logDebug(__name__, "****   *****", notFound)

		# print("DrawAngleLinesExteriorSeed2 Shape of dotPoints:", np.shape(dotPoints))
		# print(dotPoints)
		if self.linePoints == None:
			self.linePoints = dotPoints
		else:
			self.linePoints = np.hstack( (self.linePoints, dotPoints) )  # Combine the two sets of lines.
		# print("Drawn LinePoints:", self.linePoints)
		Bridson_Common.logDebug(__name__, "**** Size of line points  *****", np.shape(self.linePoints))


		# self.findMiddleLineIndex( middle )
		# print("DrawAngleLinesExteriorSeed2 Size of line points:", np.shape(self.linePoints))



	def transferSeedPairs(self, sourceMeshObj):
		# Generate the line end point pairs.
		seedPoints = []
		dxdyValues = []

		for seedPoint in sourceMeshObj.seedPoints.copy():
			endpoint1, endpoint2 = seedPoint

			print("transferSeedPairs - source points", endpoint1, endpoint2)
			cartesian1 = self.convertBarycentricPointToSelf(endpoint1, sourceMeshObj) # HERE
			cartesian2 = self.convertBarycentricPointToSelf(endpoint2, sourceMeshObj)

			# print(">>      Bridson_MeshObj->generateSeedPairs: pt1:", endpoint1, "cartesian1:", cartesian1, "pt2:", endpoint2, "cartesian2:", cartesian2, "     <<")
			endpoints = (cartesian1, cartesian2)

			seedPoint, dxdy = self.convertPairsToSeedPoints(cartesian1, cartesian2)

			seedPoints.append( (cartesian1, cartesian2) )
			dxdyValues.append( dxdy )

		self.seedPoints = seedPoints
		self.dxdyValues = dxdyValues
		pass


	# This method will find the closest point on the exterior edge.
	def closestExteriorPoint(self, point):
		nearestDistance = 10000
		nearestPoint = []
		for edge in self.DualGraph.exteriorEdges:
			intersection = Bridson_LineCalculations.pnt2line(point, self.DualGraph.points[edge[0]], self.DualGraph.points[edge[-1]] )
			if intersection[0] < nearestDistance:
				nearestDistance = intersection[0]
				nearestPoint = intersection[1]


		return nearestDistance, nearestPoint


	# This version generates the seedPoints based on self.
	def generateSeedPairsSelf(self):
		# Generate the line end point pairs.
		seedPoints = []
		dxdyValues = []

		print("----------------- generateSeedPairsSelf START ---------------------")
		for line in self.croppedLinePoints:
			print("generateSeedPairSelf - line:", line)
			endpoint1 = line[0]
			endpoint2 = line[-1]


			inTrifinder = self.trifinder(endpoint1[0], endpoint1[1]) # Check to see if the first endpoint is in the trifinder.
			if inTrifinder == -1:
				nearestDistance, nearestPoint = self.closestExteriorPoint(endpoint1)
				newPoint = [nearestPoint[0], nearestPoint[1]]
				# line[0] = newPoint
				print("generateSeedPairsSelf --> Changing point", endpoint1, "to point", newPoint)
				endpoint1 = newPoint

			inTrifinder = self.trifinder(endpoint2[0], endpoint2[1])  # Check to see if the first endpoint is in the trifinder.
			if inTrifinder == -1:
				nearestDistance, nearestPoint = self.closestExteriorPoint(endpoint2)
				newPoint = [nearestPoint[0], nearestPoint[1]]
				# line[-1] = newPoint
				print("generateSeedPairsSelf --> Changing point", endpoint2, "to point", newPoint)
				endpoint2 = newPoint

			inTrifinder = self.trifinder(endpoint1[0], endpoint1[1])
			if inTrifinder != -1:
				print("generateSeedPairsSelf endpoint1 ", endpoint1, "found in triangle", inTrifinder)
			else:
				print("generateSeedPairsSelf endpoint1 ", endpoint1, "still not in triFinder")

			inTrifinder = self.trifinder(endpoint2[0], endpoint2[1])
			if inTrifinder != -1:
				print("generateSeedPairsSelf endpoint1 ", endpoint2, "found in triangle", inTrifinder)
			else:
				print("generateSeedPairsSelf endpoint1 ", endpoint2, "still not in triFinder")

			# cartesian1 = self.convertBarycentricPointToSelf(endpoint1, sourceMeshObj)
			# cartesian2 = self.convertBarycentricPointToSelf(endpoint2, sourceMeshObj)

			# print(">>      Bridson_MeshObj->generateSeedPairs: pt1:", endpoint1, "cartesian1:", cartesian1, "pt2:", endpoint2, "cartesian2:", cartesian2, "     <<")
			# endpoints = (cartesian1, cartesian2)

			# seedPoint, dxdy = self.convertPairsToSeedPoints(endpoint1, endpoint2)

			seedPoints.append( (endpoint1, endpoint2) )
		print("generateSeedPairsSelf - all end points:", seedPoints)
		self.seedPoints = seedPoints
		pass



	def convertBarycentricPointToSelf(self, xy, sourceMeshObj):
		x, y = xy
		cartesian = Bridson_Common.convertAxesBarycentric(x, y, sourceMeshObj.triangulation, self.triangulation,
		                                                  sourceMeshObj.trifinder, sourceMeshObj.points, self.points)
		return cartesian


	def convertPairsToSeedPoints(self, xy1, xy2):
		dx = xy1[0] - xy2[0]
		dy = xy1[1] - xy2[1]
		return xy1, (dx,dy)


	# Draw vertical lines.  Use exterior points as line seeds.
	def RedrawAngleLines(self, raster, otherMeshObj):
		##
		# Consider implementing BFS Tree: https://stackoverflow.com/questions/50537967/finding-nearest-neighbours-of-a-triangular-tesellation
		##

		print("RedrawAnglesLines start")
		# Find the middle point of the raster.
		rasterShape = np.shape( raster )
		middle = [ int(rasterShape[0] / 2), int(rasterShape[1]/2) ]

		## Need new method for generating seedPoints.  Utilizes existing points from croppedLinePoints from otherMeshObj.
		## Convert using barycentric from otherMeshObj to flatMeshObj
		## Calculate the dx, dy values and retain one starting point.
		seedPoints, dxdy = self.generateSeedPairsSelf( )

		notFound = 0
		dotPoints = []
		# print("DrawAngleLinesExteriorSeed2 seedPoints:", seedPoints)
		# for pointIndex in seedPoints[28:29]: # Interesting one.  Use Attempt = 17 and Attempt = 18.  Falls off trifinder at Attempt=18.
		# for pointIndex in seedPoints[23:24]: # Interesting one.  Stuck half way through.
		# print("DrawAngleLinesExteriorSeed2 seedPoints:", seedPoints)
		seedPointCount = len(seedPoints)

		# Start a this point.
		indexOffset = int(seedPointCount / 3)

		# for pointIndex in seedPoints:
		for index in range(seedPointCount):
			pointIndex = seedPoints[ (index + indexOffset) % seedPointCount ]

			triangleTraversal = []
			attempt = 0

			# x, y = point = self.DualGraph.points[pointIndex]
			point1, point2 = seedPoints[index]
			dx, dy = dxdy[index]




			# j = x # Obtain the x value and use it for the line.
			rowPoints = []
			rowPoints.append(point1) # Add the exterior point to the line.

			if False: # Display the first cluster of triangles
				triangleList = self.DualGraph.GetPointTriangleMembership(pointIndex)
				# print("TriangleList:", triangleList)
				for triangleIndex in triangleList:
					self.colourTriangle(triangleIndex)


			# Check to see if point1 and point2 are in the trifinder.  There are several cases:
			# 1. point 1 is outside trifinder.
			# 2. point 2 is outside trifinder
			# 3. point 1 is inside trifinder
			# 4. point 2 is inside trifinder
			# 5.

			# We pass in point1 and poin2.
			intersection, edge, triangleIndex, direction = self.DualGraph.FindFirstIntersectionXY( point1, point2, self.trifinder, dx=dx, dy=dy ) ## HERE



			triangleTraversal.append( triangleIndex )
			if intersection != None:
				rowPoints.append(intersection)
				# print("DrawAngleLinesExteriorSeed2 found interscection:", intersection, triangleIndex)
				# self.colourTriangle(triangleIndex)
				if Bridson_Common.highlightEdgeTriangle:
					self.colourTriangle(triangleIndex, colour='y')
				# Find next intersection.

				while True:
					nextIntersection, edge, triangleIndex, isFinalIntersection = self.DualGraph.FindNextIntersection( intersection, edge, triangleIndex, direction, dx=dx, dy=dy )
					# self.colourTriangle(triangleIndex)
					triangleTraversal.append(triangleIndex)
					if nextIntersection != None:
						if isFinalIntersection:
							Bridson_Common.logDebug(__name__,"Last Intersection:", nextIntersection)
							# Reduce the length of the final segment to ensure final intersection is in trifinder.
							nextIntersection = self.finalIntersectionReduction(rowPoints[-1], nextIntersection)
							Bridson_Common.logDebug(__name__,"Final Last Intersection adjusted:", nextIntersection)
						rowPoints.append( nextIntersection )
						intersection = nextIntersection
						# self.colourTriangle(triangleIndex)
					else:
						break

					if isFinalIntersection:
						break

					# 18 Attempts is where the graph flies into space.
					if attempt > 1000:
						break # Exit the while loop.
					attempt += 1

			Bridson_Common.logDebug(__name__,">>>>>>>>>>>>>>>>>>> Triangle Traversal:", triangleTraversal)
			if len(rowPoints) > 0:
				# print("DrawAngleLinesExteriorSeed2 rowPoints not empty:", rowPoints)
				rowPoints = np.array(rowPoints)
				dotPoints.append(rowPoints)

		# Bridson_Common.logDebug(__name__, dotPoints)
		dotPoints = np.array(dotPoints)
		Bridson_Common.logDebug(__name__, "**** Points Not FOUND *****", notFound)

		colourArray = ['r', 'w', 'm']
		markerArray = ['o', '*', 's']
		index = 0
		if Bridson_Common.displayMesh:
			Bridson_Common.logDebug(__name__, "**** About to display lines *****", notFound)

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

		Bridson_Common.logDebug(__name__, "****   *****", notFound)

		# print("DrawAngleLinesExteriorSeed2 Shape of dotPoints:", np.shape(dotPoints))
		# print(dotPoints)
		if self.linePoints == None:
			self.linePoints = dotPoints
		else:
			self.linePoints = np.hstack( (self.linePoints, dotPoints) )  # Combine the two sets of lines.
		# print("Drawn LinePoints:", self.linePoints)
		Bridson_Common.logDebug(__name__, "**** Size of line points  *****", np.shape(self.linePoints))



	# Draw lines using arbitrary distance based on the intensity.
	def DrawAngleLinesExteriorSeed3(self, raster, regionMap, index, intensity=128, density=Bridson_Common.density, linedensity=Bridson_Common.lineDotDensity, angle=Bridson_Common.lineAngle):
		intensityDistance = Bridson_Common.determineLineSpacing(intensity)
		intensityDistance = 2 # For testing.

		'''
		1. Determine the number of lines that we need to draw.
		a. Based on the intensity Distance and the angle, determine the number of points that will fit in the raster.
		b. We want the first point to be in the middle of the image.
		2. Draw first line in the middle of the image.
		3. Draw the subsequent lines in an alternative patter on either side of the first point.
		
		'''
		# angle: 0 degrees goes north.  90 degrees goes east.  180 degrees goes south.  270 degrees goes west.
		angle = (0) % 360
		dx, dy = Bridson_Common.calculateDirection(angle)  # The direction the lines will travel.
		print("Dx Dy:", dx, dy)
		#
		lineShiftx, lineShifty = dy, dx  # Have to swap x and y because numpy array.  Numpy utilizes row, column.

		seedPoints = []

		# seedPoints.append( [1, 15] ) # Because the raster utilizes numpy array, the x is actually the row index while the y is the column index.
		# Populate the first point.
		rasterShape = np.shape(raster)

		# Get the corners of region, not just of the enclosing raster.
		regionCoordinates = regionMap.get(index)
		topLeft, bottomRight = SLIC.calculateTopLeft(regionCoordinates)
		deltax = bottomRight[0] - topLeft[0] + 10
		deltay = bottomRight[1] - topLeft[1] + 10

		print("RasterShape:", rasterShape)
		print("deltax:", deltax, "deltay:", deltay)
		firstPoint = [rasterShape[0] / 2, rasterShape[1]/2]
		print("Raster first point value:", raster[int(firstPoint[0])][ int(firstPoint[1]) ])
		seedPoints.append( firstPoint )
		inRaster = True
		index = 1
		failedPointCheck = 0
		while inRaster:
			nextPoint = [firstPoint[0] + lineShiftx*intensityDistance*index, firstPoint[1] + lineShifty*intensityDistance*index]
			print("NextPoint:", nextPoint)
			if nextPoint[0] > rasterShape[0] or nextPoint[1] > rasterShape[1] or nextPoint[0] < 0 or nextPoint[1] < 0:
				failedPointCheck += 1
				print("Point off Raster")
				# if failedPointCheck == 2:
				break
			# elif raster[int(nextPoint[1])][ int(nextPoint[0])] == 255:
			else:
				print("Adding point to seed:", raster[int(nextPoint[1])][ int(nextPoint[0])])
				seedPoints.append( nextPoint )
			index += 1

		index = 1
		failedPointCheck = 0
		while inRaster:
			nextPoint = [firstPoint[0] + lineShiftx*intensityDistance*index*-1, firstPoint[1] + lineShifty*intensityDistance*index*-1]
			print("NextPoint:", nextPoint)
			if nextPoint[0] > rasterShape[0] or nextPoint[1] > rasterShape[1] or nextPoint[0] < 0 or nextPoint[1] < 0:
				failedPointCheck += 1
				print("Point off Raster")
				# if failedPointCheck == 2:
				break
			# elif raster[int(nextPoint[1])][ int(nextPoint[0])] == 255:
			else:
				print("Adding point to seed:", raster[int(nextPoint[1])][ int(nextPoint[0])])
				seedPoints.append( nextPoint )
			index += 1
		# seedPoints = np.array( seedPoints )

		'''
		1. Draw lines to the edge of the mesh.
		'''


		edgePoints = []
		for seedPoint in seedPoints:
			linePoints = []
			linePoints.append( seedPoint )
			edgePoint = [seedPoint[0] + dx*10, seedPoint[1]+ dy*10*-1]  # Because of the orientation of the image, we have to multiply with -1.
			linePoints.append( edgePoint )
			# edgePoints.append( edgePoint )
			edgePoints.append( linePoints )

		print("EdgePoints:", edgePoints)
		edgePoints = np.array( edgePoints )


		plt.figure()
		plt.imshow(raster)
		# plt.get
		if True:
			for linePoints in edgePoints:
				print("Plotting point:", linePoints)
				plt.plot(linePoints[:, 0], linePoints[:, 1], markersize=8, color='g', marker="+")

			for point in seedPoints:
				plt.plot(point[0], point[1],  markersize=8, color='r', marker="*")

		else:
			for point in seedPoints:
				plt.plot(point[0], point[1],  markersize=8, color='r', marker="*")



		plt.show()


		seedPoints = self.DualGraph.exteriorPoints.copy()

		# print("DrawAngleLinesExteriorSeed2 seedPoints:", len(seedPoints))
		notFound = 0
		dotPoints = []
		# print("DrawAngleLinesExteriorSeed2 seedPoints:", seedPoints)
		# for pointIndex in seedPoints[28:29]: # Interesting one.  Use Attempt = 17 and Attempt = 18.  Falls off trifinder at Attempt=18.
		# for pointIndex in seedPoints[23:24]: # Interesting one.  Stuck half way through.
		# print("DrawAngleLinesExteriorSeed2 seedPoints:", seedPoints)
		seedPointCount = len(seedPoints)

		# Start a this point.
		indexOffset = int(seedPointCount / 3)

		# for pointIndex in seedPoints:
		for index in range(seedPointCount):
			pointIndex = seedPoints[ (index + indexOffset) % seedPointCount ]
			triangleTraversal = []
			attempt = 0
		# for pointIndex in seedPoints:
			# print("DrawVertical Line seed Point: ", point)
			x, y = point = self.DualGraph.points[pointIndex]
			# j = x # Obtain the x value and use it for the line.
			rowPoints = []
			rowPoints.append((x,y)) # Add the exterior point to the line.

			if False: # Display the first cluster of triangles
				triangleList = self.DualGraph.GetPointTriangleMembership(pointIndex)
				# print("TriangleList:", triangleList)
				for triangleIndex in triangleList:
					self.colourTriangle(triangleIndex)


			intersection, edge, triangleIndex, direction = self.DualGraph.FindFirstIntersection( pointIndex, self.trifinder, dx=dx, dy=dy )
			triangleTraversal.append( triangleIndex )
			if intersection != None:
				rowPoints.append(intersection)
				# print("DrawAngleLinesExteriorSeed2 found interscection:", intersection, triangleIndex)
				# self.colourTriangle(triangleIndex)
				if Bridson_Common.highlightEdgeTriangle:
					self.colourTriangle(triangleIndex, colour='y')
				# Find next intersection.

				while True:
					nextIntersection, edge, triangleIndex, isFinalIntersection = self.DualGraph.FindNextIntersection( intersection, edge, triangleIndex, direction, dx=dx, dy=dy )
					# self.colourTriangle(triangleIndex)
					triangleTraversal.append(triangleIndex)
					if nextIntersection != None:
						if isFinalIntersection:
							Bridson_Common.logDebug(__name__,"Last Intersection:", nextIntersection)
							# Reduce the length of the final segment to ensure final intersection is in trifinder.
							nextIntersection = self.finalIntersectionReduction(rowPoints[-1], nextIntersection)
							Bridson_Common.logDebug(__name__,"Final Last Intersection adjusted:", nextIntersection)
						rowPoints.append( nextIntersection )
						intersection = nextIntersection
						# self.colourTriangle(triangleIndex)
					else:
						break

					if isFinalIntersection:
						break

					# 18 Attempts is where the graph flies into space.
					if attempt > 1000:
						break # Exit the while loop.
					attempt += 1

			Bridson_Common.logDebug(__name__,">>>>>>>>>>>>>>>>>>> Triangle Traversal:", triangleTraversal)
			if len(rowPoints) > 0:
				# print("DrawAngleLinesExteriorSeed2 rowPoints not empty:", rowPoints)
				rowPoints = np.array(rowPoints)
				dotPoints.append(rowPoints)

		# Bridson_Common.logDebug(__name__, dotPoints)
		dotPoints = np.array(dotPoints)
		Bridson_Common.logDebug(__name__, "**** Points Not FOUND *****", notFound)

		colourArray = ['r', 'w', 'm']
		markerArray = ['o', '*', 's']
		index = 0
		if Bridson_Common.displayMesh:
			Bridson_Common.logDebug(__name__, "**** About to display lines *****", notFound)

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

		Bridson_Common.logDebug(__name__, "****   *****", notFound)

		# print("DrawAngleLinesExteriorSeed2 Shape of dotPoints:", np.shape(dotPoints))
		# print(dotPoints)
		if self.linePoints == None:
			self.linePoints = dotPoints
		else:
			self.linePoints = np.hstack( (self.linePoints, dotPoints) )  # Combine the two sets of lines.
		# print("Drawn LinePoints:", self.linePoints)
		Bridson_Common.logDebug(__name__, "**** Size of line points  *****", np.shape(self.linePoints))

		# print("DrawAngleLinesExteriorSeed2 Size of line points:", np.shape(self.linePoints))


	def finalIntersectionReduction(self, previousPoint, currentPoint):
		# print("Current Last Point:", currentPoint)
		x, y = currentPoint
		# print("A Current Point:", x, y)
		while True:
			trianglesFound = self.trifinder(x , y) # This can return a list of
			# print("B Current Point:", x, y)
			# print("Trifinder result:", trianglesFound)
			# print("Trifinder type:", type(trianglesFound))
			# if ( isinstance(trianglesFound, np.ndarray) ):
				# print("Trifinder length:", len(trianglesFound))
			if (trianglesFound > -1).all():
				break
			# elif trianglesFound > -1:
			# 	break
			else:
				deltax = (x - previousPoint[0]) * 0.99
				deltay = (y - previousPoint[1]) * 0.99
				# print("Deltax:", deltax, "Deltay:", deltay)
				x = previousPoint[0] + deltax
				y = previousPoint[1] + deltay
		return (x,y)


	def clearAxes(self):
		if Bridson_Common.displayMesh:
			self.ax.cla()

	def rotateClockwise90(self, angle=90):
		# rotate the points
		# rotate the mesh
		# rotate the countour lines
		self.points = Bridson_Common.rotateClockwise90(self.points)
		Bridson_Common.logDebug(__name__,"Called rotateClockwise90")
		newLinePoints = []
		# print("PRE line Points:", self.linePoints)

		try:
			for line in self.linePoints:
				newLine = Bridson_Common.rotateClockwise90(line)
				newLinePoints.append( newLine )
		except Exception as e:
			print("Error in rotateClockwise90:", e)
			print("Error details:", sys.exc_info()[0])

		Bridson_Common.logDebug(__name__, "Rotated clockwise by 90")

		self.linePoints = np.array( newLinePoints )
		# print("POST line Points:", self.linePoints)
		self.clearAxes()
		Bridson_Common.logDebug(__name__, "Cleared Axes")
		self.displayMesh('Original Mesh rotated CW 90 - ')
		self.displayLines()

	def DrawHorizontalLinesExteriorSeed(self, density=Bridson_Common.density, linedensity=Bridson_Common.lineDotDensity):
		seedPoints = self.DualGraph.exteriorPoints.copy()

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

		print("SeedPoints from DualGraph:", seedPoints)
		for pointIndex in seedPoints:
			x, y = point = self.DualGraph.points[pointIndex]
			j = y
			rowPoints = []
			for i in range(pointCount + 1):
				pointx, pointy = xlower + xincrement * i, j
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


	def TranslateBarycentricPoints(self, targetMeshObj, sourceMeshObj):
		newPoints = []
		newLine = []
		for point in targetMeshObj.points:
			x, y = point
			cartesian = Bridson_Common.convertAxesBarycentric(x, y, sourceMeshObj.triangulation, self.triangulation,
			                                                  sourceMeshObj.trifinder, sourceMeshObj.points, self.points)
			newPoints.append( cartesian ) # Append points to create new line.

		newPoints = np.array( newPoints ) # Add to list of existing lines
		return newPoints



	def TransferLinePointsFromTarget(self, otherMeshObj):
		self.linePoints = []
		for otherLinePoints in otherMeshObj.linePoints:
			# Iterate through each line row.
			newLine = []
			for point in otherLinePoints:
				# print("Point:", point)
				Bridson_Common.logDebug(__name__, "Processing point:", point)
				x, y = point
				cartesian = Bridson_Common.convertAxesBarycentric(x, y, otherMeshObj.triangulation, self.triangulation,
				                                                  otherMeshObj.trifinder, otherMeshObj.points, self.points)
				Bridson_Common.logDebug(__name__, "Cartesian:", cartesian)
				# print("Type:", type(cartesian))
				if cartesian != None:  # If there is an error condition, then the cartesian value will be None.
					newLine.append( cartesian ) # Append points to create new line.

			# self.linePoints.append( np.array(newLine) ) # Add to list of existing lines
			self.linePoints.append( np.array(newLine) ) # Add to list of existing lines
		self.linePoints = np.array( self.linePoints ) # Convert line into numpy array.
		Bridson_Common.logDebug(__name__, "LinePoints:", np.shape(self.linePoints) )
		# print("LinePoints:", self.linePoints)

		colourArray = ['r', 'w', 'm']
		markerArray = ['o', '*', 's']
		index = 0

		if Bridson_Common.displayMesh:
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


	def genTrifinder(self):
		try:
			self.trifinder = self.triangulation.get_trifinder()
			Bridson_Common.logDebug(__name__, "** Found trifinder ", self.indexLabel)
			self.trifinderGenerated = True
		except:
			Bridson_Common.logDebug(__name__, "Cannot trifinder ", self.indexLabel)
			self.trifinderGenerated = False

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

		# if False: # TEST TEST TEST
		# 	try:
		# 		self.trifinder = self.triangulation.get_trifinder()
		# 		Bridson_Common.logDebug(__name__, "** Found trifinder ", self.indexLabel)
		# 		self.trifinderGenerated = True
		# 	except:
		# 		Bridson_Common.logDebug(__name__, "Cannot trifinder ", self.indexLabel)
		# 		self.trifinderGenerated = False

		# Bridson_Common.logDebug(__name__, "tri", self.triangulation)

		if Bridson_Common.displayMesh:
			self.fig = plt.figure()
			# self.ax = plt.axes()
			self.ax = plt.subplot(1, 1, 1, aspect=1)
			self.displayMesh('Flat Triangulation')
			thismanager = pylab.get_current_fig_manager()
			thismanager.window.wm_geometry("+1300+560")

		# Plot the points on the border.
		# plt.plot(flatvertices[:, 1], xrange - flatvertices[:, 0], 'o')

		return


	def displayMesh(self, label):
		if Bridson_Common.displayMesh:
			# self.fig = plt.figure()
			# self.ax = plt.axes()
			# self.ax = plt.subplot(1, 1, 1, aspect=1)
			self.ax.set_title(label + self.indexLabel)
			# plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
			# Plot the lines representing the mesh.
			if Bridson_Common.colourCodeMesh:
				# Based on code from https://stackoverflow.com/questions/28245327/filling-triangles-in-matplotlib-triplot-with-individual-colors
				print("Length of triangles", len(self.triangulation.triangles))
				colors = np.array( [ i % Bridson_Common.colourCount for i in range(len(self.triangulation.triangles)) ] )
				print("Length of colors", len(colors))
				# plt.tripcolor(self.points[:, 1], self.points[:, 0], self.triangulation.triangles.copy() , facecolors=colors, lw=0.5)
				self.ax.tripcolor(self.points[:, 0], self.points[:, 1], self.triangulation.triangles.copy(), facecolors=colors,
				              lw=0.5)
			else:
				if Bridson_Common.invert:
					self.ax.triplot(self.points[:, 1], xrange - self.points[:, 0], self.triangulation.triangles, 'b-', lw=0.5)
				else:
					self.ax.triplot(self.points[:, 1], self.points[:, 0], self.triangulation.triangles, 'b-', lw=0.5)

			if False:
				# Display reference red triangles
				singleTriangle = np.array([self.triangulation.triangles[0]])
				singleTriangle = np.vstack((singleTriangle, self.triangulation.triangles[1]))
				# Bridson_Common.logDebug(__name__, singleTriangle)
				if Bridson_Common.invert:
					self.ax.triplot(self.points[:, 1], xrange - self.points[:, 0], singleTriangle, 'r-', lw=1)
				else:
					self.ax.triplot(self.points[:, 0], self.points[:, 1], singleTriangle, 'r-', lw=1)

				# Display reference green triangles
				singleTriangle = np.array([self.triangulation.triangles[-1]])
				singleTriangle = np.vstack((singleTriangle, self.triangulation.triangles[-2]))
				# Bridson_Common.logDebug(__name__, singleTriangle)
				if Bridson_Common.invert:
					self.ax.triplot(self.points[:, 1], xrange - self.points[:, 0], singleTriangle, 'g-', lw=1)
				else:
					self.ax.triplot(self.points[:, 0], self.points[:, 1], singleTriangle, 'g-', lw=1)

			# # plt.triplot(self.points[:, 0], xrange - self.points[:, 1], self.triangulation.triangles)
			# thismanager = pylab.get_current_fig_manager()
			# thismanager.window.wm_geometry("+1300+560")

	def setCroppedLines(self, croppedLines):
		self.croppedLinePoints = croppedLines

	def setEdgeLinePoints(self, edgeLinePoints):
		self.edgeLinePoints = edgeLinePoints

	def displayLines(self):
		colourArray = ['r', 'w', 'm']
		markerArray = ['o', '*', 's']
		index = 0

		if Bridson_Common.displayMesh:
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


	def checkLinePoints(self):
		# print("LinePoints")
		print(np.shape(self.linePoints))
		if self.linePoints == None:
			print("linePoints not initialized yet.")
			return

		emptyLines = 0

		for line in self.linePoints:
			# print("Shape of line:", len(line))
			if len(line) == 0:
				emptyLines = emptyLines + 1
		# print("Empty Lines: ", emptyLines , "/", len(self.linePoints) )




	def GenMeshFromMask(self, mask, dradius, pointCount=0):
		Bridson_Common.logDebug(__name__, "GenMeshFromMask")
		xrange, yrange = np.shape(mask)
		if Bridson_Common.generateChaincode:
			self.count, self.chain, self.chainDirection, border = Bridson_ChainCode.generateChainCode(mask, rotate=False)
			self.border = Bridson_ChainCode.generateBorder(border, dradius)

		self.invertedMask = Bridson_CreateMask.InvertMask(mask)

		# Display figure of Inverted Mask
		# plt.figure()
		# plt.subplot(1, 1, 1, aspect=1)
		# plt.title('Inverted Mask')
		# plt.imshow(self.invertedMask)
		# print("GenMeshFromMask A")

		radius, pointCount = Bridson_sampling.calculateParameters(xrange, yrange, dradius, pointCount)
		# points = np.array([])

		points = Bridson_sampling.genSquarePerimeterPoints(xrange, yrange, radius=radius, pointCount=pointCount)

		if Bridson_Common.regionDynamicRadius:
			radius = Bridson_Common.dradius

		# Bridson_Common.logDebug(__name__, np.shape(points))
		# print("GenMeshFromMask B length of points:", len(points))
		# # Merge border with square perimeter.
		# Bridson_Common.logDebug(__name__, "Points Shape: " ,  np.shape(points))
		# Bridson_Common.logDebug(__name__, "Border Shape: " , np.shape(self.border))
		# points = np.append(points, self.border, axis=0)


		# Generate all the sample points.
		points = Bridson_sampling.Bridson_sampling(width=xrange, height=yrange, radius=radius, existingPoints=points, mask=self.invertedMask)
		Bridson_Common.logDebug(__name__, np.shape(points))

		# print("GenMeshFromMask before filtering:",len(points), points)
		# print("Inverted Mask:", self.invertedMask)
		if len(self.invertedMask) > 0:
			points = self.filterOutPoints(points, self.invertedMask)
		# print("GenMeshFromMask C points:", points)

		self.points = points
		self.triangulation, self.points = Bridson_Delaunay.displayDelaunayMesh(points, radius, self.invertedMask, xrange)
		# print("GenMeshFromMask 1: Pickling String for triangulation:", pickle.dumps(self.triangulation))
		# self.trifinder = self.triangulation.get_trifinder()  # Causes pickle issue
		# print("GenMeshFromMask 2: Pickling String for triangulation:", pickle.dumps(self.triangulation))
		# print("GenMeshFromMask D")
		Bridson_Common.logDebug(__name__, "tri" , self.triangulation)
		# print("GenMeshFromMask 3: Pickling String for triangulation:", pickle.dumps(self.triangulation))
		if Bridson_Common.displayMesh:
			self.fig = plt.figure()
			# self.ax = plt.axes()
			# print("GenMeshFromMask E")
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
			# print("GenMeshFromMask G")
			# Plot red reference triangles
			singleTriangle = np.array([self.triangulation.triangles[0]])
			singleTriangle = np.vstack((singleTriangle, self.triangulation.triangles[1]))
			Bridson_Common.logDebug(__name__, singleTriangle)
			if Bridson_Common.invert:
				plt.triplot(self.points[:, 1], xrange - self.points[:, 0], singleTriangle, 'r-', lw=1)
			else:
				plt.triplot(self.points[:, 0], self.points[:, 1], singleTriangle, 'r-', lw=1)
			# print("GenMeshFromMask H")
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
		# print("GenMeshFromMask I")
		# Plot the points on the border.
		# plt.plot(points[:, 1], xrange - points[:, 0], 'o')
		# print("GenMeshFromMask 4: Pickling String for triangulation:", pickle.dumps(self.triangulation))
		if Bridson_Common.generateChaincode:
			self.generateSquareChainCode()
		# self.trifinderGenerated = True  # TEST TEST TEST
		# print("GenMeshFromMask J")
		# print("GenMeshFromMask 5: Pickling String for triangulation:", pickle.dumps(self.triangulation))
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
		# Bridson_ChainCode.writeChainCodeFile('./', 'chaincode.txt', self.squareChain)
		Bridson_ChainCode.writeChainCodeFile('', Bridson_Common.chaincodefile, self.squareChain)

		Bridson_Common.logDebug(__name__, 'Chain code length: ', len(self.squareChain))


	def filterOutPoints(self, points, mask):
		# expects inverted mask.
		# Remove points that intersect with the mesh.
		# print("FilterOutPoints Shape of mask:", np.shape(mask))
		# print("FilterOutPoints Mask:", mask)
		newPoints = []
		for point in points:
			# Points that are '0' should be retained.
			if mask[ int(point[0]) , int(point[1])] == 0:
				newPoints.append(point)
				# print("Retaining point:", point)
			else:
				# print("Point:", int(point[0]) , int(point[1]), " Mask value:",mask[int(point[0]), int(point[1])] )
				mask[int(point[0]), int(point[1])] += 0.01

		# print("POST Mask:", mask)
		return np.array(newPoints)

jit_module(nopython=True)



if __name__ == "__main__":
	Bridson_Common.displayMesh = True
	dradius = 20
	xrange, yrange = 100, 100
	print("Generating letter o mask")
	mask = Bridson_CreateMask.genLetter(xrange, yrange, character='o')
	print("Generating Mesh Object")
	meshObject = MeshObject(mask=mask, dradius=dradius)
	meshObject.genTrifinder()
	# meshObj.trifinder = meshObj.triangulation.get_trifinder() # TEST TEST TEST
	print("Generating DualGraph")
	meshObject.generateDualGraph()  # TEST TEST TEST

	print("DrawAngleLinesExteriorSeed2")
	meshObject.DrawAngleLinesExteriorSeed2(mask) # Generates the lines points.
	meshObject.croppedLinePoints = meshObject.linePoints.copy() # Hack to create cropped Lines.
	# The seed pairs are created based ont he croppedLine Points.
	meshObject.generateSeedPairsSelf()

	print("Displaying mesh")
	meshObject.displayMesh("Test")
	# meshObject.displayLines()
	plt.show()
	Bridson_Common.logDebug(__name__, "Done")
