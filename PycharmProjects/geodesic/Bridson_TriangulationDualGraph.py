import Bridson_Common
import Bridson_Main
import SLIC
import matplotlib.pyplot as plt
import pylab
import numpy as np
import math

class TriangulationDualGraph:

	'''
		When we create the TriangulationDualGraph, what are the usecases?
		1. Create TriangulationDualGraph based on OBJ file.
		2. Create TriangulationDualGraph based on vertices and facets.
		3. Create TriangulationDualGraph based on Triangulation.

		Expecting inputs:
			points
			Edges
			Triangles
			Neighbours
	'''

	class DualEdge:
		# An edge can belong to up to 2 different triangles.
		def __init__(self, start, end):
			self.start = start
			self.end = end
			self.triangleIndeces = []
			self.triangles = []
			self.vertices = (start, end)

		def addTriangle(self, triangleIndex, Triangle):
			self.triangleIndeces.append( triangleIndex )
			self.triangles.append( Triangle )

		def clearTriangles(self):
			self.triangles = []

		def getNeighbour(self, triangleIndex):
			# print("Searching for index:", triangleIndex)
			# print("Triangle List:", self.triangleIndeces)
			# Case - there is only one Triangle.
			if len(self.triangleIndeces) == 1:
				return None
			else:
				return self.triangleIndeces[ 1 - self.triangleIndeces.index(triangleIndex)]
			# Case - there are two neighbours.

	class DualTriangle:
		# A triangle has 3 edges.
		# Also have up to 3 neighbours.
		def __init__(self, triangleIndex):
			self.triangleIndex = triangleIndex
			self.edges = []
			self.neighbours = []

		def addEdge(self, edge):
			self.edges.append( edge )

		def addNeighbour(self, neighbour):
			self.neighbours.append(neighbour)

		def setVertices(self, vertices, points):
			self.vertices = vertices.copy()
			# At this point calculate the max height of the triangle.
			p1 = points[ self.vertices[0] ]
			p2 = points[ self.vertices[1] ]
			p3 = points[ self.vertices[2] ]
			self.minHeight, self.maxHeight = Bridson_Common.findMinMaxTriangleHeight(p1, p2, p3)



	def __init__(self, points, Edges, Triangles, Neighbours):

		# if 'points' in kwargs and 'Edges' in kwargs and 'Triangles' in kwargs and 'Neighbours' in kwargs:
		# 	points = kwargs.get('points')
		# 	Edges = kwargs.get('Edges')
		# 	Triangles = kwargs.get('Triangles')
		# 	Neighbours = kwargs.get('Neighbours')
		self.points = points.copy()
		self.Edges = Edges.copy()
		self.Triangles = Triangles.copy()
		self.Neighbours = Neighbours.copy()
		self.CreateGraph()
		# else:
		# 	Exception("TriangulationDualGraph creator - Not enough parameters.")


	def CreateGraph(self):
		'''

		'''
		# Edge(start, end) -> DualEdge: EdgeHashMap
		# Also populate DualEdge
		self.GenerateEdgePointsToEdgeMap()

		# Add PointIndex -> DualEdge: PointToEdgeHashMap
		self.GeneratePointHashMaps()

		# (v1,v2,v3) -> DualTriangle: TriangleHashMap
		# Also DualEdge -> DualTriangle
		self.GenerateVertexToTriangleMap()

		# Generate list of exterior point indeces: exteriorPoints
		self.GenerateExteriorPoints()
		# Generate list of exterior point -> exterior edge map.
		self.GenerateExteriorPointToEdgesMap()



	def GetMaxTriangleHeightFromPoint(self, pointIndex):
		'''
		1. Get list of associated triangles.
		:param pointIndex:
		:return:
		'''

		relatedTriangleList = self.GetPointTriangleMembership( pointIndex ) # This will be a list of all the triangles that are associated with the pointIndex.
		Bridson_Common.logDebug(__name__,"Related Triangles:", relatedTriangleList)

		# Iterate through the triangles and determine their respective max heights.
		triangleHeights = []
		for triangleIndex in relatedTriangleList:
			triangleVertices = self.Triangles[ triangleIndex ]
			v1,v2,v3 = triangleVertices
			# Bridson_Common.logDebug(__name__,"Triangle Vertices:", v1, v2, v3)
			dualTriangle = self.TriangleHashMap[ (v1,v2,v3) ]
			triangleHeights.append( dualTriangle.maxHeight )
			triangleHeights.append(dualTriangle.minHeight)

		Bridson_Common.logDebug(__name__,"Triangle Heights:", triangleHeights )
		maxHeight = np.sort(triangleHeights)[-1]
		minHeight = np.sort(triangleHeights)[0]
		return minHeight, maxHeight, relatedTriangleList


	def GetMaxTriangleHeightFromEdge(self, edge, triangleIndex):
		'''
		1. Get list of associated triangles.
		:param pointIndex:
		:return:
		'''

		dualEdge = self.EdgeHashMap[ edge ]
		relatedTriangleList = dualEdge.triangleIndeces.copy()
		Bridson_Common.logDebug(__name__,"GetMaxTriangleHeightFromEdge Related Triangles:", relatedTriangleList)
		# relatedTriangleList = self.GetPointTriangleMembership( pointIndex ) # This will be a list of all the triangles that are associated with the pointIndex.
		relatedTriangleList.pop( relatedTriangleList.index( triangleIndex ) )

		if len(relatedTriangleList) == 0:
			return None, None, None

		Bridson_Common.logDebug(__name__,"GetMaxTriangleHeightFromEdge Related Triangles:", relatedTriangleList)

		# Iterate through the triangles and determine their respective max heights.
		triangleHeights = []
		for triangleIndex in relatedTriangleList:
			triangleVertices = self.Triangles[ triangleIndex ]
			v1,v2,v3 = triangleVertices
			Bridson_Common.logDebug(__name__,"GetMaxTriangleHeightFromEdge Triangle Vertices:", v1, v2, v3)
			dualTriangle = self.TriangleHashMap[ (v1,v2,v3) ]
			triangleHeights.append( dualTriangle.maxHeight )
			triangleHeights.append( dualTriangle.minHeight )

		Bridson_Common.logDebug(__name__,"GetMaxTriangleHeightFromEdge Triangle Heights:", triangleHeights )
		maxHeight = np.sort(triangleHeights)[-1]
		minHeight = np.sort(triangleHeights)[0]
		return minHeight, maxHeight, relatedTriangleList



	def GenerateExteriorPointToEdgesMap(self):
		exteriorEdges = []

		for pointIndex in self.exteriorPoints:
			dualEdgeList = self.PointToEdgeHashMap.get( pointIndex ) # Returns list of DualEdge objects that the point belongs to.
			for edge in dualEdgeList:
				start, end = edge
				if start in self.exteriorPoints and end in self.exteriorPoints:
					dualEdge = self.EdgeHashMap[ (start,end) ]
					if len( dualEdge.triangleIndeces ) == 1:
						# Also need to check if the edge has only one triangle membership.
						exteriorEdges.append( edge )

		self.exteriorEdges = exteriorEdges


	def sortExteriorPoints(self):

		tempPoints = []
		for point in self.points:
			tempPoints.append( list(point))
		print("Points:", tempPoints)

		# At this point, the self.exteriorPoints is a list of point indeces.
		actualPoints = []
		for pointIndex in self.exteriorPoints:
			actualPoints.append( tempPoints[ pointIndex ].copy() )
		# actualPoints = np.array( actualPoints, dtype=[('x', np.float), ('y', np.float)] )
		actualPoints = np.array( actualPoints )
		# print("Exterior Point Index:", self.exteriorPoints)
		# print("Exterior Points:", actualPoints)
		# actualPoints = actualPoints[ actualPoints[:, 2].argsort() ]
		# Using the sort algorithm from here: https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column/2828121#2828121
		# sortedActualPoints = actualPoints[actualPoints[:, 1].argsort( kind='mergesort' )]
		sortedActualPoints = actualPoints[actualPoints[:, 0].argsort( kind='mergesort' )] # Sort by the X axis.
		# actualPoints.view('f,f,f')
		# print(type(actualPoints))
		# print("Sorted Points:", sortedActualPoints )

		newExteriorPoints = []
		# Recreate the exterior Points as indeces.
		for point in sortedActualPoints:
			# result = np.where( self.points == point )
			# print('Result:', result)
			# print("Indeces:", tempPoints.index( list(point) ))
			newExteriorPoints.append( tempPoints.index( list(point) ) )

		# print("New Exterior Point Indeces:", newExteriorPoints)
		self.exteriorPoints = newExteriorPoints.copy()



	def GenerateExteriorPoints(self):
		exteriorPoints = []
		for pointIndex in range( len(self.points) ):
			pointAngle = self.CalculateAngleAroundPoint(pointIndex)
			# print("Point Angle:", pointAngle)
			if pointAngle < 359.9999:  # 359.99 as a fudge factor.
				exteriorPoints.append( pointIndex )

		self.exteriorPoints = exteriorPoints
		if Bridson_Common.sortExteriorPoints == True:
			self.sortExteriorPoints()




	# (v1,v2,v3) -> DualTriangle
	def GenerateVertexToTriangleMap(self):
		TriangleHashMap = {}
		# Map all Triangles
		for i  in range(len(self.Triangles)):
			currentTri = self.Triangles[i]
			newTriangle = TriangulationDualGraph.DualTriangle( i )
			newTriangle.setVertices( self.Triangles[i] , self.points )
			v1 = currentTri[0]
			v2 = currentTri[1]
			v3 = currentTri[2]
			edge1 = self.EdgeHashMap.get((v1,v2))
			edge2 = self.EdgeHashMap.get((v2, v3))
			edge3 = self.EdgeHashMap.get((v3, v1))

			newTriangle.addEdge(edge1)
			newTriangle.addEdge(edge2)
			newTriangle.addEdge(edge3)
			TriangleHashMap[(v1, v2, v3)] = newTriangle
			TriangleHashMap[(v2, v3, v1)] = newTriangle
			TriangleHashMap[(v3, v1, v2)] = newTriangle

			# Register the Triangle with the Edges.
			edge1.addTriangle(i, newTriangle )
			edge2.addTriangle(i, newTriangle)
			edge3.addTriangle(i, newTriangle)

			# Need to create a map from triangleInde


			neighbours = self.Neighbours[i]
			# For neighbours, we need to add the neighbour to the current Triangle.
			# According to the documentation, the neighbour[i,j] is the triangle that is the neighbour to the edge from point index triangles[i,j] to point index triangles[i, (j+1)%3]
			for neighbourIndex in neighbours:
				if neighbourIndex != -1:
					currentNeighbour = self.Triangles[ neighbourIndex ]
					newTriangle.addNeighbour( currentNeighbour )

		self.TriangleHashMap = TriangleHashMap


	# Edge(start, end) -> Edge
	def GenerateEdgePointsToEdgeMap(self):
		EdgeHashMap = {}
		for edge in self.Edges:
			start, end = edge
			newEdge = TriangulationDualGraph.DualEdge(start, end)
			# Map the two combinations that produce the same edge.
			EdgeHashMap[(start,end)] = newEdge
			EdgeHashMap[(end,start)] = newEdge
		self.EdgeHashMap = EdgeHashMap

	# Edge(start, end) -> Triangle ** We already have this mapping in the Edge itself.  Obtain from the Edge.


	# Given a Point, return a list of Triangles.
	def GetPointTriangleMembership(self, pointIndex):
		# Find the edges that the point belongs to.
		edges = self.PointToEdgeHashMap.get( pointIndex ).copy()
		triangleIndexList = []
		# print("Edges:", edges )
		for edgeVertices in edges:
			dualEdge = self.EdgeHashMap.get( edgeVertices )
			# print("DualEdge triangle vertices:", dualEdge.triangleIndeces )
			triangleIndexList.extend( dualEdge.triangleIndeces.copy() )


		# Make the list unique.
		triangleIndexList = list ( np.unique(triangleIndexList) )
		# print("GetPointTriangleMembership triangleIndexList:", triangleIndexList)
		# print("Triangles:", triangleIndexList)
		return triangleIndexList



	def CalculateAngleAroundPoint(self, pointIndex):
		triangleIndexList = self.GetPointTriangleMembership(pointIndex)
		angle = 0
		for triangleIndex in triangleIndexList:
			# Calculate the angle of the point.
			angle += self.__CalculatePointAngle(pointIndex, triangleIndex)

		# print("TotalAngle:", angle)
		return angle

	def __CalculatePointAngle(self, pointIndex, triangleIndex):
		p1 = self.points[ pointIndex ]
		triangleVertices = list(self.Triangles[ triangleIndex ]).copy()
		# Remove the current point from the list of trianglePoints
		triangleVertices.pop( triangleVertices.index( pointIndex ) )
		p2, p3 = triangleVertices
		p2 = self.points[ p2 ]
		p3 = self.points[ p3 ]

		P12 = Bridson_Common.euclidean_distance(p1, p2)
		P13 = Bridson_Common.euclidean_distance(p1, p3)
		P23 = Bridson_Common.euclidean_distance(p2, p3)
		# print("Bridson_TriangulationDualGraph:", P12, P13, P23) # Output the angles.
		# print("Bridson_TriangulationDualGraph Denominator:", (P12*P12 + P13*P13 - P23*P23) / (2*P12*P13) )  # Output the angles.
		value = (P12*P12 + P13*P13 - P23*P23) / (2*P12*P13)
		if value >= 1.0:
			value = 0.99999999999999999999
		elif value <= -1.0:
			value = -0.99999999999999999999
		angle = math.acos( value )*180/math.pi
		# print("angle1: ", angle)
		return angle

	def GetEdgesFromTriangleList(self, triangleIndexList):
		edgeList = []
		for triangleIndex in triangleIndexList:
			triangleVertices = tuple(self.Triangles[triangleIndex])
			dualTriangle = self.TriangleHashMap[triangleVertices]
			for dualEdge in dualTriangle.edges:
				if not dualEdge.vertices in edgeList: # Create a unique list.
					edgeList.append( dualEdge.vertices )

		return edgeList

	# Given points and Edges
	# Use Case: Find Exterior Points
	# Use Case: Given Point, find Exterior Edges
	# Use Case: Give Point, find complement Points - This can potentially provide many choices.  ** Useful for determining the angle about a point.
	# Use Case: Given Point, find complement Point along exterior Edge
	# Find Angles around a point


	# HashMap: PointIndex -> Edge(start,end)
	def GeneratePointHashMaps(self):
		Edges = self.Edges
		PointToEdgeHashMap = {}

		for edge in Edges:
			start, end = edge
			newEdge = self.EdgeHashMap.get((start,end))

			# Add PointIndex -> Edge
			if PointToEdgeHashMap.get(start) == None:
				PointToEdgeHashMap[start] = []
			pointAssociatedEdges = PointToEdgeHashMap.get(start)
			pointAssociatedEdges.append( newEdge.vertices )
			if PointToEdgeHashMap.get(end) == None:
				PointToEdgeHashMap[end] = []
			pointAssociatedEdges = PointToEdgeHashMap.get(end)
			pointAssociatedEdges.append( newEdge.vertices )
		self.PointToEdgeHashMap = PointToEdgeHashMap
		Bridson_Common.logDebug(__name__,"PointToEdgeMap: ", self.PointToEdgeHashMap)



	def FindFirstIntersection(self, pointIndex, trifinder):
		'''
			1. Determine the direction of the line.
			a. Test going up and check to see if new point is in the trifinder.  How far should we step?
				i. Obtain the max height of all triangles that connect to the exterior point.
				ii. Create a vertical line that is maxTriangleHeights in length in the up direction.
				iii. If we are in the trifinder, keep going this direction.
				iv. Else direction is down.
			b. Continue in the direction until we fall off the trifinder.
		'''
		# Get triangles associated with the point.
		# - Have to obtain Point -> DualEdge
		# - Then obtain DualEdge -> Triangle
		# Obtain all THE max triangleHeights.
		# Add method to DualGraph to obtain this information.
		x, y = self.points[pointIndex]
		Bridson_Common.logDebug(__name__,"FindFirstIntersection pointIndex:", pointIndex)
		minHeight, maxHeight, triangleIndexList = self.GetMaxTriangleHeightFromPoint(pointIndex)

		# Find the proper direction, either up or down.
		upX, upY = x, y + maxHeight
		if trifinder(upX, upY) > -1:
			# The upDirection is the correct direction.
			direction = 1.0
		else:
			# The direction will be negative.
			direction = -1.0

		Bridson_Common.logDebug(__name__,"FindFirstIntersection Direction:", direction)

		# Obtain the edges based on the triangles.
		edges = self.GetEdgesFromTriangleList(triangleIndexList).copy()
		Bridson_Common.logDebug(__name__,"FindFirstIntersection Edge List:", edges)

		# We have the associated edges.
		# Create the check line.  The CheckLine will start from 0.9*minHeight -> 1.1*maxHeight.
		CheckLineStart = (x, y + minHeight * direction) # For some reason, it requires the minHeight.
		CheckLineEnd = (x, y + 100.0 * maxHeight * direction)
		ReferenceLine = (CheckLineStart, CheckLineEnd)
		found = False
		# find the intersection
		for edge in edges:
			edgeSegment = (self.points[edge[0]], self.points[edge[1]])
			intersection = Bridson_Common.line_intersect(ReferenceLine, edgeSegment)
			if intersection == None:
				continue
			else:
				found = True
				break

				Bridson_Common.logDebug(__name__,"FindFirstIntersection Interesecting Edge:", edge)
		# Need to determine which triangle.
		if found:
			# Need to determine which triangleIndex
			# Get DualEdge
			dualEdge = self.EdgeHashMap[ edge ]
			Bridson_Common.logDebug(__name__,"FindFirstIntersection interesection point:", intersection)
			trianglesList = dualEdge.triangleIndeces.copy()
			Bridson_Common.logDebug(__name__,"FindFirstIntersection triangleList:", trianglesList)
			if len(trianglesList) > 0:
				for triangleIndex in trianglesList:
					vertices = self.Triangles[ triangleIndex ].copy()
					Bridson_Common.logDebug(__name__,"FindFirstIntersection vertices:", vertices)
					if pointIndex in vertices:
						break
					else:
						continue

				Bridson_Common.logDebug(__name__,"FindFirstIntersection Edge intersection:", edge)
				Bridson_Common.logDebug(__name__,"Intersection:", intersection)
				Bridson_Common.logDebug(__name__,"FindFirstIntersection TriangleIndex:", triangleIndex)

				return intersection, edge, triangleIndex, direction
			else:
				return None, None, None, None
		else:
			Bridson_Common.logDebug(__name__,"No intersection found:", intersection)
			return None, None, None, None


	def FindNextIntersection(self, currentIntersection, edge, triangleIndex, direction):
		'''

		:param currentIntersection: (x, y)
		:param edge: (start, end)
		:param triangleIndex: index of triangle
		:return:

		1. Based on the edge, determine the next triangle.
		2. Based on the triangle, obtain the minHeight and maxHeight.
		3. If there is a next triangle, determine the next intersection.
		'''
		dualEdge = self.EdgeHashMap[ edge ]

		Bridson_Common.logDebug(__name__,"triangleIndex:", triangleIndex)
		Bridson_Common.logDebug(__name__,"Starting Edge:", edge)
		triangleIndexList = dualEdge.triangleIndeces.copy()
		Bridson_Common.logDebug(__name__,"triangleIndexList:", triangleIndexList)
		triangleIndexList.pop( triangleIndexList.index(triangleIndex) ) # Remove the current triangleIndex.
		Bridson_Common.logDebug(__name__,"triangleIndexList:", triangleIndexList)
		if len(triangleIndexList) < 1:
			# No further triangles.
			return None, None, None, False

		minHeight, maxHeight, triangleIndexList = self.GetMaxTriangleHeightFromEdge(edge, triangleIndex)
		triangleIndex = triangleIndexList[0]
		Bridson_Common.logDebug(__name__,"Heights:", minHeight, maxHeight, triangleIndexList)
		if minHeight == None:
			return None, None, triangleIndex, False

		x,y = currentIntersection
		# We have the associated edges.
		# Create the check line.  The CheckLine will start from 0.9*minHeight -> 1.1*maxHeight.
		CheckLineStart = (x, y )
		CheckLineEnd = (x, y + 100.0 * maxHeight * direction) # 10.0 to handle triangles that are really squished.
		ReferenceLine = (CheckLineStart, CheckLineEnd)
		Bridson_Common.logDebug(__name__,"ReferenceLine:", ReferenceLine)
		found = False

		edges = self.GetEdgesFromTriangleList(triangleIndexList).copy()
		# Remove the starting edge from teh list.
		edges.pop( edges.index(edge) )
		Bridson_Common.logDebug(__name__,"Edge List:", edges)
		# find the intersection
		for edge in edges:
			edgeSegment = (self.points[edge[0]], self.points[edge[1]])
			Bridson_Common.logDebug(__name__,"ReferenceLine:", ReferenceLine)
			Bridson_Common.logDebug(__name__,"edgeSegment:", edgeSegment)
			intersection = Bridson_Common.line_intersect(ReferenceLine, edgeSegment)
			Bridson_Common.logDebug(__name__,"Intersection:", intersection)
			if intersection == None:
				continue
			else:
				found = True
				break

		# Need to determine which triangle.
		if found:
			if intersection in self.Edges:
				print("************* FindNextIntersection at Vertex.")
				Bridson_Common.logDebug(__name__,"Found Intersection.")
			Bridson_Common.logDebug(__name__, "FindNextIntersection Edge intersection:", edge)
			Bridson_Common.logDebug(__name__,"Intersection:", intersection)
			Bridson_Common.logDebug(__name__,"FindNextIntersection TriangleIndex:", triangleIndex)

			finalIntersection = self.isFinalIntersection(edge, triangleIndex)

			return intersection, edge, triangleIndex, finalIntersection
		else:
			Bridson_Common.logDebug(__name__,"No intersection found:", intersection)
			return None, None, triangleIndex, False


	def isFinalIntersection(self, edge, currentTriangleIndex):
		dualEdge = self.EdgeHashMap[ edge ]
		triangleIndexList = dualEdge.triangleIndeces.copy()
		Bridson_Common.logDebug(__name__,"TriangleList:", triangleIndexList)
		triangleIndexList.pop(triangleIndexList.index(currentTriangleIndex))  # Remove the current triangleIndex.
		Bridson_Common.logDebug(__name__,"remaining TriangleList:", triangleIndexList)
		if len(triangleIndexList) == 0:
			return True
		else:
			return False


def test():
	imageraster, regionMap = Bridson_Main.SLICImage()

	successfulRegions = 0

	for index in [31]:
	# for index in range( len(regionMap.keys()) ):
		print("Starting Region: ", index)

		# Generate the raster for the first region.
		raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
		print("Raster:", raster)
		Bridson_Main.displayRegionRaster(raster, index)

		# Bridson_Common.logDebug(__name__, raster)
		# Bridson_Common.arrayInformation( raster )
		for i in range(1):
			indexLabel = index + i / 10
			Bridson_Common.writeMask(raster)
			meshObj, flatMeshObj, LineSeedPointsObj, trifindersuccess = Bridson_Main.processMask(raster, dradius, indexLabel)


		# meshObj.DualGraph.GetMaxTriangleHeight( 0 )
		# triangleList = meshObj.DualGraph.GetPointTriangleMembership( 0 )
		# print("TriangleList:", triangleList )
		# for triangleIndex in triangleList:
		# 	meshObj.colourTriangle( triangleIndex )
		# 	# meshObj.colourTriangleCluster( triangleIndex )

		# meshObj.DrawVerticalLinesExteriorSeed2()
		# flatMeshObj.TransferLinePointsFromTarget( meshObj )
		flatMeshObj.DrawVerticalLinesExteriorSeed2()
		meshObj.TransferLinePointsFromTarget( flatMeshObj )
		# meshObj.DrawVerticalLinesExteriorSeed() # Start testing the vertical line drawing.


	plt.show()

def test2():
	points = [[ 4.5 ,     37.5     ],[ 6.5  ,    52.5     ],[15.5   ,   18.5     ],[ 9.5   ,   27.5     ],[ 8.5   ,   62.5     ],
	          [23.5    ,   9.5     ],[19.073324, 47.92784 ],[19.5  ,    67.5     ],[33.5  ,     3.5     ],[26.542643, 29.841183],
	          [29.5  ,    71.5     ],[38.5  ,    10.5     ],[38.5  ,    22.5     ],[38.5  ,    34.5     ],[38.5   ,   46.5     ],[38.5   ,   58.5     ],[38.5  ,    70.5     ]]
	points = np.array(points)
	Triangles = [[12,  9 , 5],[12, 13,  9],[ 9,  6,  3],[ 6,  1 , 0],[ 6 , 9, 14],[ 6 ,15 , 7]]
	Edges = [[ 1 , 0],[ 6 , 0],[ 6 , 1],[ 6 , 3],[ 7 , 6],[ 9 , 3],[ 9 , 5],[ 9 , 6],[12,  5],[12,  9],[13 , 9],[13, 12],[14 , 6],[14 , 9],[15 , 6],[15,  7]]
	Neighbours = [[ 1, -1 ,-1],[-1, -1 , 0],[ 4 ,-1 ,-1],[-1, -1 ,-1],[ 2, -1, -1],[-1 ,-1 ,-1]]

	DualGraph = TriangulationDualGraph(points, Edges, Triangles, Neighbours)

	if Bridson_Common.debug:
		plt.figure()
		plt.subplot(1, 1, 1, aspect=1)
		plt.title('Display Delaunay')
		if Bridson_Common.invert:
			plt.triplot(points[:,1], xrange-points[:,0], Triangles)
		else:
			plt.triplot(points[:, 1], points[:, 0], Triangles)
		# plt.triplot(points[:, 0], points[:, 1], tri.triangles)
		if Bridson_Common.invert:
			plt.plot(points[:, 1], xrange-points[:, 0], 'o')
		else:
			plt.plot(points[:, 1], points[:, 0], 'o')
		thismanager = pylab.get_current_fig_manager()
		thismanager.window.wm_geometry("+640+0")
		plt.show()



if __name__ == '__main__':
	dradius = Bridson_Common.dradius # 3 seems to be the maximum value.
	xrange, yrange = 10, 10
	# test()
	test()
