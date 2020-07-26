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

		def setVertices(self, vertices):
			self.vertices = vertices

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



	def GenerateExteriorPoints(self):
		exteriorPoints = []
		for pointIndex in range( len(self.points) ):
			pointAngle = self.CalculateAngleAroundPoint(pointIndex)
			# print("Point Angle:", pointAngle)
			if pointAngle < 359.9999:  # 359.99 as a fudge factor.
				exteriorPoints.append( pointIndex )

		self.exteriorPoints = exteriorPoints

	# (v1,v2,v3) -> Triangle
	def GenerateVertexToTriangleMap(self):
		TriangleHashMap = {}
		# Map all Triangles
		for i  in range(len(self.Triangles)):
			currentTri = self.Triangles[i]
			newTriangle = TriangulationDualGraph.DualTriangle( i )
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
		edges = self.PointToEdgeHashMap.get( pointIndex )
		triangleIndexList = []
		# print("Edges:", edges )
		for edgeVertices in edges:
			dualEdge = self.EdgeHashMap.get( edgeVertices )
			# print("DualEdge triangle vertices:", dualEdge.triangleIndeces )
			triangleIndexList.extend( dualEdge.triangleIndeces )

		# Make the list unique.
		triangleIndexList = list ( np.unique(triangleIndexList) )
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

		angle = math.acos( (P12*P12 + P13*P13 - P23*P23) / (2*P12*P13) )*180/math.pi
		# print("angle1: ", angle)
		return angle


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
		print("PointToEdgeMap: ", self.PointToEdgeHashMap)





	# def FindAngleAroundPoint(self):
		# May also need Point -> Triangle Map.


def test():
	imageraster, regionMap = Bridson_Main.SLICImage()

	successfulRegions = 0

	for index in range(37,38):
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

		meshObj.generateDualGraph()
		meshObj.colourTriangleCluster(15)
		meshObj.colourTriangleCluster(32)
		meshObj.colourTriangleCluster(88)
		meshObj.DualGraph.GetPointTriangleMembership(0)
		meshObj.ax.plot(meshObj.points[0][0],meshObj.points[0][1], color='r', markersize=5, marker='*')
		meshObj.DualGraph.CalculateAngleAroundPoint( 0)

		# Draw exterior dots
		for pointIndex in meshObj.DualGraph.exteriorPoints:
			point = meshObj.DualGraph.points[ pointIndex ]
			meshObj.ax.plot(point[0], point[1], color='r', markersize=5, marker='*')


		# Draw exterior edges
		for edge in meshObj.DualGraph.exteriorEdges:
			exteriorEdges = []
			start, end = edge
			startPoint = meshObj.DualGraph.points[start]
			endPoint = meshObj.DualGraph.points[end]
			exteriorEdges.append(startPoint)
			exteriorEdges.append(endPoint)
			exteriorEdges = np.array(exteriorEdges)
			meshObj.ax.plot(exteriorEdges[:, 0], exteriorEdges[:, 1], color='c')


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
