import Bridson_Common
import Bridson_Main
import SLIC
import matplotlib.pyplot as plt
import pylab
import numpy as np

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

	class Edge:
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

	class Triangle:
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
		self.CreateGraph(points, Edges, Triangles, Neighbours)
		# else:
		# 	Exception("TriangulationDualGraph creator - Not enough parameters.")


	def CreateGraph(self, points, Edges, Triangles, Neighbours):
		'''

		'''
		EdgeHashMap = {}
		for edge in Edges:
			start, end = edge
			newEdge = TriangulationDualGraph.Edge(start, end)
			# Map the two combinations that produce the same edge.
			EdgeHashMap[(start,end)] = newEdge
			EdgeHashMap[(end,start)] = newEdge
		self.EdgeHashMap = EdgeHashMap

		TriangleHashMap = {}
		# Map all Triangles
		for i  in range(len(Triangles)):
			currentTri = Triangles[i]
			newTriangle = TriangulationDualGraph.Triangle( i )
			v1 = currentTri[0]
			v2 = currentTri[1]
			v3 = currentTri[2]
			edge1 = EdgeHashMap.get((v1,v2))
			edge2 = EdgeHashMap.get((v2, v3))
			edge3 = EdgeHashMap.get((v3, v1))

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

			neighbours = Neighbours[i]
			# For neighbours, we need to add the neighbour to the current Triangle.
			# According to the documentation, the neighbour[i,j] is the triangle that is the neighbour to the edge from point index triangles[i,j] to point index triangles[i, (j+1)%3]
			for neighbourIndex in neighbours:
				if neighbourIndex != -1:
					currentNeighbour = Triangles[ neighbourIndex ]
					newTriangle.addNeighbour( currentNeighbour )

		self.TriangleHashMap = TriangleHashMap







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
