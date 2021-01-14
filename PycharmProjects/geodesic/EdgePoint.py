


class EdgePoint:
	def __init__(self, xyCoordinates, associateLine, regionIndex, pointIndex):
		self.xy = xyCoordinates
		self.associatedLine = associateLine
		self.regionIndex = regionIndex
		self.pointIndex = pointIndex # The index of the x,y coordinates on the line.
		self.adjacentIndex = None
		self.adjacencyEdge = None
		pass


	def setAdjacentIndex(self, adjacentIndex):
		self.adjacentIndex = adjacentIndex
		pass

	def getAdjacentIndex(self):
		return self.adjacentIndex

	def setAdjacencyEdge(self, adjacencyEdge):
		self.adjacencyEdge = adjacencyEdge
		pass

	def getAdjacencyEndge(self):
		return self.adjacencyEdge