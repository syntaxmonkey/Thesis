# Will represent the edge of an adjacency.
# Will point to region1 and region 2.
# Will contain the list of EdgePoints in region1 along the common edge.
# Will contain the list of EdgePoints in region2 along the common edge.

class AdjancecyEdge:
	def __init__(self, currentIndex, adjancentIndex):
		self.currentIndex = currentIndex
		self.adjacentIndex = adjancentIndex
		self.currentIndexEdgePoints = []
		self.adjacentIndexEdgePoints = []

