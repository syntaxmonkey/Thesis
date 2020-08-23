

class RegionPixelConnectivity:
	def __init__(self, pixelList):
		# This will be a list of pixel coordinates.
		self.edgePixelList = pixelList
		# This will map the pixel
		self.pixelAdjacentMap = {}

		pass

	def setEdgePixelList(self, pixelList):
		# List of pixels on the Edge.
		self.edgePixelList = pixelList

	def setPointOnEdge(self, pointsOnEdge):
		# List of EdgePoint objects on the edge.
		self.pointsOnEdge = pointsOnEdge


	'''
		Can add adjacency map.
		<starting region index, ending region index> = [
			[starting region EdgePoints],
			[ending regino EdgePoints]
			]
			
			
			
		Add EdgePoint class:
			-> (x,y)
			-> pixel(x,y)
			-> line
			-> region index
			-> adjacent index
			
			
	'''
