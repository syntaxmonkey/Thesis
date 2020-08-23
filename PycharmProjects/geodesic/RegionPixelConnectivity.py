

class RegionPixelConnectivity:
	def __init__(self, pixelList):
		# This will be a list of pixel coordinates.
		self.edgePixelList = pixelList
		# This will map the pixel
		self.pixelAdjacentMap = {}

		pass

	def setEdgePixelList(self, pixelList):
		self.edgePixelList = pixelList

	def setPointOnEdge(self, pointsOnEdge):
		self.pointsOnEdge = pointsOnEdge