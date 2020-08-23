


class EdgePoint:
	def __init__(self, xyCoordinates, associateLine, regionIndex, pointIndex):
		self.xy = xyCoordinates
		self.associatedLine = associateLine
		self.regionIndex = regionIndex
		self.pointIndex = pointIndex # The index of the x,y coordinates on the line.
		pass

