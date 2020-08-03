
import matplotlib.pyplot as plt
import matplotlib
import SLIC
import Bridson_Common

class FinishedImage:
	def __init__(self, *args, **kwargs):
		plt.figure()
		self.ax = plt.subplot(1, 1, 1, aspect=1)
		self.ax.set_title('Merged Image' )
		# self.ax.invert_yaxis()

	def setXLimit(self, left, right):
		print("Left:", left, "Right:", right)
		self.ax.set_xlim(left=left, right=right)


	def drawRegionContourLines(self, regionMap, index, meshObj):
		regionCoordinates = regionMap.get( index  )
		# Grab the shiftx and shifty based on regionMap.
		topLeft, bottomRight = SLIC.calculateTopLeft( regionCoordinates )

		# Obtain the linePoints from the meshObj.
		linePoints = meshObj.linePoints.copy()

		print("TopLeft: ", topLeft)
		# print("LinePoints:",linePoints)

		for index in range(len(linePoints)):
			colour = Bridson_Common.colourArray[ (index % len(Bridson_Common.colourArray) ) ]
			line = linePoints[index].copy()
			line = line * Bridson_Common.mergeScale
			# For some reason we need to swap the topLeft x,y with the line x,y.
			line[:, 0] = line[:, 0] + topLeft[1] + 5
			line[:, 1] = line[:, 1] - topLeft[0] + 5  # Required to be negative.

			self.ax.plot(line[:, 0], line[:, 1], color=colour)