
import matplotlib.pyplot as plt
import matplotlib
import SLIC
import Bridson_Common
import numpy as np
from skimage.segmentation import mark_boundaries

class FinishedImage:
	def __init__(self, *args, **kwargs):
		plt.figure()
		self.ax = plt.subplot(1, 1, 1, aspect=1)
		self.ax.set_title('Merged Image' )
		# self.ax.invert_yaxis()

	def setXLimit(self, left, right):
		print("Left:", left, "Right:", right)
		self.ax.set_xlim(left=left, right=right)


	def cropContourLines(self, linePoints, raster):
		# Raster will have 255 for valid points.
		newLinePoints = []
		for line in linePoints:
			newLine = []
			emptyLine = True
			for point in line:
				x, y = int(point[0]), int(0-point[1])
				# print("Point:", x,y)
				if raster[y][x] == 255: # Because of the differing coordinates system, we have to flip the order of x,y
					# print("Point:", point)
					# print("Point in Mask:", x, y)
					newLine.append( point )
					emptyLine = False
				else:
					# print("Point NOT in Mask:", x, y)
					pass

			if emptyLine == False:
				newLinePoints.append(np.array(newLine))

		newLinePoints = np.array(newLinePoints)
		return newLinePoints

	def maxLinePoints(self, linePoints):
		maxx, maxy = -1000000, -1000000
		for line in linePoints:
			for point in line:
				x,y = point
				if x > maxx:
					maxx = x

				if y > maxy:
					maxy = y

		return (x,y)


	def drawSLICRegions(self, regionRaster, segments):
		if Bridson_Common.drawSLICRegions:
			self.ax.imshow(mark_boundaries(regionRaster, segments))
			self.ax.grid()

	def drawRegionContourLines(self, regionMap, index, meshObj):

		# If we are not drawing the SLIC regions, we do not need to flip the Y coordinates.
		# If we draw the SLIC regions, we need to flip the Y coordinates.
		if Bridson_Common.drawSLICRegions == False:
			flip = 1
		else:
			flip = -1


		# Need to utilize the region Raster.
		raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
		# Obtain the linePoints from the meshObj.
		linePoints = meshObj.linePoints.copy()
		# topLeftSource = self.maxLinePoints(linePoints)

		# print("Region Coordinates:", regionCoordinates)
		# print("LinePoints:", linePoints)

		# Grab the shiftx and shifty based on regionMap.
		regionCoordinates = regionMap.get( index  )
		topLeftTarget, bottomRightTarget = SLIC.calculateTopLeft( regionCoordinates )
		# topLeftSource = self.maxLinePoints( linePoints )
		# shiftCoordinates = (bottomRightTarget[0] - topLeftSource[0], bottomRightTarget[1] - topLeftSource[1])

		linePoints = self.cropContourLines(linePoints, raster)

		# print("NewLinePoints:", linePoints)
		# print("actualTopLeft: ", actualTopLeft)
		# print("TopLeftTarget: ", topLeftTarget)
		# print("bottomRightTarget: ", bottomRightTarget)
		# print("TopLeftSource: ", topLeftSource)
		# # print("shiftCoordinates:", shiftCoordinates)
		# print("LinePoints:",linePoints)

		for index in range(len(linePoints)):
			colour = Bridson_Common.colourArray[ (index % len(Bridson_Common.colourArray) ) ]
			line = linePoints[index].copy()
			line = line * Bridson_Common.mergeScale
			# For some reason we need to swap the topLeft x,y with the line x,y.
			line[:, 0] = line[:, 0] + topLeftTarget[1] - 6 # Needed to line up with regions.
			line[:, 1] = line[:, 1] - topLeftTarget[0] + 6  # Required to be negative.

			self.ax.plot(line[:, 0], line[:, 1]*flip, color=colour)