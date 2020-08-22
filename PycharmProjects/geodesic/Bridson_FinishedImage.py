
import matplotlib.pyplot as plt
import matplotlib
import SLIC
import Bridson_Common
import numpy as np
from skimage.segmentation import mark_boundaries
import math
from scipy.spatial import distance

class FinishedImage:
	def __init__(self, *args, **kwargs):
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(1, 1, 1, aspect=1)
		self.ax.set_title('Merged Image' )
		# self.set
		# self.ax.invert_yaxis()

	def setTitle(self, filename):
		self.ax.set_title('Merged Image - ' + filename + ' - Segments: ' + str(Bridson_Common.segmentCount) + ' - regionPixels: ' + str(Bridson_Common.targetRegionPixelCount) )

	def setXLimit(self, left, right):
		# print("Left:", left, "Right:", right)
		self.ax.set_xlim(left=left, right=right)

	def setYLimit(self, top, bottom):
		# print("Left:", left, "Right:", right)
		self.ax.set_ylim(top=top, bottom=bottom)


	def cropContourLines(self, linePoints, raster):
		if Bridson_Common.cropContours:
			# Raster will have 255 for valid points.
			newLinePoints = []
			for line in linePoints:
				newLine = []
				emptyLine = True
				for point in line:
					x, y = int(point[0]), int(0-point[1])
					# print("Point:", x,y)
					try:
						if raster[y][x] == 255: # Because of the differing coordinates system, we have to flip the order of x,y
							# print("Point:", point)
							# print("Point in Mask:", x, y)
							newLine.append( point )
							emptyLine = False
						else:
							# print("Point NOT in Mask:", x, y)
							pass
					except:
						print("********************* Problem with cropping contour lines. **********************")
						print("X,Y:", x,y)
						print("Raster dimensions:", np.shape(raster))
						print("Raster:", raster)

				if emptyLine == False:
					newLinePoints.append(np.array(newLine))

			newLinePoints = np.array(newLinePoints)
			return newLinePoints
		else:
			return linePoints


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
		# if Bridson_Common.drawSLICRegions:
		self.ax.imshow(mark_boundaries(regionRaster, segments, color=(255,0,0)))
		self.ax.grid()

	def shiftRastersMeshObj(self, regionMap, regionRaster, maskRasterCollection, meshObjCollection ):
		# for index in [5]:
		for index in maskRasterCollection.keys():
			raster = maskRasterCollection[ index ]
			meshObj = meshObjCollection[ index ]

			# print("Shape of Region Raster:", np.shape(regionRaster))


			regionCoordinates = regionMap.get(index)
			topLeftTarget, bottomRightTarget = SLIC.calculateTopLeft(regionCoordinates)

			# Create new raster that has been shifted.
			x, y = np.shape(raster)
			linePoints = meshObj.linePoints
			newRaster = np.zeros(np.shape(regionRaster))
			# print("Shape of raster:", (x,y))
			for i in range(x):
				for j in range(y):
					newRaster[i - topLeftTarget[0] - 3 ][j + topLeftTarget[1] - 5 ] = raster[i][j]
					# newRaster[i - topLeftTarget[0] + 5][j + topLeftTarget[1] - 5] = raster[i][j]

			newLinePoints = []
			for line in linePoints:
				newline = line.copy()
				newline[:, 0] = line[:, 0] + topLeftTarget[1] - 5 # Needed to line up with regions.
				newline[:, 1] = line[:, 1] - topLeftTarget[0] + 3  # Required to be negative.
				newLinePoints.append( newline )

			plt.figure()
			ax = plt.subplot(1, 1, 1, aspect=1)
			plt.title('shifted Lines')
			plt.grid()
			for line in newLinePoints:
				ax.plot(line[:, 0], line[:, 1], color='r', marker='*')

			# meshObj.linePoints = newLinePoints
			# maskRasterCollection[index] = newRaster

			plt.figure()
			plt.subplot(1, 1, 1, aspect=1)
			plt.title('shifted Mask')
			plt.grid()
			plt.imshow(newRaster)





	def mergeLines(self, regionMap, regionRaster, maskRasterCollection, meshObjCollection ):
		self.shiftRastersMeshObj( regionMap, regionRaster, maskRasterCollection, meshObjCollection )

		# For each region, determine the points on the lines that are close to the region edge.  Make a registry of these points.
		for index in maskRasterCollection.keys():
			raster = maskRasterCollection[ index ]
			meshObj = meshObjCollection[ index ]

			regionCoordinates = regionMap.get(index)
			topLeftTarget, bottomRightTarget = SLIC.calculateTopLeft(regionCoordinates)






		pass


	def drawRegionContourLines(self, regionMap, index, meshObj, regionIntensity, drawSLICRegions = Bridson_Common.drawSLICRegions):

		# If we are not drawing the SLIC regions, we do not need to flip the Y coordinates.
		# If we draw the SLIC regions, we need to flip the Y coordinates.
		if drawSLICRegions == False:
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
		currentLine = linePoints[0] * -100  # Set the starting line way off the current raster region.
		initial=True
		# count = 0

		# Allow regions to be blank.
		if Bridson_Common.allowBlankRegion == True:
			if regionIntensity > 250:
				return
			# if self.calculateLineSpacing(linePoints[0], linePoints[-1], intensity=regionIntensity) == False:
			# 	return

		for index in range(len(linePoints)):
			# if count > 5:
			# 	break
			if index % Bridson_Common.lineSkip == 0:
				colour = Bridson_Common.colourArray[ (index % len(Bridson_Common.colourArray) ) ]
				line = linePoints[index].copy()
				line = line * Bridson_Common.mergeScale
				# For some reason we need to swap the topLeft x,y with the line x,y.
				line[:, 0] = line[:, 0] + topLeftTarget[1] - 5 # Needed to line up with regions.
				line[:, 1] = line[:, 1] - topLeftTarget[0] + 5  # Required to be negative.
				if self.calculateLineSpacing(currentLine, line, intensity=regionIntensity) == True:
					self.ax.plot(line[:, 0], line[:, 1]*flip, color=colour)
					if Bridson_Common.closestPointPair:  # Only place the dots when we are calculating closest point pair.
						if initial == False:  # Do not display this dot the first time around.
							self.ax.plot(currentLine[self.markPoint[0]][0], currentLine[self.markPoint[0]][1] * flip, marker='*',
							             markersize=6, color='g')  # Colour middle point.
						self.ax.plot(line[self.markPoint[1]][0], line[self.markPoint[1]][1]*flip, marker='o', markersize=2, color='r')  # Colour middle point.
					currentLine = line
					initial = False
					# count += 1


	def findCloserDistance(self, l1p1, l1p2, l2p1):
		firstDistance = Bridson_Common.euclidean_distance(l1p1, l2p1)
		secondDistance = Bridson_Common.euclidean_distance(l1p2, l2p1)
		if firstDistance < secondDistance:
			return firstDistance
		else:
			return secondDistance

	def findClosestPointPair(self, line1, line2):
		a = distance.cdist(line1, line2, 'euclidean')
		# print(a)

		minDistance = a.min()
		# print("Min Value:", minDistance)
		result = np.where(a == minDistance)
		# print("Result:", result)
		listOfCordinates = list(zip(result[0], result[1]))
		# print("Min Index:", listOfCordinates)

		minPointIndex = listOfCordinates[0]
		# print(a[minPointIndex[0], minPointIndex[1]])
		self.markPoint = minPointIndex
		return minDistance

	def calculateLineSpacing(self, line1, line2, factor=Bridson_Common.lineCullingDistanceFactor, intensity=255):
		intensityDistance = intensity / 25
		# Get the endPoints of the lines.
		distance = 0
		if Bridson_Common.closestPointPair == False:
			if Bridson_Common.middleAverageOnly == False:
				# If middleAverageOnly is set to true,
				# print("3 point average.")
				distance += self.findCloserDistance(line1[0], line1[-1], line2[0])
				distance += self.findCloserDistance(line1[0], line1[-1], line2[-1])

			distance += Bridson_Common.euclidean_distance(line1[math.floor(len(line1)/2)], line2[math.floor( len(line2)/2)])

			distance = distance / Bridson_Common.divisor
		else:
			distance = self.findClosestPointPair(line1, line2)
		# print("Distance:", distance, Bridson_Common.dradius*factor)
		# if distance > Bridson_Common.dradius*factor:
		if distance > intensityDistance:
			# print("Far enough")
			return True
		else:
			# print("Too close")
			return False