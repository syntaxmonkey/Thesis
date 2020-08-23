
import matplotlib.pyplot as plt
import matplotlib
import SLIC
import Bridson_Common
import numpy as np
from skimage.segmentation import mark_boundaries
import math
from scipy.spatial import distance
import sys
import RegionPixelConnectivity
import EdgePoint
import AdjacencyEdge

np.set_printoptions(threshold=sys.maxsize)  # allow printing without ellipsis: https://stackoverflow.com/questions/44311664/print-numpy-array-without-ellipsis

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

	def copyFromOther(self, otherFinishedImage):
		self.maskRasterCollection = otherFinishedImage.maskRasterCollection
		self.meshObjCollection = otherFinishedImage.meshObjCollection
		self.regionEdgePoints = otherFinishedImage.regionEdgePoints
		self.distanceRasters = otherFinishedImage.distanceRasters
		self.globalEdgePointMap = otherFinishedImage.globalEdgePointMap

		self.regionEdgePointMap =  otherFinishedImage.regionEdgePointMap # Map of (x,y) coordinates
		self.regionAdjancencyMap =  otherFinishedImage.regionAdjancencyMap # Map to contain AdjancencyEdge objects
		self.regionAdjacentRegions = otherFinishedImage.regionAdjacentRegions # Map containing list of adjacent regions for current region.


	def cropContourLines(self, linePoints, raster, topLeftTarget):

		if Bridson_Common.cropContours:
			# Raster will have 255 for valid points that should be retained.
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
						if Bridson_Common.debug:
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


	def cullLines(self, linePoints, regionIntensity):

		newLinePoints = []
		empty = True

		# Allow regions to be blank.
		if Bridson_Common.allowBlankRegion == True:
			if regionIntensity > 250:
				return empty, newLinePoints
			# if self.calculateLineSpacing(linePoints[0], linePoints[-1], intensity=regionIntensity) == False:
			# 	return

		currentLine = linePoints[-1]*-100
		for lineIndex in range(len(linePoints)):
			# if lineIndex % Bridson_Common.lineSkip == 0:
			if True:
				line = linePoints[lineIndex].copy()
				line = line * Bridson_Common.mergeScale
				# newLinePoints.append( line )

				if self.calculateLineSpacing(currentLine, line, intensity=regionIntensity) == True:
					newLinePoints.append( line )
					empty=False
					currentLine = line
		return empty, newLinePoints


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


	def shiftRastersMeshObj(self, regionMap, regionRaster):
		# for index in [5]:
		for index in self.maskRasterCollection.keys():
			raster = self.maskRasterCollection[ index ]
			meshObj = self.meshObjCollection[ index ]

			print("Shape of Region Raster:", np.shape(regionRaster))


			regionCoordinates = regionMap.get(index)
			topLeftTarget, bottomRightTarget = SLIC.calculateTopLeft(regionCoordinates)

			# Create new raster that has been shifted.
			x, y = np.shape(raster)
			linePoints = meshObj.linePoints
			newRaster = np.zeros(np.shape(regionRaster))
			# print("Shape of raster:", (x,y))
			for i in range(x):
				for j in range(y):
					# newRaster[i - topLeftTarget[0] - 3 ][j + topLeftTarget[1] - 5 ] = raster[i][j]
					if raster[i][j] != 0: # Only shift the points in the raster that have non-zero values.
						newRaster[i + topLeftTarget[0] - 5 ][j + topLeftTarget[1] - 5 ] = raster[i][j]

			newLinePoints = []
			for line in linePoints:
				newline = line.copy()
				newline[:, 0] = line[:, 0] + topLeftTarget[1] - 5 # Needed to line up with regions. Affects the x axis.
				newline[:, 1] = line[:, 1] - topLeftTarget[0] + 5   # Required to be negative.  Affects the y axis.
				newLinePoints.append( newline )

			if Bridson_Common.displayMesh:
				plt.figure()
				ax = plt.subplot(1, 1, 1, aspect=1)
				plt.title('shifted Lines for region ' + str(index))
				plt.grid()
				for line in newLinePoints:
					ax.plot(line[:, 0], line[:, 1], color='r', marker='*')

				plt.figure()
				plt.subplot(1, 1, 1, aspect=1)
				plt.title('shifted Mask for region ' + str(index))
				plt.grid()
				plt.imshow(newRaster)


			meshObj.linePoints = newLinePoints
			self.maskRasterCollection[index] = newRaster



	def setCollections(self, maskRasterCollection, meshObjCollection, regionEdgePoints, distanceRasters ):
		self.maskRasterCollection = maskRasterCollection
		self.meshObjCollection = meshObjCollection
		self.regionEdgePoints = regionEdgePoints
		self.distanceRasters = distanceRasters


	def mergeLines(self, regionMap, regionRaster, maskRasterCollection, meshObjCollection, regionIntensityMap ):
		self.maskRasterCollection = maskRasterCollection
		self.meshObjCollection = meshObjCollection
		self.regionMap = regionMap
		self.regionRaster = regionRaster
		self.regionEdgePoints = {}
		self.distanceRasters = {}
		self.regionIntensityMap = regionIntensityMap
		self.globalEdgePointMap = {} # Map of (x,y) coordinates that point to EdgePoint objects, if they exist.
		self.regionEdgePointMap = {} # Map of (x,y) coordinates
		self.regionAdjancencyMap = {} # Map to contain AdjancencyEdge objects
		self.regionAdjacentRegions = {} # Map containing list of adjacent regions for current region.

		self.shiftRastersMeshObj( regionMap, regionRaster )

		# For each region, determine the points on the lines that are close to the region edge.  Make a registry of these points.
		for index in self.maskRasterCollection.keys():
			raster = self.maskRasterCollection[ index ]
			meshObj = meshObjCollection[ index ]

			regionCoordinates = regionMap.get(index)
			topLeftTarget, bottomRightTarget = SLIC.calculateTopLeft(regionCoordinates)

			empty, culledLines = self.cullLines(  meshObj.linePoints, regionIntensityMap[index] )
			# meshObj.setCroppedLines( self.cullLines( index, meshObj.linePoints, regionIntensityMap[index] )  )
			if not empty:
				meshObj.setCroppedLines( culledLines )
			else:
				meshObj.setCroppedLines( [] )

			meshObj.setCroppedLines( self.cropContourLines(meshObj.croppedLinePoints, self.maskRasterCollection[index], topLeftTarget) )

			# We Generate the edge pixels and then create the edge connectivity object.
			distanceRaster, distance1pixelIndeces = self.genDistancePixels( raster )
			regionEdgePixels = RegionPixelConnectivity.RegionPixelConnectivity( distance1pixelIndeces )
			print("Setting regionEdgePoints index:", index)
			self.regionEdgePoints[ index ] = regionEdgePixels
			self.distanceRasters[ index ] = distanceRaster

			# Find the points that exist in the edge pixels.
			croppedLinePoints = meshObj.croppedLinePoints
			self.findLineEdgePoints(index, regionEdgePixels, croppedLinePoints)



		# self.displayDistanceMask( index, topLeftTarget, bottomRightTarget )
			###################################
			# DEBUG DEBUG : This section is for debugging purposes.
			# distanceMask = raster
			# Bridson_Common.displayDistanceMask(distanceMask, str(index), topLeftTarget, bottomRightTarget)
			####################################


	def highLightEdgePoints(self, index, color='g', drawSLICRegions=Bridson_Common.drawSLICRegions):
		# Get the list adjacencyEdge object.
		#adjancencyEdge = self.regionAdjancencyMap[ index ]
		if drawSLICRegions == False:
			flip = 1
		else:
			flip = -1

		adjacentRegions = self.regionAdjacentRegions[ index ]
		print("highLightEdgePoints adjacenct regions: " , adjacentRegions)
		for adjacentIndex in adjacentRegions:
			startingIndex = index if index < adjacentIndex else adjacentIndex
			endingIndex = index if index > adjacentIndex else adjacentIndex
			adjacencyEdge = self.regionAdjancencyMap[ (startingIndex,endingIndex) ]

			if index < adjacentIndex:
				edgePoints = adjacencyEdge.currentIndexEdgePoints
			else:
				edgePoints = adjacencyEdge.adjacentIndexEdgePoints

			for edgePoint in edgePoints:
				print("Plotting:", edgePoint.xy)
				self.ax.plot(edgePoint.xy[0], edgePoint.xy[1]*flip, marker='x', color=color)


	def genAdjacencyMap(self):
		# traversalMap = [ [-1,1], [0,1], [1,1], [-1, 0],  [1, 0],[-1, -1], [0, -1], [1, -1] ]
		for index in self.maskRasterCollection.keys():
			# adjancencyList = []
			# self.genAdjacencyMap[ index ] =  adjancencyList
			# edgePoints = self.regionEdgePoints[ index ]
			print("genAdjacencyMap Searching for regionEdgePoints index:", index)
			regionEdgePoints = self.regionEdgePoints[ index ]

			for edgePoint in regionEdgePoints.pointsOnEdge:
				currentPixel = edgePoint.xy
				currentPixel = [-int(currentPixel[1]), int(currentPixel[0])]  ## Switching from x,y to row, column
				for adjacentPixelRelativePosition in Bridson_Common.traversalMap:
					adjacentPixel = tuple([currentPixel[0]+adjacentPixelRelativePosition[0], currentPixel[1]+adjacentPixelRelativePosition[1] ])
					# print("genAdjacencyMap Searching for pixel:", adjacentPixel)
					if adjacentPixel in self.globalEdgePointMap: # Is the adjacent pixel an EdgePoint?
						adjacentEdgePoints = self.globalEdgePointMap[ adjacentPixel ]

						for adjacentEdgePoint in adjacentEdgePoints:
							print("genAdjacencyMap Found adjacent pixels: ", adjacentEdgePoint.xy, adjacentEdgePoint.regionIndex)
							adjacentIndex = adjacentEdgePoint.regionIndex

							if index != adjacentIndex:
								print("genAdjacencyMap Adding point")
								# Does the region to region map exist for current region?
								if index in self.regionAdjacentRegions:
									if  adjacentIndex not in self.regionAdjacentRegions[index]:
										self.regionAdjacentRegions[index].append(adjacentIndex)
								else:
									self.regionAdjacentRegions[ index ] = [ adjacentIndex ]

								# Does the region to region map exist for other region?
								if adjacentIndex in self.regionAdjacentRegions:
									if  index not in self.regionAdjacentRegions[adjacentIndex]:
										self.regionAdjacentRegions[adjacentIndex].append( index )
								else:
									self.regionAdjacentRegions[ adjacentIndex ] = [ index ]

								# Ensure starting index is lower than endingindex.
								startingIndex = index if index < adjacentIndex else adjacentIndex
								endingIndex = index if index > adjacentIndex else adjacentIndex

								# Populate the adjacency edge information.
								if (startingIndex,endingIndex) in self.regionAdjancencyMap:
									# Mapping already created.  Add new points.
									adjacencyEdge = self.regionAdjancencyMap[ (startingIndex, endingIndex)]
								else:
									# Mapping doesn't already exist.  Create new adjancency and populate.
									adjacencyEdge = AdjacencyEdge.AdjancecyEdge(startingIndex, endingIndex)
									self.regionAdjancencyMap[ (startingIndex, endingIndex)] = adjacencyEdge

								if edgePoint.regionIndex < adjacentEdgePoint.regionIndex:
									adjacencyEdge.currentIndexEdgePoints.append( edgePoint )
									adjacencyEdge.adjacentIndexEdgePoints.append( adjacentEdgePoint )
								else:
									adjacencyEdge.currentIndexEdgePoints.append(adjacentEdgePoint)
									adjacencyEdge.adjacentIndexEdgePoints.append(edgePoint)




			'''
				1. Iterate through each region.
				2. Create AdjacencyMap<starting region, end region> = [
							[starting region EdgePoint],
							[ending region EdgePoint]
							]

			'''




	def findLineEdgePoints(self, index, regionEdgePixels, croppedLinePoints):
		'''

		:param regionEdgeConnectivity: Object containing edge pixel information.
		:param croppedLinePoints: croppedLinePoints for this region.
		:return: Will populate regionEdgeConnectivity with line points that exist in the edge pixels.
		'''
		print("findLineEdgePoints edgePixels:", regionEdgePixels.edgePixelList)
		edgePixelList = regionEdgePixels.edgePixelList
		pointsOnEdge = []
		for line in croppedLinePoints:
			startPoint = line[0]
			searchValue = [-int(startPoint[1]), int(startPoint[0])] ## Switching from x,y to row, column
			print("findLineEdgePoints searching For value:", searchValue)
			if searchValue in edgePixelList:
				startEdgePoint = EdgePoint.EdgePoint( startPoint.copy(), line, index, 0)
				# pointsOnEdge.append( startPoint.copy() )
				pointsOnEdge.append( startEdgePoint )
				if tuple(searchValue) in self.globalEdgePointMap:
					self.globalEdgePointMap[tuple(searchValue)].append( startEdgePoint )
				else:
					self.globalEdgePointMap[ tuple(searchValue)] = [startEdgePoint]

			endPoint = line[-1]
			searchValue = [-int(endPoint[1]), int(endPoint[0])] ## Switching from x,y to row, column
			if searchValue in edgePixelList:
				endEdgePoint = EdgePoint.EdgePoint(endPoint.copy(), line, index, -1)
				pointsOnEdge.append( endEdgePoint )
				if tuple(searchValue) in self.globalEdgePointMap:
					self.globalEdgePointMap[tuple(searchValue)].append(endEdgePoint)
				else:
					self.globalEdgePointMap[ tuple(searchValue) ] = [endEdgePoint]

		regionEdgePixels.setPointOnEdge ( pointsOnEdge )
		print("findLineEdgePoints Points on Edge: ", pointsOnEdge)



	def displayDistanceMask(self, indexLabel, topLeftTarget, bottomRightTarget):
		distanceRaster = self.distanceRasters[ indexLabel ]

		plt.figure()
		ax = plt.subplot(1, 1, 1, aspect=1, label='Region Raster ' + str(indexLabel))
		plt.title('Distance Raster ' + str(indexLabel))
		''' Draw Letter blob '''

		# blankRaster = np.zeros(np.shape(imageraster))
		# ax3 = plt.subplot2grid(gridsize, (0, 1), rowspan=1)
		# ax3.imshow(blankRaster)
		# distanceRaster[5][5]=255 # Reference point

		print("DistanceRaster Display:\n",  distanceRaster[topLeftTarget[0]:bottomRightTarget[0]+1, topLeftTarget[1]:bottomRightTarget[1]+1 ] )
		ax.imshow(distanceRaster)
		# plt.plot(5, 5, color='r', markersize=10)
		ax.grid()


	def genDistancePixels(self, raster):
		distanceRaster = Bridson_Common.distance_from_edge(raster.copy())
		distance1pixelIndices = np.argwhere(distanceRaster == 1)  # Get pixels that have a distance of 1 from the edge.
		# distance1pixelIndices = np.argwhere((distanceRaster > 0) & (distanceRaster <= 2))  # Get pixels that have a distance of 2 or less, but greater than 0.
		return distanceRaster, distance1pixelIndices


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
		linePoints = meshObj.croppedLinePoints.copy()
		# topLeftSource = self.maxLinePoints(linePoints)

		# print("Region Coordinates:", regionCoordinates)
		# print("LinePoints:", linePoints)

		# Grab the shiftx and shifty based on regionMap.
		regionCoordinates = regionMap.get( index  )
		topLeftTarget, bottomRightTarget = SLIC.calculateTopLeft( regionCoordinates )
		# topLeftSource = self.maxLinePoints( linePoints )
		# shiftCoordinates = (bottomRightTarget[0] - topLeftSource[0], bottomRightTarget[1] - topLeftSource[1])

		currentLine = linePoints[0] * -100  # Set the starting line way off the current raster region.
		initial=True
		# count = 0

		# Allow regions to be blank.
		if Bridson_Common.allowBlankRegion == True:
			if regionIntensity > 250:
				return
			# if self.calculateLineSpacing(linePoints[0], linePoints[-1], intensity=regionIntensity) == False:
			# 	return

		for lineIndex in range(len(linePoints)):
			# if count > 5:
			# 	break
			# if lineIndex % Bridson_Common.lineSkip == 0:
			colour = Bridson_Common.colourArray[ (lineIndex % len(Bridson_Common.colourArray) ) ]
			line = linePoints[lineIndex].copy()
			line = line * Bridson_Common.mergeScale
			# For some reason we need to swap the topLeft x,y with the line x,y.
			############## Shifting the lines.
			# line[:, 0] = line[:, 0] + topLeftTarget[1] - 5 # Needed to line up with regions.
			# line[:, 1] = line[:, 1] - topLeftTarget[0] + 5  # Required to be negative.

			# if self.calculateLineSpacing(currentLine, line, intensity=regionIntensity) == True:
			if True:
				self.ax.plot(line[:, 0], line[:, 1]*flip, color=colour)
				if Bridson_Common.closestPointPair:  # Only place the dots when we are calculating closest point pair.
					if initial == False:  # Do not display this dot the first time around.
						self.ax.plot(currentLine[self.markPoint[0]][0], currentLine[self.markPoint[0]][1] * flip, marker='*',
						             markersize=6, color='g')  # Colour middle point.
					self.ax.plot(line[self.markPoint[1]][0], line[self.markPoint[1]][1]*flip, marker='o', markersize=2, color='r')  # Colour middle point.
				currentLine = line
				initial = False
					# count += 1

		if Bridson_Common.highlightEndpoints:
			regionEdgePoints = self.regionEdgePoints[ index ]
			for edgePoint in regionEdgePoints.pointsOnEdge:
				self.ax.plot(edgePoint.xy[0], edgePoint.xy[1]*flip, marker='x', color='g', markersize=4)


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
		print("Distance:", distance, intensityDistance)
		# if distance > Bridson_Common.dradius*factor:
		if distance > intensityDistance:
			print("Far enough")
			return True
		else:
			print("Too close")
			return False