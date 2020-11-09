
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
import copy
import itertools
import os
import Bridson_StructTensor
import Bridson_ColourOperations
import cv2 as cv
from skimage.future import graph
import Bridson_Angles
from numba import njit, jit_module

np.set_printoptions(threshold=sys.maxsize)  # allow printing without ellipsis: https://stackoverflow.com/questions/44311664/print-numpy-array-without-ellipsis

class FinishedImage:
	def __init__(self, *args, **kwargs):
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(1, 1, 1, aspect=1)
		self.ax.set_title('Merged Image' )
		self.ax.grid()
		self.tempLines = []
		# self.set
		# self.ax.invert_yaxis()

	def setTitle(self, filename):
		if Bridson_Common.SLIC0:
			self.ax.set_title('Merged Image - ' + filename + '\nSegments: ' + str(Bridson_Common.segmentCount) + ' - compactness: SLIC0' )
		else:
			self.ax.set_title('Merged Image - ' + filename + '\nSegments: ' + str(Bridson_Common.segmentCount) + ' - compactness: ' + str(Bridson_Common.compactnessSLIC) )

	def setXLimit(self, left, right):
		# print("Left:", left, "Right:", right)
		self.ax.set_xlim(left=left, right=right)

	def setYLimit(self, top, bottom):
		# print("Left:", left, "Right:", right)
		self.ax.set_ylim(top=top, bottom=bottom)

	def copyFromOther(self, otherFinishedImage):

		if hasattr(otherFinishedImage, 'unshiftedImageMaskedRegion') : self.unshiftedImageMaskedRegion = otherFinishedImage.unshiftedImageMaskedRegion.copy()
		if hasattr(otherFinishedImage, 'shiftedMaskRasterCollection'): self.shiftedMaskRasterCollection = otherFinishedImage.shiftedMaskRasterCollection.copy()
		if hasattr(otherFinishedImage, 'maskRasterCollection'): self.maskRasterCollection = otherFinishedImage.maskRasterCollection.copy()
		if hasattr(otherFinishedImage, 'meshObjCollection'): self.meshObjCollection = otherFinishedImage.meshObjCollection.copy()
		if hasattr(otherFinishedImage, 'regionEdgePoints'): self.regionEdgePoints = otherFinishedImage.regionEdgePoints.copy()
		if hasattr(otherFinishedImage, 'distanceRasters'): self.distanceRasters = otherFinishedImage.distanceRasters.copy()
		if hasattr(otherFinishedImage, 'globalEdgePointMap'): self.globalEdgePointMap = otherFinishedImage.globalEdgePointMap.copy()

		if hasattr(otherFinishedImage, 'regionEdgePointMap'): self.regionEdgePointMap =  otherFinishedImage.regionEdgePointMap.copy() # Map of (x,y) coordinates
		if hasattr(otherFinishedImage, 'regionAdjancencyMap'): self.regionAdjancencyMap =  otherFinishedImage.regionAdjancencyMap.copy() # Map to contain AdjancencyEdge objects
		if hasattr(otherFinishedImage, 'regionAdjacentRegions'): self.regionAdjacentRegions = otherFinishedImage.regionAdjacentRegions.copy() # Map containing list of adjacent regions for current region.
		if hasattr(otherFinishedImage, 'regionMap'): self.regionMap = otherFinishedImage.regionMap.copy()
		if hasattr(otherFinishedImage, 'regionIntensityMap'): self.regionIntensityMap = otherFinishedImage.regionIntensityMap.copy()
		if hasattr(otherFinishedImage, 'regionDirection'): self.regionDirection = otherFinishedImage.regionDirection.copy()

		if hasattr(otherFinishedImage, 'regionCoherency'): self.regionCoherency = otherFinishedImage.regionCoherency.copy()
		if hasattr(otherFinishedImage, 'diffAttractThreshold'): self.diffAttractThreshold = otherFinishedImage.diffAttractThreshold
		if hasattr(otherFinishedImage, 'diffRepelThreshold'): self.diffRepelThreshold = otherFinishedImage.diffRepelThreshold
		if hasattr(otherFinishedImage, 'stableThreshold'): self.stableThreshold = otherFinishedImage.stableThreshold

	def cropContourLines(self, linePoints, raster, topLeftTarget):
		'''
			Operations will contain a combination of - and + symbols.  The - indicate skipping a point.  The + indicate adding a point.
			To check if we have a situation where we add, skip, then add again, we will split the string based on add.
			We will then look for +-+ in the string.
			Use approach here to remove sequentially duplicate values: https://stackoverflow.com/questions/39469691/python-merge-repeating-characters-ins-sequence-in-string

		'''
		# print("Length of raster:", len(raster))
		if Bridson_Common.cropContours:
			# Raster will have 255 for valid points that should be retained.
			newLinePoints = []
			for line in linePoints:
				operations = ''
				newLine = []
				emptyLine = True
				for point in line:
					x, y = int(point[0]), int(0-point[1])
					# print("Point:", x,y)
					try:
						if y < len(raster) and x < len(raster[y]) and raster[y][x] == 255: # Because of the differing coordinates system, we have to flip the order of x,y
							# print("Point:", point)
							# print("Point in Mask:", x, y)
							newLine.append( point )
							emptyLine = False
							operations = operations + '+'
						else:
							operations = operations + '-'
					except:
						operations = operations + '-'
						# if Bridson_Common.debug:
						if True:
							print("********************* Problem with cropping contour lines. **********************")
							print("X,Y:", x,y)
							print("Raster dimensions:", np.shape(raster))
							print("Raster:", raster)

					# print("Operations:", operations)

				# Merge repeat characters.  Use this to detect region, no region, region.
				operations = ''.join(ch for ch, _ in itertools.groupby(operations))
				# print("shrunk Operations:", operations)
				if operations.find('+-+') > -1:
					emptyLine = True # If we detect add, skip, add sequence, we avoid adding this line.

				if emptyLine == False:
					newLinePoints.append(np.array(newLine))

			newLinePoints = np.array(newLinePoints)
			return newLinePoints
		else:
			return linePoints



	def findMiddleLine(self, linePoints):

		pass


	def findLongestLine(self, linePoints):
		longestLength = -1
		longestIndex = -1
		for lineIndex in range(len(linePoints)):
			line = linePoints[lineIndex]
			lineDistance = 0
			for pointIndex in range(len(line) - 1):
				lineDistance += Bridson_Common.euclidean_distance(line[pointIndex], line[pointIndex+1])

			if lineDistance > longestLength:
				longestLength = lineDistance
				longestIndex = lineIndex

		# Rearrange the lines so that our starting index is at the longest line.
		linePoints = np.roll(linePoints, -longestIndex)

		return linePoints

	def cullLines(self, linePoints, regionIntensity):
		'''
			We want to cull the lines based on a distance between the lines.
			When the lines are vertical, we can sort the line order by x coordinates.
			When the lines are not vertical, we have a problem with sorting the lines.

		:param linePoints:
		:param regionIntensity:
		:return:
		'''
		newLinePoints = []
		empty = True
		if len(linePoints) == 0:
			# Handle the situation that the line points list is empty.
			return empty, newLinePoints

		# Allow regions to be blank.
		if Bridson_Common.allowBlankRegion == True:
			if regionIntensity >= Bridson_Common.cullingBlankThreshold:
				return empty, newLinePoints
			# if self.calculateLineSpacing(linePoints[0], linePoints[-1], intensity=regionIntensity) == False:
			# 	return

		# Find the longest line.  Rotate the array such that the longest line is the first element.
		linePoints = self.findLongestLine( linePoints )

		currentLine = linePoints[-1]*-100
		for lineIndex in range(len(linePoints)):
			# if lineIndex % Bridson_Common.lineSkip == 0:
			line = linePoints[lineIndex].copy()
			line = line * Bridson_Common.mergeScale
			# newLinePoints.append( line )
			# print("Handling Line index:", lineIndex)
			# print("cullLines:", line[0], line[-1])
			if self.calculateLineSpacing(currentLine, line, intensity=regionIntensity) == True:
				# The line has passed the initial check against the previous line.
				# Now have to check against existing lines.
				if len(newLinePoints) > 0:
					flippedLine = np.flip(line, axis=0)
					farEnough = True
					for addedLine in newLinePoints:
						if self.calculateLineSpacing(addedLine, line, intensity=regionIntensity) == True and self.calculateLineSpacing(addedLine, np.flip(line, axis=0), intensity=regionIntensity) == True:
							pass
						else:
							farEnough = False
					# The line is not too close to other added lines.
					if farEnough:
						newLinePoints.append( line )
						empty=False
						currentLine = line
				else:
					newLinePoints.append(line)
					empty = False
					currentLine = line

		# print("cullLines empty:", empty)
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
		# newRegionRaster = Bridson_Common.scaleArray(regionRaster)
		# newSegments = Bridson_Common.scaleArray(segments)
		self.ax.imshow(mark_boundaries(regionRaster, segments, color=(214/255, 214/255, 136/255)))
		# self.ax.imshow(mark_boundaries(newRegionRaster, newSegments, color=(214 / 255, 214 / 255, 136 / 255)))
		# self.ax.grid()


	def shiftRastersMeshObj(self, regionMap, regionRaster, originalImage):
		# Will shift the raster to actual x,y
		# Will create the masked region of the original image.  Will unshift imageMaskedRegion for Structure Tensor.
		self.unshiftedImageMaskedRegion = {}
		self.shiftedMaskRasterCollection = {}
		for index in self.maskRasterCollection.keys():
			raster = self.maskRasterCollection[ index ]

			# Only process the region is it exists.  Can fail if trifinder is not generated.
			# meshObj = self.meshObjCollection[ index ]

			# print("Shape of Region Raster:", np.shape(regionRaster))

			regionCoordinates = regionMap.get(index)
			topLeftTarget, bottomRightTarget = SLIC.calculateTopLeft(regionCoordinates)

			# Create new raster that has been shifted.
			x, y = np.shape(raster)
			# linePoints = meshObj.linePoints
			shiftedRaster = np.zeros(np.shape(regionRaster))
			# print("Shape of raster:", (x,y))
			for i in range(x):
				for j in range(y):
					# newRaster[i - topLeftTarget[0] - 3 ][j + topLeftTarget[1] - 5 ] = raster[i][j]
					if raster[i][j] != 0: # Only shift the points in the raster that have non-zero values.
						shiftedRaster[i + topLeftTarget[0] - 5 ][j + topLeftTarget[1] - 5 ] = raster[i][j]

			# meshObj.linePoints = newLinePoints
			self.shiftedMaskRasterCollection[index] = shiftedRaster

			# Create the image mask of original image.
			maxValue = np.max(shiftedRaster)
			# print("Max Value:", maxValue)
			normalizedMask = shiftedRaster / maxValue
			shiftedMaskedImage = normalizedMask * originalImage

			# print("Creating unshiftedMaskedImage")
			# print("Shape of Raster:", np.shape(raster))
			# Create new region that is the same size as the original raster.
			unshiftedMaskedImage = np.zeros( np.shape(raster) )
			# Create the unshifted image mask of original image.
			for i in range(x):
				for j in range(y):
					# newRaster[i - topLeftTarget[0] - 3 ][j + topLeftTarget[1] - 5 ] = raster[i][j]
					if raster[i][j] != 0: # Only shift the points in the raster that have non-zero values.
						unshiftedMaskedImage[i][j] = shiftedMaskedImage[i + topLeftTarget[0] - 5 ][j + topLeftTarget[1] - 5 ]

			# print("unshiftedMaskedImage created.")
			self.unshiftedImageMaskedRegion[index] = unshiftedMaskedImage
			# print("UnshiftedMaskedImage:", unshiftedMaskedImage)

			# newLinePoints = []
			# for line in linePoints:
			# 	newline = line.copy()
			# 	newline[:, 0] = line[:, 0] + topLeftTarget[1] - 5 # Needed to line up with regions. Affects the x axis.
			# 	newline[:, 1] = line[:, 1] - topLeftTarget[0] + 5   # Required to be negative.  Affects the y axis.
			# 	newLinePoints.append( newline )

			if Bridson_Common.displayMesh:
				# plt.figure()
				# ax = plt.subplot(1, 1, 1, aspect=1)
				# plt.title('shifted Lines for region ' + str(index))
				# plt.grid()
				# for line in newLinePoints:
				# 	ax.plot(line[:, 0], line[:, 1], color='r', marker='*')

				plt.figure()
				plt.subplot(1, 1, 1, aspect=1)
				plt.title('shifted Mask for region ' + str(index))
				plt.grid()
				plt.imshow(shiftedRaster)




	def shiftLinePoints(self):
		# for index in [5]:
		# for index in self.maskRasterCollection.keys():

		for index in self.meshObjCollection.keys():
			raster = self.maskRasterCollection[ index ]

			# Only process the region is it exists.  Can fail if trifinder is not generated.
			meshObj = self.meshObjCollection[ index ]

			# print("Shape of Region Raster:", np.shape(regionRaster))

			regionCoordinates = self.regionMap.get(index)
			topLeftTarget, bottomRightTarget = SLIC.calculateTopLeft(regionCoordinates)

			# Create new raster that has been shifted.
			x, y = np.shape(raster)
			linePoints = meshObj.linePoints
			# newRaster = np.zeros(np.shape(regionRaster))
			# print("Shape of raster:", (x,y))
			# for i in range(x):
			# 	for j in range(y):
			# 		# newRaster[i - topLeftTarget[0] - 3 ][j + topLeftTarget[1] - 5 ] = raster[i][j]
			# 		if raster[i][j] != 0: # Only shift the points in the raster that have non-zero values.
			# 			newRaster[i + topLeftTarget[0] - 5 ][j + topLeftTarget[1] - 5 ] = raster[i][j]

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

				# plt.figure()
				# plt.subplot(1, 1, 1, aspect=1)
				# plt.title('shifted Mask for region ' + str(index))
				# plt.grid()
				# plt.imshow(newRaster)


			meshObj.linePoints = newLinePoints
			# self.shiftedMaskRasterCollection[index] = newRaster

	def setCollections(self, maskRasterCollection, meshObjCollection, regionEdgePoints, distanceRasters ):
		self.maskRasterCollection = maskRasterCollection
		self.meshObjCollection = meshObjCollection
		self.regionEdgePoints = regionEdgePoints
		self.distanceRasters = distanceRasters


	def mergeLines(self):
		# Iterate through each region.
		for index in self.regionAdjacentRegions.keys(): # Iterate through regions that have adjacencies defined.
			# Obtain the adjacent regions for the current index.
			adjacentRegions = self.regionAdjacentRegions[ index ]
			for adjacentIndex in adjacentRegions:
				startingIndex = index if index < adjacentIndex else adjacentIndex
				endingIndex = index if index > adjacentIndex else adjacentIndex

				# Obtain the adjacencyEdge.
				if (startingIndex, endingIndex) in self.regionAdjancencyMap:
					# Only continue if the startingIndex, endingIndex pair exists.
					adjacencyEdge = self.regionAdjancencyMap[ (startingIndex, endingIndex) ]

					# Get the current region EdgePoints
					if index < adjacentIndex:
						currentIndexEdgePoints = copy.copy(adjacencyEdge.currentIndexEdgePoints)
						adjacentIndexEdgePoints = copy.copy(adjacencyEdge.adjacentIndexEdgePoints)
					else:
						adjacentIndexEdgePoints = copy.copy(adjacencyEdge.currentIndexEdgePoints)
						currentIndexEdgePoints = copy.copy(adjacencyEdge.adjacentIndexEdgePoints)


					currentPoints = self.constructLocationForEdgePoints(currentIndexEdgePoints)
					adjacentPoints = self.constructLocationForEdgePoints(adjacentIndexEdgePoints)

					while len(currentPoints) > 0 and len(adjacentPoints) > 0:
						# print("CurrentPoints:", currentPoints)
						# print("AdjacentPoints:", adjacentPoints)

						shortestPair = Bridson_Common.findClosestIndex(currentPoints, adjacentPoints)

						# print("Pair:", currentPoints[shortestPair[0][0]], adjacentPoints[shortestPair[1][0]])
						# Pair the two EdgePoints.
						connected = self.connectTwoPoints( currentIndexEdgePoints[shortestPair[0][0]], adjacentIndexEdgePoints[shortestPair[1][0]])
						# Increase the distance for the point pair so that they are never close again.
						if connected == False:
							# Exit the for loop because the closest points are now too far apart.
							break

						currentPoints.pop(shortestPair[0][0] ) # Remove the paired points from the list.
						adjacentPoints.pop(shortestPair[1][0] ) # Remove the paired points from the list.
						currentIndexEdgePoints.pop(shortestPair[0][0]) # Remove the paired points from the list.
						adjacentIndexEdgePoints.pop( shortestPair[1][0]) # Remove the paired points from the list.
				# currentPoints[ shortestPair[0][0] ] = (-bigDistance,-bigDistance)
				# adjacentPoints[ shortestPair[1][0] ] = (bigDistance, bigDistance)


				# Remove the EdgePoints from the lists.


	def mergeLines2(self):
		# Iterate through each region.
		count = 0
		self.processedRegions = {}
		for index in self.regionAdjacentRegions.keys(): # Iterate through regions that have adjacencies defined.
			# Obtain the adjacent regions for the current index.
			adjacentRegions = self.regionAdjacentRegions[ index ]
			for adjacentIndex in adjacentRegions:
				startingIndex = index if index < adjacentIndex else adjacentIndex
				endingIndex = index if index > adjacentIndex else adjacentIndex

				if (startingIndex, endingIndex) not in self.processedRegions.keys():
				# Obtain the adjacencyEdge.
					if (startingIndex, endingIndex) in self.regionAdjancencyMap:
						# Only continue if the startingIndex, endingIndex pair exists.
						adjacencyEdge = self.regionAdjancencyMap[ (startingIndex, endingIndex) ]

						# Get the current region EdgePoints
						if index < adjacentIndex:
							currentIndexEdgePoints = copy.copy(adjacencyEdge.currentIndexEdgePoints)
							adjacentIndexEdgePoints = copy.copy(adjacencyEdge.adjacentIndexEdgePoints)
						else:
							adjacentIndexEdgePoints = copy.copy(adjacencyEdge.currentIndexEdgePoints)
							currentIndexEdgePoints = copy.copy(adjacencyEdge.adjacentIndexEdgePoints)

						currentPoints = self.constructLocationForEdgePoints(currentIndexEdgePoints)
						adjacentPoints = self.constructLocationForEdgePoints(adjacentIndexEdgePoints)


						self.pairAllAND2(currentPoints, adjacentPoints, count, threshold=Bridson_Common.dradius*Bridson_Common.mergePairFactor)
						# self.pairAllOR(currentPoints, adjacentPoints, threshold=5.0)
						#*************************************** Diagnostic display
						if Bridson_Common.diagnosticDisplay:
							if count < Bridson_Common.diagnosticDisplayCount:
								print("******** ************* ******** mergeLines2 count:", count)
								self.displayAllPairs()
								plt.title("Count " + str(count) )
							count += 1

						# We want to iterate through the clusters and then perform the same logic as connectTwoPoints.
						#HERE
						self.connectPoints(currentIndexEdgePoints, adjacentIndexEdgePoints)

				# Add the region pairs to the list of processed pairs.
				self.processedRegions[ (startingIndex, endingIndex) ] = 1
				self.processedRegions[(endingIndex, startingIndex)] = 1


		if Bridson_Common.diagnosticDisplay:
			plt.show()

					# Replace with pairAllOR or pairAllAND
					# while len(currentPoints) > 0 and len(adjacentPoints) > 0:
					# 	# print("CurrentPoints:", currentPoints)
					# 	# print("AdjacentPoints:", adjacentPoints)
					#
					# 	shortestPair = Bridson_Common.findClosestIndex(currentPoints, adjacentPoints)
					#
					# 	# print("Pair:", currentPoints[shortestPair[0][0]], adjacentPoints[shortestPair[1][0]])
					# 	# Pair the two EdgePoints.
					# 	connected = self.connectTwoPoints( currentIndexEdgePoints[shortestPair[0][0]], adjacentIndexEdgePoints[shortestPair[1][0]])
					# 	# Increase the distance for the point pair so that they are never close again.
					# 	if connected == False:
					# 		# Exit the for loop because the closest points are now too far apart.
					# 		break
					#
					# 	currentPoints.pop(shortestPair[0][0] ) # Remove the paired points from the list.
					# 	adjacentPoints.pop(shortestPair[1][0] ) # Remove the paired points from the list.
					# 	currentIndexEdgePoints.pop(shortestPair[0][0]) # Remove the paired points from the list.
					# 	adjacentIndexEdgePoints.pop( shortestPair[1][0]) # Remove the paired points from the list.
				# currentPoints[ shortestPair[0][0] ] = (-bigDistance,-bigDistance)
				# adjacentPoints[ shortestPair[1][0] ] = (bigDistance, bigDistance)


				# Remove the EdgePoints from the lists.




	def pairAllAND(self, s1, s2, allowDangle=False, threshold=1.5):
		self.s0 = s1
		self.s1 = s2
		# print("Bridson_Common:findClosestIndex s1:", s1)
		# print("Bridson_Common:findClosestIndex s2:", s2)
		# both s1 and s2 should be 2D.

		# Want to pair all points.
		# Iterate through all the points until everything has been paired.
		# print("Original points:", s1, s2)

		firstListMoved = list(range(len(s1)))
		secondListMoved = list(range(len(s2)))

		distances = distance.cdist(s1, s2)
		allDistances = np.sort( distances.flatten() )

		# print("Distances:", distances)
		# shortestDistances = distance.cdist(s1, s2).min(axis=1)
		allPairings = []
		side0Cluster = {}
		side1Cluster = {}
		clusterGroups = {}


		currentCluster = 0

		# Pass 1 - OR logic.
		for currentDistance in allDistances:
			if currentDistance > threshold:
				# If the currentDistance is greater than the threshold, skip this pairing.
				continue
			# print("Current Distance:", currentDistance)
			location = np.where(distances == currentDistance)
			# print("Location:", location)
			index0 = location[0][0]
			index1 = location[1][0]
			# print("Location0:", index0)
			# print("Location1:", index1)


			if firstListMoved[index0] == -1 and secondListMoved[index1] == -1:
			# if firstListMoved[index0] == -1 or secondListMoved[index1] == -1:
				# The two points have already been merged.  Do nothing.
				pass
			else:
				# At least one of the points has not been merged.
				allPairings.append(location)
				firstListMoved[index0] = -1
				secondListMoved[index1] = -1
				# print("Pairing:", location, " between points: ", s1[index0], s2[index1])
				# If either index already exists in a cluster
				if index0 in side0Cluster.keys():
					# Index already exists.  Get the
					existingCluster = side0Cluster[ index0 ]
				elif index1 in side1Cluster.keys():
					existingCluster = side1Cluster[ index1 ]
				else:
					# Create new cluster.
					existingCluster = currentCluster
					currentCluster += 1
				side0Cluster[ index0 ] = existingCluster
				side1Cluster[ index1 ] = existingCluster
				if not existingCluster in clusterGroups.keys():
					newList = [ (index0, index1) ]
					clusterGroups[ existingCluster ] = newList
				else:
					existingList = clusterGroups[ existingCluster ]
					existingList.append( (index0, index1) )

		# print("** side0Cluster:", side0Cluster)
		# print("** side1Cluster:", side1Cluster)
		# print("** clusterGroups:", clusterGroups)


		# print("** side0Cluster:", side0Cluster)
		# print("** side1Cluster:", side1Cluster)
		# print("** clusterGroups:", clusterGroups)
		self.side0Cluster = side0Cluster
		self.side1Cluster = side1Cluster
		self.clusterGroups = clusterGroups

		self.allPairs = allPairings

		self.averageClusters()
		# plt.show()

	def simpleUnique(self, a):
		newList = []
		for element in a:
			element = list(element)
			# print("Element:", element)
			if element not in newList:
				newList.append(element)
		newList = np.array(newList)
		return newList

	def pairAllAND2(self, s1, s2, count, threshold=1.5):
		# s1 = self.simpleUnique(s1)
		# s2 = self.simpleUnique(s2)
		self.s0 = s1
		self.s1 = s2
		# print("Bridson_Common:findClosestIndex s1:", s1)
		# print("Bridson_Common:findClosestIndex s2:", s2)
		# both s1 and s2 should be 2D.

		# Want to pair all points.
		# Iterate through all the points until everything has been paired.
		if Bridson_Common.diagnosticDisplay:
			if count < Bridson_Common.diagnosticDisplayCount:
				print("Original s1:", s1)
				print("Original s2:", s2)

		firstListMoved = len(s1)*[-1]
		secondListMoved = len(s2)*[-1]

		distances = distance.cdist(s1, s2)
		allDistances = np.sort( np.unique(distances.flatten()) )

		# print("Distances:", distances)
		# shortestDistances = distance.cdist(s1, s2).min(axis=1)
		allPairings = []
		side0Cluster = {}
		side1Cluster = {}
		clusterGroups = {}
		self.side0Cluster = side0Cluster
		self.side1Cluster = side1Cluster
		self.clusterGroups = clusterGroups

		currentCluster = 0

		# Pass 1 - OR logic.
		for currentDistance in allDistances:
			if currentDistance > threshold:
				# If the currentDistance is greater than the threshold, skip this pairing.
				continue
			# print("Current Distance:", currentDistance)
			location = np.where(distances == currentDistance)
			# print("Location:", location)
			for index in range(len(location[0])):
				index0 = location[0][index]
				index1 = location[1][index]
				# print("Location0:", index0)
				# print("Location1:", index1)
				addToCluster = False

				if firstListMoved[index0] != -1 and secondListMoved[index1] != -1:
				# if firstListMoved[index0] == -1 or secondListMoved[index1] == -1:
					# The two points have already been merged.  Do nothing.
					addToCluster = False
				elif firstListMoved[index0] != -1 and secondListMoved[index1] == -1:
					# First already belongs to a cluster.
					left, right  = self.clusterBalance( firstListMoved[index0] )
					if left == 1:
						addToCluster = True
				elif firstListMoved[index0] == -1 and secondListMoved[index1] != -1:
					# Second belongs to a cluster
					left, right = self.clusterBalance( secondListMoved[index1] )
					if right == 1:
						addToCluster = True
				else:
					addToCluster = True


				if addToCluster == True:
					# At least one of the points has not been merged.
					# allPairings.append(location)
					allPairings.append((np.array([index0]), np.array([index1])))  # Separate the pairings in the case there are multiple index pairs.
					# print("Pairing:", location, " between points: ", s1[index0], s2[index1])
					# If either index already exists in a cluster
					if index0 in side0Cluster.keys():
						# Index already exists.  Get the
						existingCluster = side0Cluster[ index0 ]
					elif index1 in side1Cluster.keys():
						existingCluster = side1Cluster[ index1 ]
					else:
						# Create new cluster.
						existingCluster = currentCluster
						currentCluster += 1
					side0Cluster[ index0 ] = existingCluster
					side1Cluster[ index1 ] = existingCluster

					firstListMoved[index0] = existingCluster
					secondListMoved[index1] = existingCluster

					if not existingCluster in clusterGroups.keys():
						newList = [ (index0, index1) ]
						clusterGroups[ existingCluster ] = newList
					else:
						existingList = clusterGroups[ existingCluster ]
						existingList.append( (index0, index1) )

		# print("** side0Cluster:", side0Cluster)
		# print("** side1Cluster:", side1Cluster)
		# print("** clusterGroups:", clusterGroups)


		# print("** side0Cluster:", side0Cluster)
		# print("** side1Cluster:", side1Cluster)
		# print("** clusterGroups:", clusterGroups)

		self.allPairs = allPairings

		self.averageClusters()
		# plt.show()


	def clusterBalance(self, clusterIndex):
		# Returns the tuple: # elements in list 1, # elements in list 2.
		clusterListing = self.clusterGroups[ clusterIndex ]
		# print("clusterListing:", clusterListing)
		group1 = []
		group2 = []
		for pair in clusterListing:
			left, right = pair
			group1.append( left )
			group2.append( right )

		group1 = set(group1)
		group2 = set(group2)
		balance = ( len(group1), len(group2) )
		# print("Cluster Balance:", balance )
		return balance


	def pairAllOR(self, s1, s2, threshold=1.5):
		self.s0 = s1
		self.s1 = s2
		# print("Bridson_Common:findClosestIndex s1:", s1)
		# print("Bridson_Common:findClosestIndex s2:", s2)
		# both s1 and s2 should be 2D.

		# Want to pair all points.
		# Iterate through all the points until everything has been paired.
		# print("Original points:", s1, s2)

		firstListMoved = list(range(len(s1)))
		secondListMoved = list(range(len(s2)))

		distances = distance.cdist(s1, s2)
		allDistances = np.sort( distances.flatten() )

		# print("Distances:", distances)
		# shortestDistances = distance.cdist(s1, s2).min(axis=1)
		allPairings = []
		side0Cluster = {}
		side1Cluster = {}
		clusterGroups = {}


		currentCluster = 0

		# Pass 1 - OR logic.
		for currentDistance in allDistances:
			if currentDistance > threshold:
				continue
			# print("Current Distance:", currentDistance)
			location = np.where(distances == currentDistance)
			# print("Location:", location)
			index0 = location[0][0]
			index1 = location[1][0]
			# print("Location0:", index0)
			# print("Location1:", index1)


			# if firstListMoved[index0] == -1 and secondListMoved[index1] == -1:
			if firstListMoved[index0] == -1 or secondListMoved[index1] == -1:
				# The two points have already been merged.  Do nothing.
				pass
			else:
				# At least one of the points has not been merged.
				allPairings.append(location)
				firstListMoved[index0] = -1
				secondListMoved[index1] = -1
				# print("Pairing:", location, " between points: ", s1[index0], s2[index1])
				# If either index already exists in a cluster
				if index0 in side0Cluster.keys():
					# Index already exists.  Get the
					existingCluster = side0Cluster[ index0 ]
				elif index1 in side1Cluster.keys():
					existingCluster = side1Cluster[ index1 ]
				else:
					# Create new cluster.
					existingCluster = currentCluster
					currentCluster += 1
				side0Cluster[ index0 ] = existingCluster
				side1Cluster[ index1 ] = existingCluster
				if not existingCluster in clusterGroups.keys():
					newList = [ (index0, index1) ]
					clusterGroups[ existingCluster ] = newList
				else:
					existingList = clusterGroups[ existingCluster ]
					existingList.append( (index0, index1) )

		# print("** side0Cluster:", side0Cluster)
		# print("** side1Cluster:", side1Cluster)
		# print("** clusterGroups:", clusterGroups)


		# print("** side0Cluster:", side0Cluster)
		# print("** side1Cluster:", side1Cluster)
		# print("** clusterGroups:", clusterGroups)
		self.side0Cluster = side0Cluster
		self.side1Cluster = side1Cluster
		self.clusterGroups = clusterGroups

		self.allPairs = allPairings

		self.averageClusters()
		# self.displayAllPairs()
		# plt.title('pairAllOR')
		# plt.show()


	def averageClusters(self):
		averagePoints = {}
		for cluster in self.clusterGroups.keys():
			averagePoint = np.array([0.0,0.0])
			values = []
			for indeces in self.clusterGroups[ cluster ]:
				averagePoint += np.array(self.s0[indeces[0]])
				averagePoint += np.array(self.s1[indeces[1]])
				values += indeces

			# print("Values:", values)
			averagePoints[ cluster ] =  list( averagePoint / len( values ) )

		self.averagePoints = averagePoints
		# print(">>> averagePoints:", self.averagePoints)

	def connectPoints(self, currentIndexEdgePoints, adjacentIndexEdgePoints):

		# print("************************  Starting connectPoints *************************")
		for cluster in self.clusterGroups.keys():
			# Obtain the indeces for the lines in their respective list.
			# print("Processing cluster:", cluster)
			# print("Current indeces:", self.clusterGroups[ cluster ])
			for indeces in self.clusterGroups[ cluster ]:
				# print("Indeces:", indeces)
				edgePoint1 = currentIndexEdgePoints[indeces[0]]
				edgePoint2 = adjacentIndexEdgePoints[indeces[1]]
				# print("EdgePoint1:", edgePoint1.associatedLine)
				# print("EdgePoint2:", edgePoint2.associatedLine)
				# print("Updating AssociatedLine 1:", edgePoint1.associatedLine[edgePoint1.pointIndex], "with value ", self.averagePoints[ cluster ])
				# print("Updating AssociatedLine 2:", edgePoint2.associatedLine[edgePoint2.pointIndex], "with value ", self.averagePoints[ cluster ])

				edgePoint1.associatedLine[edgePoint1.pointIndex] = self.averagePoints[ cluster ]
				edgePoint2.associatedLine[edgePoint2.pointIndex] = self.averagePoints[ cluster ]

		# print("************************  Ending connectPoints *************************")
		# distance = Bridson_Common.euclidean_distance(edgePoint1.xy, edgePoint2.xy)
		# if distance < 20.0:
		# 	# print("EdgePoint1:", edgePoint1)
		# 	# print("EdgePoint2:", edgePoint2)
		# 	xAvg = (edgePoint1.xy[0] + edgePoint2.xy[0]) / 2
		# 	yAvg = (edgePoint1.xy[1] + edgePoint2.xy[1]) / 2
		# 	edgePoint1.associatedLine[edgePoint1.pointIndex] = [xAvg, yAvg]
		# 	edgePoint2.associatedLine[edgePoint2.pointIndex] = [xAvg, yAvg]
		# 	return True
		# else:
		# 	return False


	def connectTwoPoints(self, edgePoint1, edgePoint2):
		distance = Bridson_Common.euclidean_distance(edgePoint1.xy, edgePoint2.xy)
		if distance < 20.0:
			# print("EdgePoint1:", edgePoint1)
			# print("EdgePoint2:", edgePoint2)
			xAvg = (edgePoint1.xy[0] + edgePoint2.xy[0]) / 2
			yAvg = (edgePoint1.xy[1] + edgePoint2.xy[1]) / 2
			edgePoint1.associatedLine[ edgePoint1.pointIndex ] = [xAvg, yAvg]
			edgePoint2.associatedLine[ edgePoint2.pointIndex ] = [xAvg, yAvg]
			return True
		else:
			return False


	def displayAllPairs( self ):
		plt.figure()

		for pair in self.allPairs:
			# print("Pair:", pair)
			index0 = pair[0][0]
			index1 = pair[1][0]
			# print("Index 0:", index0)
			# print("Index 1:", index1)

			# print("coordinates 1:", self.s0[index0])
			# print("coordinates 2:", self.s1[index1])
			xValues = [self.s0[index0][0], self.s1[index1][0]]
			yValues = [self.s0[index0][1], self.s1[index1][1]]
			# print("xValues:", xValues)
			# print("yValues:", yValues)
			plt.plot(xValues, yValues)

		for point in self.s0:
			plt.plot(point[0], point[1], color='r', marker='o', markersize=5)

		for point in self.s1:
			plt.plot(point[0], point[1], color='g', marker='o', markersize=5)

		for point in self.averagePoints.values():
			plt.plot(point[0], point[1], color='b', marker='*', markersize=5)





	def constructLocationForEdgePoints(self, edgePoints):
		newList = []
		for edgePoint in edgePoints:
			newList.append( edgePoint.xy )
		return newList


	def checkCroppedLines(self):
		emptyRegions = 0
		for index in self.meshObjCollection.keys():
			print("Checking region", index)
			meshObj = self.meshObjCollection[ index ]

			if meshObj.croppedLinePoints == None:
				print("CroppedLinePoints not initialized yet.")
				return

			if len(meshObj.croppedLinePoints) == 0:
				emptyRegion = emptyRegion + 1

		print("CroppedLinePoints Emptys regions: ", emptyRegions , "/", len(self.meshObjCollection.keys()  ) )


	def setMaps(self,regionMap, regionRaster, maskRasterCollection, meshObjCollection, regionIntensityMap):
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


	def cropCullLines(self ):
		# self.maskRasterCollection = maskRasterCollection
		# self.meshObjCollection = meshObjCollection
		# self.regionMap = regionMap
		# self.regionRaster = regionRaster
		# self.regionEdgePoints = {}
		# self.distanceRasters = {}
		# self.regionIntensityMap = regionIntensityMap
		# self.globalEdgePointMap = {} # Map of (x,y) coordinates that point to EdgePoint objects, if they exist.
		# self.regionEdgePointMap = {} # Map of (x,y) coordinates
		# self.regionAdjancencyMap = {} # Map to contain AdjancencyEdge objects
		# self.regionAdjacentRegions = {} # Map containing list of adjacent regions for current region.

		# self.shiftRastersMeshObj( regionMap, regionRaster )

		# For each region, determine the points on the lines that are close to the region edge.  Make a registry of these points.
		for index in self.meshObjCollection.keys():
			# print("Cropping index:", index)
			raster = self.shiftedMaskRasterCollection[ index ]
			meshObj = self.meshObjCollection[ index ]

			regionCoordinates = self.regionMap.get(index)
			topLeftTarget, bottomRightTarget = SLIC.calculateTopLeft(regionCoordinates)

			meshObj.setCroppedLines( self.cropContourLines(meshObj.linePoints, self.shiftedMaskRasterCollection[index], topLeftTarget) )
			# print("CropCullLines region croppedLines:", index, len(meshObj.croppedLinePoints) )

			empty, culledLines = self.cullLines(  meshObj.croppedLinePoints, self.regionIntensityMap[index])
			# meshObj.setCroppedLines( self.cullLines( index, meshObj.linePoints, regionIntensityMap[index] )  )
			# print("CropCullLines region croppedLines empty:", empty)
			if not empty:
				meshObj.setCroppedLines( culledLines )
			else:
				meshObj.setCroppedLines( [] )
			# print("Done cropping index:", index)

			# try:
			# We Generate the edge pixels and then create the edge connectivity object.
			distanceRaster, distance1pixelIndeces = self.genDistancePixels( raster )
			regionEdgePixels = RegionPixelConnectivity.RegionPixelConnectivity( distance1pixelIndeces )
			Bridson_Common.logDebug(__name__,"Setting regionEdgePoints index:" + str(index) )
			self.regionEdgePoints[ index ] = regionEdgePixels
			self.distanceRasters[ index ] = distanceRaster
			# Find the points that exist in the edge pixels.
			croppedLinePoints = meshObj.croppedLinePoints
			self.findLineEdgePoints(index, regionEdgePixels, croppedLinePoints)
			# except Exception as e:
			# 	# Display stack trace: https://stackoverflow.com/questions/438894/how-do-i-stop-a-program-when-an-exception-is-raised-in-python
			# 	print("Error calling cropCulllines:", e)
			# 	exc_type, exc_obj, exc_tb = sys.exc_info()
			# 	fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			# 	print(exc_type, fname, exc_tb.tb_lineno)
			# 	sys.exit(1)

	# self.displayDistanceMask( index, topLeftTarget, bottomRightTarget )
			###################################
			# DEBUG DEBUG : This section is for debugging purposes.
			# distanceMask = raster
			# Bridson_Common.displayDistanceMask(distanceMask, str(index), topLeftTarget, bottomRightTarget)
			####################################


	def highLightEdgePoints(self, color='g', drawSLICRegions=Bridson_Common.drawSLICRegions):
		for index in self.regionAdjacentRegions:
			# Get the list adjacencyEdge object.
			# adjancencyEdge = self.regionAdjancencyMap[ index ]
			if drawSLICRegions == False:
				flip = 1
			else:
				flip = -1


			adjacentRegions = self.regionAdjacentRegions[ index ]
			# print("highLightEdgePoints adjacenct regions: " , adjacentRegions)
			for adjacentIndex in adjacentRegions:
				startingIndex = index if index < adjacentIndex else adjacentIndex
				endingIndex = index if index > adjacentIndex else adjacentIndex
				adjacencyEdge = self.regionAdjancencyMap[ (startingIndex,endingIndex) ]

				if index < adjacentIndex:
					edgePoints = adjacencyEdge.currentIndexEdgePoints
				else:
					edgePoints = adjacencyEdge.adjacentIndexEdgePoints

				for edgePoint in edgePoints:
					# print("Plotting:", edgePoint.xy)
					self.ax.plot(edgePoint.xy[0], edgePoint.xy[1]*flip, marker='x', color=color)




	def calculateRegionDirection(self, regionList, iterations=0):
		self.regionDirection = {}
		self.regionCoherency = {}
		self.regionToRegions = {}
		'''
		1. Determine the direction of current region.
		2. Compare the regions and adjust directions.
		'''
		# Calculate initial direction.
		for index in regionList:
			# print("calculateRegionDirection1")
			unshiftedMaskedImage = self.unshiftedImageMaskedRegion[ index ].copy()
			# print("D")
			# print("calculateRegionDirection2")
			st = Bridson_StructTensor.ST(unshiftedMaskedImage)
			# print("E")
			# print("calculateRegionDirection3")
			direction, coherency = st.calculateEigenVector()
			# print("calculateRegionDirection4")
			self.regionDirection[ index ] = direction

			if coherency > Bridson_Common.coherencyThreshold:
				meshAngle = Bridson_Common.determineAngle(direction[0], direction[1]) + 90
			else:
				meshAngle = Bridson_Common.lineAngle + 90

			meshAngle = meshAngle % 360
			self.regionDirection[ index ] = meshAngle
			self.regionCoherency[ index ] = coherency
			# print("calculateRegionDirection5")
		# print("Region Coherency:", list(self.regionCoherency.values()) )
		# print("Size of Coherency:", len(list(self.regionCoherency.values())))
		# print("Region Intensity:", list(self.regionIntensityMap.values()) )



	def generateRAG(self, filename, segments, regionColourMap):
		# Generate RAG (Region Adjacency Graph)
		self.regionColourMap = regionColourMap
		originalImage = cv.imread(filename)
		rag = graph.rag_mean_color(originalImage, segments)  # Generate the Region AdjacencyGraph.
		self.rag = rag
		# print("RAG Nodes:", rag.nodes)
		# print("RAG Edges:", rag.edges)

		self.calculateRegionDifferences()


	def calculateRegionDifferences(self):
		'''
		Data structure
		1. Iterate through all regions.
		2. Calculate their "difference".
		3. Store their "difference" in a map.  Store the (startIndex,endIndex) key and the reverse direction.
		'''
		regionDifferences = {}
		for regionPair in self.rag.edges:
			startIndex, endIndex = regionPair
			if regionPair not in regionDifferences.keys():
				diff = Bridson_ColourOperations.diffCIEColours(self.regionColourMap[startIndex], self.regionColourMap[endIndex])
				regionDifferences[regionPair] = diff

			# Create region to region mapping.
			if startIndex not in self.regionToRegions.keys():
				self.regionToRegions[ startIndex ] = [ endIndex ]
			else:
				self.regionToRegions[ startIndex ].append( endIndex )

			if endIndex not in self.regionToRegions.keys():
				self.regionToRegions[ endIndex ] = [ startIndex ]
			else:
				self.regionToRegions[ endIndex ].append( startIndex )

		self.regionDifferences = regionDifferences
		# print("Region Differences:", self.regionDifferences)
		# print("regionToRegions:", self.regionToRegions)
		# Calculate the difference threshold.  Regions with differences below this threshold are
		# candidates for changing their direction.
		self.calculateThresholds()


	def calculateThresholds(self):

		if False:
			# Use the histogram bins.
			hist, bin_edges = np.histogram( list(self.regionDifferences.values()), bins=Bridson_Common.binSize)
			self.diffAttractThreshold = bin_edges[ Bridson_Common.attractionBin ]
			print("DiffAttractThreshold:", self.diffAttractThreshold)
			self.diffRepelThreshold = bin_edges[ Bridson_Common.repelBin ]
			print("DiffRepelThreshold:", self.diffRepelThreshold)

			hist, bin_edges = np.histogram( list(self.regionCoherency.values()) , bins=Bridson_Common.binSize)
			self.stableThreshold = bin_edges[ Bridson_Common.stableBin ]
			print("stable threshold:", self.stableThreshold)

		if True:
			# Percentile approach for calculating the thresholds.
			# print( "Region Differences Values:", list( self.regionDifferences.values() ))
			self.diffAttractThreshold = np.percentile(list( self.regionDifferences.values() ), Bridson_Common.diffAttractPercentile)
			print("diff attract threshold:", self.diffAttractThreshold)

			self.diffRepelThreshold = np.percentile(list( self.regionDifferences.values() ), Bridson_Common.diffRepelPercentile)
			print("diff repel threshold:", self.diffRepelThreshold)
			self.stableThreshold = np.percentile( list(self.regionCoherency.values()), Bridson_Common.stableCoherencyPercentile)
			print("stable threshold:", self.stableThreshold)



	def adjustRegionAngles2(self, iterations=Bridson_Common.angleAdjustIterations):
		invertNormalize = lambda x: (255 - x) / 255
		# print("Iterations:", iterations)
		# Iterate through all regions.
		# Check the coherency for stableRegionThreshold
		# Check the differences between the regions.
		# If the difference value is below the diffThreshold, average the angles.
		# print("Region Directions:", self.regionDirection)
		for i in range(iterations):
			newDirection = self.regionDirection.copy()
			for index in self.regionToRegions.keys():
				# print("Current region:", index)
				# print("region ", index, "coherency", self.regionCoherency[index] )
				if index not in self.regionCoherency.keys():
					print("Region", index, "not in coherency map")
				elif self.regionCoherency[ index ] < self.stableThreshold:
					# print("Not Stable region:", index)
					# If the region coherency is below the threshold, continue with checking
					adjacentRegions = self.regionToRegions[ index ]
					# print("Adjacency regions", adjacentRegions)
					AttractList = []
					RepelList = []
					for adjacentIndex in adjacentRegions:
						startIndex, endIndex = index, adjacentIndex
						if startIndex > endIndex:
							startIndex, endIndex = adjacentIndex, index

						pairIndex = (startIndex, endIndex)

						if pairIndex in self.regionDifferences.keys():
							# if self.regionDifferences[ pairIndex ] < self.diffAttractThreshold and invertNormalize(self.regionIntensityMap[index]) <= invertNormalize(self.regionIntensityMap[adjacentIndex])*Bridson_Common.attractFudge: # V2
							if self.regionDifferences[pairIndex] < self.diffAttractThreshold: # V3
								# Attract case.  Repeat this step twice.
								# AttractList.append( self.regionDirection[ adjacentIndex ] )
								# newDirection[index] = Bridson_Angles.calcAverageAngleWeighted(newDirection[index], self.regionDirection[adjacentIndex], self.regionCoherency[index], self.regionCoherency[adjacentIndex])
								newDirection[index] = Bridson_Angles.calcAverageAngleWeighted(newDirection[index], self.regionDirection[adjacentIndex], invertNormalize(self.regionIntensityMap[index]), invertNormalize(self.regionIntensityMap[adjacentIndex]) )
							# elif self.regionDifferences[ pairIndex ] >= self.diffRepelThreshold and invertNormalize(self.regionIntensityMap[index])*Bridson_Common.attractFudge > invertNormalize(self.regionIntensityMap[adjacentIndex]): # V2
							elif self.regionDifferences[ pairIndex ] >= self.diffRepelThreshold: # V3
								# Repel case.
								# RepelList.append( (self.regionDirection[ adjacentIndex ] + 90) % 360  )
								newDirection[index] = Bridson_Angles.calcAverageAngleWeighted(newDirection[index], (self.regionDirection[adjacentIndex] + 90) % 360, invertNormalize(self.regionIntensityMap[index]), invertNormalize(self.regionIntensityMap[adjacentIndex]) )

				elif self.regionCoherency[ index ] >= self.stableThreshold:
					# print("Stable region:", index)
					# The current region is stable.
					adjacentRegions = self.regionToRegions[index]
					# print("Adjacent regions:", adjacentRegions)
					for adjacentIndex in adjacentRegions:
						startIndex, endIndex = index, adjacentIndex
						if startIndex > endIndex:
							startIndex, endIndex = adjacentIndex, index

						pairIndex = (startIndex, endIndex)
						# print("Pair Index:", pairIndex)
						if pairIndex in self.regionDifferences.keys():
							if self.regionDifferences[ pairIndex ] < self.diffAttractThreshold and self.regionCoherency[adjacentIndex] < self.stableThreshold:
								if Bridson_Common.stableAttractSet:
									newDirection[adjacentIndex] = self.regionDirection[index]
								else:
									newDirection[adjacentIndex] = Bridson_Angles.calcAverageAngleWeighted(newDirection[adjacentIndex], self.regionDirection[index], invertNormalize(self.regionIntensityMap[adjacentIndex]), invertNormalize(self.regionIntensityMap[index]) * 2.0 )
								# print("Setting region", adjacentIndex, "to angle", newDirection[ adjacentIndex ] )
							# elif self.regionDifferences[ pairIndex ] >= self.diffRepelThreshold and invertNormalize(self.regionIntensityMap[index])*Bridson_Common.attractFudge > invertNormalize(self.regionIntensityMap[adjacentIndex]): # V2
							elif self.regionDifferences[pairIndex] >= self.diffRepelThreshold and self.regionCoherency[adjacentIndex] < self.stableThreshold: # V3
								# Force the repel of adjacent regions to perpendicular angle.
								newDirection[ adjacentIndex ] = Bridson_Angles.calcAverageAngleWeighted(newDirection[adjacentIndex], (self.regionDirection[index] + 90) % 360, invertNormalize(self.regionIntensityMap[adjacentIndex]), invertNormalize(self.regionIntensityMap[index])*2.0 )
								# newDirection[adjacentIndex] = (self.regionDirection[index] + 90) % 360
								# print("Region", index, "repels region", adjacentIndex)
								# print("Setting region", adjacentIndex, "to angle", newDirection[adjacentIndex])
						else:
							print("PairIndex does not exist:", pairIndex)

			self.regionDirection = newDirection



	def genLineAdjacencyMap(self):
		# traversalMap = [ [-1,1], [0,1], [1,1], [-1, 0],  [1, 0],[-1, -1], [0, -1], [1, -1] ]
		for index in self.meshObjCollection.keys():
			# adjancencyList = []
			# self.genLineAdjacencyMap[ index ] =  adjancencyList
			# edgePoints = self.regionEdgePoints[ index ]
			# print("genLineAdjacencyMap Searching for regionEdgePoints index:", index)
			regionEdgePoints = self.regionEdgePoints[ index ]

			for edgePoint in regionEdgePoints.pointsOnEdge:
				currentPixel = edgePoint.xy
				currentPixel = [-int(currentPixel[1]), int(currentPixel[0])]  ## Switching from x,y to row, column
				for adjacentPixelRelativePosition in Bridson_Common.traversalMap:
					adjacentPixel = tuple([currentPixel[0]+adjacentPixelRelativePosition[0], currentPixel[1]+adjacentPixelRelativePosition[1] ])
					# print("genLineAdjacencyMap Searching for pixel:", adjacentPixel)
					if adjacentPixel in self.globalEdgePointMap: # Is the adjacent pixel an EdgePoint?
						adjacentEdgePoints = self.globalEdgePointMap[ adjacentPixel ]

						for adjacentEdgePoint in adjacentEdgePoints:
							# print("genLineAdjacencyMap Found adjacent pixels: ", adjacentEdgePoint.xy, adjacentEdgePoint.regionIndex)
							adjacentIndex = adjacentEdgePoint.regionIndex

							if index != adjacentIndex:
								# print("genLineAdjacencyMap Adding point")
								# Does the region to region map e4xist for current region?
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


		# print("genLineAdjacencyMap.regionAdjacencyRegions keys", self.regionAdjacentRegions.keys())
		# print("genLineAdjacencyMap.regionAdjancencyMap keys", self.regionAdjancencyMap.keys())


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
		# print("findLineEdgePoints edgePixels:", regionEdgePixels.edgePixelList)
		edgePixelList = regionEdgePixels.edgePixelList
		pointsOnEdge = []
		for line in croppedLinePoints:
			startPoint = line[0]
			searchValue = [-int(startPoint[1]), int(startPoint[0])] ## Switching from x,y to row, column
			# print("findLineEdgePoints searching For value:", searchValue)
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
		# print("findLineEdgePoints Points on Edge: ", pointsOnEdge)



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


	def drawOnlyLongestLine(self, index, drawSLICRegions = Bridson_Common.drawSLICRegions):
		# If we are not drawing the SLIC regions, we do not need to flip the Y coordinates.
		# If we draw the SLIC regions, we need to flip the Y coordinates.
		if drawSLICRegions == False:
			flip = 1
		else:
			flip = -1

		regionMap = self.regionMap
		meshObj = self.meshObjCollection[ index ]
		regionIntensity = self.regionIntensityMap[ index ]

		# Need to utilize the region Raster.
		raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
		# Obtain the linePoints from the meshObj.
		linePoints = meshObj.croppedLinePoints.copy()
		# print("A")
		regionCoordinates = regionMap.get( index  )
		topLeftTarget, bottomRightTarget = SLIC.calculateTopLeft( regionCoordinates )

		# print("Length of linePoints:", len(linePoints))
		if len(linePoints) == 0:
			print("Region ", index, "has no lines.")
			return
		currentLine = linePoints[0] * -100  # Set the starting line way off the current raster region.

		# Allow regions to be blank.
		if Bridson_Common.allowBlankRegion == True:
			if regionIntensity > Bridson_Common.cullingBlankThreshold:
				return

		# print("FinishedImage drawing line count:", len(linePoints))
		# for lineIndex in range(len(linePoints)):
		lineIndex = 0
		# colour = Bridson_Common.colourArray[ (lineIndex % len(Bridson_Common.colourArray) ) ]
		colour = '#711fa3'
		line = linePoints[lineIndex].copy()
		line = line * Bridson_Common.mergeScale

		# Add code to high light stable regions.
		if self.regionCoherency[index] >= self.stableThreshold:
			colour = 'r'
		self.tempLines.append( self.ax.plot(line[:, 0], line[:, 1]*flip, color=colour, linewidth=1.5) )
		if Bridson_Common.closestPointPair:  # Only place the dots when we are calculating closest point pair.
			self.ax.plot(currentLine[self.markPoint[0]][0], currentLine[self.markPoint[0]][1] * flip, marker='*',
			             markersize=6, color='g')  # Colour middle point.
			self.ax.plot(line[self.markPoint[1]][0], line[self.markPoint[1]][1]*flip, marker='o', markersize=2, color='r')  # Colour middle point.

		if Bridson_Common.highlightEndpoints:
			regionEdgePoints = self.regionEdgePoints[ index ]
			for edgePoint in regionEdgePoints.pointsOnEdge:
				self.ax.plot(edgePoint.xy[0], edgePoint.xy[1]*flip, marker='x', color='g', markersize=4)


	def removeTempLines(self):
		# The returned data structure requires us to pop and remove the line that was drawn.
		for line in self.tempLines:
			line.pop().remove()

	def calculateLineWidth(self, index):
		# The line width will be a linear calculation: -0.2 * x / 255 + 0.2.  At 0, it should be 0.25 at intensity 0 and 0.05 at intensity 255.
		# Change the line thickness.  Assume minimum line thickness is 0.1.  We want the maximum line thickness to be 1.0.
		return (-1.0 * self.regionIntensityMap[index] / 255) + 1.01

	def drawRegionContourLines(self, index, drawSLICRegions = Bridson_Common.drawSLICRegions):
		# If we are not drawing the SLIC regions, we do not need to flip the Y coordinates.
		# If we draw the SLIC regions, we need to flip the Y coordinates.
		if drawSLICRegions == False:
			flip = 1
		else:
			flip = -1

		regionMap = self.regionMap
		meshObj = self.meshObjCollection[ index ]
		regionIntensity = self.regionIntensityMap[ index ]

		# Need to utilize the region Raster.
		raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
		# Obtain the linePoints from the meshObj.
		linePoints = meshObj.croppedLinePoints.copy()
		# topLeftSource = self.maxLinePoints(linePoints)

		# print("Region Coordinates:", regionCoordinates)
		# print("LinePoints:", linePoints)

		# Grab the shiftx and shifty based on regionMap.
		# print("A")
		regionCoordinates = regionMap.get( index  )
		topLeftTarget, bottomRightTarget = SLIC.calculateTopLeft( regionCoordinates )
		# topLeftSource = self.maxLinePoints( linePoints )
		# shiftCoordinates = (bottomRightTarget[0] - topLeftSource[0], bottomRightTarget[1] - topLeftSource[1])

		# print("Length of linePoints:", len(linePoints))
		if len(linePoints) == 0:
			print("Region ", index, "has no lines.")
			return
		currentLine = linePoints[0] * -100  # Set the starting line way off the current raster region.

		initial=True
		# count = 0

		# Allow regions to be blank.
		if Bridson_Common.allowBlankRegion == True:
			if regionIntensity > Bridson_Common.cullingBlankThreshold:
				return
			# if self.calculateLineSpacing(linePoints[0], linePoints[-1], intensity=regionIntensity) == False:
			# 	return

		lineWidth = self.calculateLineWidth(index)

		# print("FinishedImage drawing line count:", len(linePoints))
		for lineIndex in range(len(linePoints)):
			# if count > 5:
			# 	break
			# if lineIndex % Bridson_Common.lineSkip == 0:
			# colour = Bridson_Common.colourArray[ (lineIndex % len(Bridson_Common.colourArray) ) ]
			colour = '#711fa3'
			line = linePoints[lineIndex].copy()
			line = line * Bridson_Common.mergeScale

			# Add code to high light stable regions.
			if self.regionCoherency[index] >= self.stableThreshold:
				colour = 'r'

			colour = Bridson_Common.colourArray[(lineIndex % len(Bridson_Common.colourArray))]
			# self.ax.plot(line[:, 0], line[:, 1]*flip, color=colour, linewidth=Bridson_Common.lineWidth)
			if Bridson_Common.diagnosticMerge:
				colour = Bridson_Common.colourArray[(lineIndex % len(Bridson_Common.colourArray))]
				lineWidth = 0.1  # HERE
				self.ax.plot(line[:, 0], line[:, 1] * flip, color=colour, linewidth=lineWidth)
			else:
				self.ax.plot(line[:, 0], line[:, 1]*flip, color=colour, linewidth=lineWidth)
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
		intensityDistance = Bridson_Common.determineLineSpacing(intensity)

		# print("Intensity:", intensity, "intensityDistance:", intensityDistance)
		# print("IntensityDistance:", intensityDistance)
		# Get the endPoints of the lines.
		distance = 0
		# print("calculateLineSpacing Line1:", line1[0], line1[-1] )
		# print("calculateLineSpacing Line2:", line2[0], line2[-1])
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
		# print("Distance:", distance, intensityDistance)
		# if distance > Bridson_Common.dradius*factor:
		if distance > intensityDistance:
			# print("Far enough")
			return True
		else:
			# print("Too close")
			return False

jit_module(nopython=True)