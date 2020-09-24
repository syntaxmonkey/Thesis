
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

np.set_printoptions(threshold=sys.maxsize)  # allow printing without ellipsis: https://stackoverflow.com/questions/44311664/print-numpy-array-without-ellipsis

class FinishedImage:
	def __init__(self, *args, **kwargs):
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(1, 1, 1, aspect=1)
		self.ax.set_title('Merged Image' )
		# self.set
		# self.ax.invert_yaxis()

	def setTitle(self, filename):
		self.ax.set_title('Merged Image - ' + filename + ' - Segments: ' + str(Bridson_Common.segmentCount) + ' - regionPixels: ' + str(Bridson_Common.targetRegionPixelCount) + ' - compactness: ' + str(Bridson_Common.compactnessSLIC) )

	def setXLimit(self, left, right):
		# print("Left:", left, "Right:", right)
		self.ax.set_xlim(left=left, right=right)

	def setYLimit(self, top, bottom):
		# print("Left:", left, "Right:", right)
		self.ax.set_ylim(top=top, bottom=bottom)

	def copyFromOther(self, otherFinishedImage):
		self.unshiftedImageMaskedRegion = otherFinishedImage.unshiftedImageMaskedRegion
		self.shiftedMaskRasterCollection = otherFinishedImage.shiftedMaskRasterCollection
		self.maskRasterCollection = otherFinishedImage.maskRasterCollection
		self.meshObjCollection = otherFinishedImage.meshObjCollection
		self.regionEdgePoints = otherFinishedImage.regionEdgePoints
		self.distanceRasters = otherFinishedImage.distanceRasters
		self.globalEdgePointMap = otherFinishedImage.globalEdgePointMap

		self.regionEdgePointMap =  otherFinishedImage.regionEdgePointMap # Map of (x,y) coordinates
		self.regionAdjancencyMap =  otherFinishedImage.regionAdjancencyMap # Map to contain AdjancencyEdge objects
		self.regionAdjacentRegions = otherFinishedImage.regionAdjacentRegions # Map containing list of adjacent regions for current region.


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


	def cullLines(self, linePoints, regionIntensity):
		'''
			We want to cull the lines based on a distance between the lines.
			When the lines are vertical, we can sort the line order by x coordinates.
			When the lines are not vertical, we have a problem with sorting the

		:param linePoints:
		:param regionIntensity:
		:return:
		'''
		newLinePoints = []
		empty = True
		# Allow regions to be blank.
		if Bridson_Common.allowBlankRegion == True:
			if regionIntensity >= Bridson_Common.cullingBlankThreshold:
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
		self.ax.imshow(mark_boundaries(regionRaster, segments, color=(255,0,0)))
		self.ax.grid()


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




	def shiftLinePoints(self, regionMap, regionRaster):
		# for index in [5]:
		# for index in self.maskRasterCollection.keys():

		for index in self.meshObjCollection.keys():
			raster = self.maskRasterCollection[ index ]

			# Only process the region is it exists.  Can fail if trifinder is not generated.
			meshObj = self.meshObjCollection[ index ]

			# print("Shape of Region Raster:", np.shape(regionRaster))

			regionCoordinates = regionMap.get(index)
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

	def cropCullLines(self, regionMap, regionRaster, maskRasterCollection, meshObjCollection, regionIntensityMap ):
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
			meshObj = meshObjCollection[ index ]

			regionCoordinates = regionMap.get(index)
			topLeftTarget, bottomRightTarget = SLIC.calculateTopLeft(regionCoordinates)

			meshObj.setCroppedLines( self.cropContourLines(meshObj.linePoints, self.shiftedMaskRasterCollection[index], topLeftTarget) )
			# print("CropCullLines region croppedLines:", index, len(meshObj.croppedLinePoints) )

			empty, culledLines = self.cullLines(  meshObj.croppedLinePoints, regionIntensityMap[index])
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
			unshiftedMaskedImage = self.unshiftedImageMaskedRegion[ index ].copy()
			# print("D")
			st = Bridson_StructTensor.ST(unshiftedMaskedImage)
			# print("E")
			direction, coherency = st.calculateEigenVector()
			self.regionDirection[ index ] = direction

			if coherency > Bridson_Common.coherencyThreshold:
				meshAngle = Bridson_Common.determineAngle(direction[0], direction[1]) + 90
			else:
				meshAngle = Bridson_Common.lineAngle + 90

			meshAngle = meshAngle % 360
			self.regionDirection[ index ] = meshAngle
			self.regionCoherency[ index ] = coherency
		print("Region Coherency", self.regionCoherency)



	def generateRAG(self, filename, segments, regionColourMap):
		# Generate RAG (Region Adjacency Graph)
		self.regionColourMap = regionColourMap
		originalImage = cv.imread(filename)
		rag = graph.rag_mean_color(originalImage, segments)  # Generate the Region AdjacencyGraph.
		self.rag = rag
		print("RAG Nodes:", rag.nodes)
		print("RAG Edges:", rag.edges)

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
		print("Region Differences:", self.regionDifferences)
		print("regionToRegions:", self.regionToRegions)
		# Calculate the difference threshold.  Regions with differences below this threshold are
		# candidates for changing their direction.
		print( "Region Differences Values:", list( self.regionDifferences.values() ))
		self.diffThreshold = np.percentile( list( self.regionDifferences.values() ), Bridson_Common.differencePercentile)
		print("diff threshold", self.diffThreshold)


	def adjustRegionAngles(self, iterations=1):
		# Iterate through all regions.
		# Check the coherency for stableRegionThreshold
		# Check the differences between the regions.
		# If the difference value is below the diffThreshold, average the angles.
		print("Region Directions:", self.regionDirection)
		for i in range(iterations):
			for index in self.regionToRegions.keys():
				# print("Current region:", index)
				# print("region ", index, "coherency", self.regionCoherency[index] )
				if index not in self.regionCoherency.keys():
					print("Region", index, "not in coherency map")
				elif self.regionCoherency[ index ] < Bridson_Common.stableRegionThreshold:
					# print("Not Stable region:", index)
					# If the region coherency is below the threshold, continue with checking
					adjacentRegions = self.regionToRegions[ index ]
					# print("Adjacency regions", adjacentRegions)
					for adjacentIndex in adjacentRegions:
						startIndex, endIndex = index, adjacentIndex
						if startIndex > endIndex:
							startIndex, endIndex = adjacentIndex, index

						pairIndex = (startIndex, endIndex)
						# print("Pair Index", pairIndex)
						if self.regionDifferences[ pairIndex ] < self.diffThreshold:
							# Adjust this region's angle.
							# print("Index", index, "Previous Angle", self.regionDirection[ index ] )
							self.regionDirection[ index ] = Bridson_Angles.calcAverageAngle( self.regionDirection[ index ], self.regionDirection[ adjacentIndex ] )
							# print("Adjusting angle of region", index, "now has direction",self.regionDirection[index])
				# else:
				# 	print("Region is stable", index)
		print("POST Region Directions:", self.regionDirection)



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


		print("genLineAdjacencyMap.regionAdjacencyRegions keys", self.regionAdjacentRegions.keys())
		print("genLineAdjacencyMap.regionAdjancencyMap keys", self.regionAdjancencyMap.keys())


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
			if regionIntensity > 250:
				return
			# if self.calculateLineSpacing(linePoints[0], linePoints[-1], intensity=regionIntensity) == False:
			# 	return

		# print("FinishedImage drawing line count:", len(linePoints))
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
				self.ax.plot(line[:, 0], line[:, 1]*flip, color=colour, linewidth=Bridson_Common.lineWidth)
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