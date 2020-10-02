import math
import Bridson_Common
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np
import numpy.linalg as la # https://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
import Bridson_CreateMask
import matplotlib.pyplot as plt
import inspect
import os
from scipy.ndimage.morphology import distance_transform_cdt
import pandas as pd
from scipy.spatial import distance
import sys
from skimage import io



# Increase width of console printing: https://stackoverflow.com/questions/25628496/getting-wider-output-in-pycharms-built-in-console
desired_width = 320
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=desired_width)

seedValue = 11

SLIC0=True
compactnessSLIC=1
timeoutPeriod = 5
SLICIterations=50
SLICGrey = False

#####################################
bulkGeneration = True
smallBatch=False

coreCount = 4
if os.path.exists("./output") == True:
	if os.path.isdir("./output") == False:
		exit(-1)
else:
	os.mkdir("./output")

if Bridson_Common.bulkGeneration:
	sys.stdout = open("./output/detailLogs.txt", "a")
	pass

#####################################

debug=False

if bulkGeneration == True:
	displayMesh = False
else:
	displayMesh = True

generatePREImage=False

displayMesh = False

diagnostic=False # Generally avoid dispalying meshes.  Only count the number of successful trifinder generations.
highlightEdgeTriangle=False # Highlight the edge triangle that contains the exterior point of the vertical lines.
drawSLICRegions = False

EqualizeHistogram=False
Median=True

sortExteriorPoints=True
lineSkip = 1
lineCullingDistanceFactor = 2
allowBlankRegion=False # Allow the region to be blank.  Otherwise, regions can have one single line even if the intensity is high.
cullingBlankThreshold=230 # If the region has intensity greater than this value, then make the region blank.
highlightEndpoints=False
lineCullAlgorithm='segmented'  # Valid values: 'log', 'exp', 'none', 'segmented'.

closestPointPair=False
middleAverageOnly=False
if middleAverageOnly == True:
	divisor = 1.0
else:
	divisor = 3.0

print("Divisor:", Bridson_Common.divisor)

normalizeUV = True
invert = False

colourCodeMesh = True
colourCount = 20

linesOnFlat = True
verticalLines = True
lineAngle = 90
coherencyThreshold = 0.1
lineWidth = 0.25
stableCoherencyPercentile = 95 # Regions percentile with a coherency above this value are considered stable.
diffAttractPercentile = 60 # Regions with differences below this percentile will attract.
diffRepelPercentile =85 # Regions with differences above this percentile will repel.
attractionBin = 3
repelBin = 3
stableBin = -4
stableAttractSet=False  # If true, during the attract, stable regions will simply set adjacent regions equal to the desired angle.  Otherwise, it will take the average.
binSize=10
angleAdjustIterations=1000
attractFudge=1.05 # Fudge factor to use when comparing region intensities.

semanticSegmentation='none' # Valid values: 'deeplabv3', 'mask_rcnn', 'both', 'none'
semanticSegmentationRatio=0.5 # This is the weighting of the semantic segmentation.
semanticInvertMaskrcnn=True
intensityMapUserMergeCNN=True # When generating the regionIntensityMap, utilize the CNN merged image when set to True.  Otherwise, use the original image.


barycentricVertexCorrection = True

# These are the rotation values.
if linesOnFlat:
	barycentricCorrectionValue = 2
else:
	barycentricCorrectionValue = 1

drawDots = False

density = 0.01
lineDotDensity = 0.01
lineRadiusFactor = 1

targetPercent = 0.15
targetRegionPixelCount = 800
segmentCount = 80
dradius = 1.5 # Important that dradius is greater than 1.0.  When the value is 1.0 or lower, BFF seems to have lots of issues with the mesh.
radiusDefault = 1.5
radiusDivisor = 30 # the number of radii for each region.
regionDynamicRadius = True
mergeDistance = 6 # radius to

increaseContrast=False
contrastFactor=1.5 # Values above 1 increase contrast.  Values below 1 reduce contrast.


# Image Generation
GreyscaleSLIC = True  # Generate the greyscale image of the regions.


colourArray = ['r', 'b', 'm']
colourArray = ['b', 'b', 'b']

mergeScale = 1  # How much to scale the contour lines before merging.
cropContours = True

Bridson_Common.test1obj = ""
Bridson_Common.test1_outobj = ""
Bridson_Common.test1_out_flatobj = ""
Bridson_Common.chaincodefile = ""
objPath = "../../boundary-first-flattening/build/"


# traversalMap = [ [-1,1], [0,1], [1,1],
#                  [-1, 0],  [1, 0],
#                  [-1, -1], [0, -1], [1, -1] ]


def outputEnvironmentVariables():
	print("------- Environment Variables ----------------------------------- ")
	print("segmentCount:", Bridson_Common.segmentCount)
	print("lineCullAlgorithm: ", Bridson_Common.lineCullAlgorithm)
	print("lineAngle:", Bridson_Common.lineAngle)
	print("coherencyThreshold:", Bridson_Common.coherencyThreshold)
	print("highlightEndpoints:", Bridson_Common.highlightEndpoints)
	print("middleAverageOnly:", Bridson_Common.middleAverageOnly)
	print("compactnessSLIC:", Bridson_Common.compactnessSLIC)
	print("SLICIterations:", Bridson_Common.SLICIterations)
	print("SLIC0:", Bridson_Common.SLIC0)
	print("allowBlankRegion:", Bridson_Common.allowBlankRegion)
	print("cullingBlankThreshold:", Bridson_Common.cullingBlankThreshold)
	print("nearbyDistance:", mergeDistance)
	print("semanticSegmentation:", semanticSegmentation)
	print("semanticSegmentationRatio:", semanticSegmentationRatio)
	print("semanticInvertMaskrcnn:", semanticInvertMaskrcnn)
	print("stableCoherencyPercentile:", stableCoherencyPercentile)
	print("differencePercentile:", diffAttractPercentile)
	print("diffRepelPercentile:", diffRepelPercentile)
	print("generatePREImage:", generatePREImage)
	print("attractionBin:", attractionBin)
	print("repelBin:", repelBin)
	print("stableBin:", stableBin)
	print("stableAttractSet:", stableAttractSet)
	print("binSize:", binSize)
	print("attractFudge:", attractFudge)

	print("------------------------------------------------------------------")

def determineLineSpacing( intensity):
	if Bridson_Common.lineCullAlgorithm == 'log':
		'''
		Logrithmic scale.			
		Intensity: 255 produces 51.57303927154884
		Intensity: 200 produces 49.36711720583027
		Intensity: 128 produces 45.322383916284224
		Intensity: 100 produces 43.09074884943549
		Intensity: 50 produces 36.85897369805666
		Intensity: 30 produces 32.318595570519726
		Intensity: 10 produces 22.86924638832273
		Intensity: 1 produces 7.321629908943605
		Intensity: 0 produces 1.0
		'''
		intensityDistance = math.log10(intensity+1)*21 + 1
	elif Bridson_Common.lineCullAlgorithm == 'exp':
		'''
		Exponential scale.  Lighter regions have significantly fewer lines.
		Intensity: 255 produces 164.0219072999017
		Intensity: 200 produces 54.598150033144236
		Intensity: 128 produces 12.935817315543076
		Intensity: 100 produces 7.38905609893065
		Intensity: 50 produces 2.718281828459045
		Intensity: 30 produces 1.8221188003905089
		Intensity: 10 produces 1.2214027581601699
		Intensity: 1 produces 1.0202013400267558
		Intensity: 0 produces 1.0
		'''
		intensityDistance = math.exp(intensity/60)
	elif Bridson_Common.lineCullAlgorithm == 'segmented':
		if intensity < 128:
			intensityDistance = intensity / 10
		elif intensity < 200:
			intensityDistance = intensity / 5
		else:
			intensityDistance = intensity
	else:
		intensityDistance = intensity

		intensityDistance += 1
	return intensityDistance




def readImagefile(filename):
	image = io.imread(filename)
	imageArr = np.asarray( Image.fromarray(image).convert('L') )
	return imageArr

def determineRadius(width=1, height=1):
	if Bridson_Common.regionDynamicRadius:
		span = width if width < height else height
		Bridson_Common.dradius = span / Bridson_Common.radiusDivisor
		Bridson_Common.dradius = Bridson_Common.dradius if Bridson_Common.dradius > Bridson_Common.radiusDefault else Bridson_Common.radiusDefault
	else:
		Bridson_Common.dradius = Bridson_Common.radiusDefault
	print("Bridson_Common.dradius:", Bridson_Common.dradius)

def calculateDirection(angle):
	angle = angle % 360
	# print("calculateDirection starting angle:", angle)
	# based on the angle, return the delta x and delta y.
	if angle == 0:
		dx = 0
		dy = 1
	elif angle == 180:
		dx = 0
		dy = -1
	elif angle == 90:
		dx = 1
		dy = 0
	elif angle == 270:
		dx = -1
		dy = 0
	elif angle > 0 and angle < 90:
		dx = math.sin(angle * math.pi / 180)
		dy = math.cos(angle * math.pi / 180)
	elif angle > 90 and angle < 180:
		tempAngle = angle - 90
		dx = math.cos(tempAngle * math.pi / 180)
		dy = -math.sin(tempAngle * math.pi / 180)
	elif angle > 180 and angle < 270:
		tempAngle = angle - 180
		dx = -math.sin(tempAngle * math.pi / 180)
		dy = -math.cos(tempAngle * math.pi / 180)
	elif angle > 270 and angle < 360:
		tempAngle = angle - 270
		dx = -math.cos(tempAngle * math.pi / 180)
		dy = math.sin(tempAngle * math.pi / 180)
	# print("calculateDirection dx, dy:", dx, dy)
	return dx, dy

def determineAngle(dx, dy):
	# https://stackoverflow.com/questions/36727257/calculating-rotation-degrees-based-on-delta-x-y-movement-of-touch-or-mouse
	# print("calculateAngle Dx Dy:", dx, dy)
	# //direction from old to new location in radians, easy to convert to degrees
	dir = ( math.atan2(dx, dy) * 180 / math.pi ) % 360;
	# print("calculateAngle Recovered angle:", dir)
	return dir


def generateTraversalMap(radius):
	traversalMap = []
	for i in range(-radius, radius+1):
		for j in range(-radius, radius+1):
			traversalMap.append( [i, j] )

	traversalMap.pop( traversalMap.index([0,0]) )
	return traversalMap

traversalMap = generateTraversalMap(Bridson_Common.mergeDistance)

def findClosestIndex(s1, s2):
	# print("Bridson_Common:findClosestIndex s1:", s1)
	# print("Bridson_Common:findClosestIndex s2:", s2)
	# both s1 and s2 should be 2D.

	distances = distance.cdist(s1, s2)
	# shortestDistances = distance.cdist(s1, s2).min(axis=1)
	location = np.where(distances == distances.min())
	# print('Location:', location, distances.min())
	# Return the an tuple.  Tuple[0] contains indeces in s1.  Tuple[1] contains indeces in s2.
	return location


def displayDistanceMask(mask, indexLabel, topLeftTarget, bottomRightTarget):
	#################################
	# distanceMask = Bridson_CreateMask.InvertMask( mask5x )
	if Bridson_Common.displayMesh == True:
		distanceMask = mask.copy()
		print("DistanceMask Max", np.max(distanceMask))
		# distanceMask = Bridson_CreateMask.InvertMask( distanceMask )

		# Generate the distance raster based on the mask.
		# Should we instead generate the distance raster based on the original mask?
		distanceRaster = Bridson_Common.distance_from_edge(distanceMask)
		# distanceRaster = np.ma.masked_array(distanceRaster, np.logical_and(distanceRaster < 2, distanceRaster > 0))  # Filters out the pixels with distance of 1.
		# Want to retain the pixels that have a distance of 1.  Set the other pixels to zero.

		# Try to retain the zeros and ones.
		# distanceRaster = np.ma.masked_array(distanceRaster,  distanceRaster > 1) ## Yay, this seems to retain the 1s and zeros.


		# Find the pixels that contain the value 1: https://stackoverflow.com/questions/44296310/get-indices-of-elements-that-are-greater-than-a-threshold-in-2d-numpy-array
		distance1pixelIndices = np.argwhere( (distanceRaster > 0) & (distanceRaster <= 2) )
		print("Distance 1 Indices count:", len(distance1pixelIndices)  )

		print("-------------------------------")
		for index in distance1pixelIndices:
			print(index, " has value: ", distanceRaster[index[0], index[1] ])
		print("-------------------------------")

		# Count the number of 1's in the raster: https://www.kite.com/python/answers/how-to-count-the-occurrences-of-a-value-in-a-numpy-array-in-python
		print("Distance 1 Count: ", np.count_nonzero( distanceRaster <= 2) )

		plt.figure()
		ax = plt.subplot(1, 1, 1, aspect=1, label='Region Raster ' + str(indexLabel))
		plt.title('Distance Raster Debug ' + str(indexLabel))
		''' Draw Letter blob '''

		# blankRaster = np.zeros(np.shape(imageraster))
		# ax3 = plt.subplot2grid(gridsize, (0, 1), rowspan=1)
		# ax3.imshow(blankRaster)
		# distanceRaster[5][5]=255 # Reference point

		print("DistanceRaster Display:\n",  distanceRaster[topLeftTarget[0]:bottomRightTarget[0]+1, topLeftTarget[1]:bottomRightTarget[1]+1 ] )
		ax.imshow(distanceRaster)
		# plt.plot(5, 5, color='r', markersize=10)
		ax.grid()

def distance_from_edge(x):
    x = np.pad(x, 1, mode='constant')
    dist = distance_transform_cdt(x, metric='chessboard')
    return dist[1:-1, 1:-1]


def saveImage(filename, postFix, fig):
	if os.path.exists("./output") == True:
		if os.path.isdir("./output") == False:
			exit(-1)
	else:
		os.mkdir("./output")

	try:
		# Save the figures to files: https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
		actualFileName = "./output/" + filename + "_segments_"+ str(Bridson_Common.segmentCount)
		if Bridson_Common.SLIC0:
			actualFileName = actualFileName + "_compactness_SLIC0"
		else:
			actualFileName = actualFileName + "_compactness_" + str(Bridson_Common.compactnessSLIC)

		actualFileName = actualFileName + "_cnn_" + Bridson_Common.semanticSegmentation + "_semanticRatio_" + str(
			Bridson_Common.semanticSegmentationRatio) + "_" + postFix + ".png"
		fig.savefig( actualFileName )
		if Bridson_Common.bulkGeneration: # Delete the figures when we are bulk generating.
			plt.close(fig=fig)
	except Exception as e:
		print("Error saving file:", e)

def rotateClockwise90(array, angle=90):
	'''
		Swap the X and Y values.  Multiply the Y values with -1.
	'''
	# print("Pre Array:", array)
	newArray = Bridson_Common.swapXY(array) # Swap the X Y values
	newArray[:, 1] *= -1  # Multiple the Y values with -1
	# print("Post Array:", newArray)
	return newArray


def swapXY(array):
	newArray = np.copy(array)
	newArray[:,0] = array[:,1]
	newArray[:,1] = array[:,0]
	return newArray

def logDebug(moduleName, *argv):
	if Bridson_Common.debug:
		callingFrame = inspect.stack()[1] # https://docs.python.org/3/library/inspect.html#inspect.getmembers -
		callingFunction = callingFrame[3]
		print(moduleName , "-->", callingFunction , ": ", end='')
		for arg in argv:
			print(arg, " ", end='')
		print()

def convertAxesBarycentric(x, y, sourceTriang, targetTriang, sourcetriFinder, Originalsamples, TargetSamples):
	# Convert the coordinates on one axes to the cartesian axes on the second axes.
	# Convert from triang to triang2.
	cartesian = None
	try:
		tri = sourcetriFinder(x, y)

		# Bridson_Common.logDebug(__name__, "Source Tri: ", tri)
		# Bridson_Common.logDebug(__name__, triang.triangles[tri])
		face = []
		for vertex in sourceTriang.triangles[tri]:  # Create triangle from the coordinates.
			curVertex = Originalsamples[vertex]
			face.append([curVertex[0], curVertex[1]])
		bary1 = calculateBarycentric(face, (x, y))  # Calculate the barycentric coordinates.

		face2 = []
		vertices = list(targetTriang.triangles[tri])
		# print("Original Vertices:", vertices)
		if Bridson_Common.barycentricVertexCorrection:
			vertices = vertices[Bridson_Common.barycentricCorrectionValue:] + vertices[:barycentricCorrectionValue] # HSC - rotate
		# print("New Vertices:", vertices)
		# for vertex in targetTriang.triangles[tri]:
		for vertex in vertices:
			curVertex = TargetSamples[vertex]
			face2.append([curVertex[0], curVertex[1]])
		# Bridson_Common.logDebug(__name__, "Target Tri: ", face2)
		cartesian = get_cartesian_from_barycentric(bary1, face2)
	except Exception as e:
		print("Cause of error:", e)
		print("Execution Info:", sys.exc_info()[0])
	return cartesian


def calculateBarycentric(vertices, point):
	# Calculating barycentric coodinates: https://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
	T = (np.array(vertices[:-1]) - vertices[-1]).T
	v = np.dot(la.inv(T), np.array(point) - vertices[-1])
	v.resize(len(vertices))
	v[-1] = 1 - v.sum()
	return v


def get_cartesian_from_barycentric(b, t):
	# https://stackoverflow.com/questions/56328254/how-to-make-the-conversion-from-barycentric-coordinates-to-cartesian-coordinates
	''' The expected data format
		b = np.array([0.25,0.3,0.45]) # Barycentric coordinates
		t = np.transpose(np.array([[0,0],[1,0],[0,1]])) # Triangle
	'''
	tnew = np.transpose(np.array(t))
	bnew = np.array(b)
	# print("Barycentric:", tnew.dot(bnew))
	bary = tnew.dot(bnew)
	return [bary[0], bary[1]]


def euclidean_distance(a, b):
	dx = a[0] - b[0]
	dy = a[1] - b[1]
	return math.sqrt(dx * dx + dy * dy)

def findArea(v1, v2, v3):
	# https://www.javatpoint.com/python-area-of-triangle
	a = euclidean_distance(v1, v2)
	b = euclidean_distance(v2, v3)
	c = euclidean_distance(v3, v1)
	s = (a+b+c)/2
	try:
		core = s * (s - a) * (s - b) * (s - c)
		if core > 0:
			area = math.sqrt(s*(s-a)*(s-b)*(s-c))
		else:
			area = 0
	except:
		Bridson_Common.logDebug(__name__, ">>> Error calculating the area: ", a, b, c, s, s*(s-a)*(s-b)*(s-c), v1, v2, v3)
	if area == 0:
		Bridson_Common.logDebug(__name__, ">>> Area is ZERO: ", a, b, c, s,
		                        s * (s - a) * (s - b) * (s - c), v1, v2, v3)
	return area


def findAverageArea(triangles, samples):
	count = 0
	area = 0
	# for facet in triangleValues.simplices.copy():
	for facet in triangles:
		area += Bridson_Common.findArea(samples[facet[0]], samples[facet[1]], samples[facet[2]])
		count+=1
	averageArea = area / count
	return averageArea


def blurArray(array, blur):
	img = Image.fromarray(array)
	# img.save('BlurArrayImage.gif')
	# Use dilation to increase the size of the mask: https://stackoverflow.com/questions/44195007/equivalents-to-opencvs-erode-and-dilate-in-pil
	img = img.filter(ImageFilter.MinFilter(blur))  # Erosion
	# img = img.filter(ImageFilter.MaxFilter(blur))  # Dilation
	# img = img.filter(ImageFilter.GaussianBlur(0))
	# img = img.filter(ImageFilter.BoxBlur(blur))

	# Bridson_Common.logDebug(__name__, 'Bridson_Common.blurArray: ', np.array(img))
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.blurArray mode: ', img.mode)
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.blurArray shape: ', np.shape(np.array(img)))
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.blurArray Max: ', np.max(np.array(img)))
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.blurArray Min: ', np.min(np.array(img)))
	#
	return np.array(img)


def findTriangleHeight(p1, p2, p3):
	a = euclidean_distance(p1, p2)
	b = euclidean_distance(p2, p3)
	c = euclidean_distance(p3, p1)

	s = (a + b + c) / 2
	core = s*(s-a)*(s-b)*(s-c)
	if core < 0:
		core = 0

	h1 = 2*math.sqrt( core ) / a
	h2 = 2*math.sqrt( core ) / b
	h3 = 2*math.sqrt( core ) / c

	h = np.array([h1 / a, h2 / b, h3 / c])  # Find the ratio of the heigh relative to the base.
	return h


def findMinMaxTriangleHeight(p1, p2, p3):
	a = euclidean_distance(p1, p2)
	b = euclidean_distance(p2, p3)
	c = euclidean_distance(p3, p1)

	s = (a + b + c) / 2
	core = s*(s-a)*(s-b)*(s-c)
	if core < 0:
		core = 0

	h1 = 2*math.sqrt( core ) / a
	h2 = 2*math.sqrt( core ) / b
	h3 = 2*math.sqrt( core ) / c

	hmax = np.sort([h1 , h2 , h3 ])[-1]  # Find the largest height in the triangle.
	hmin = np.sort([h1 , h2 , h3 ])[0]
	return hmin, hmax

def readMask(filename='BlurArrayImage.gif'):
	Bridson_Common.logDebug(__name__, "*** Bridson_Common.readMask ***")
	img = Image.open( filename )
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.readMask: ', np.array(img))
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.readMask mode: ', img.mode)
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.readMask shape: ', np.shape(np.array(img)))
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.readMask Max: ', np.max(np.array(img)))
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.readMask Min: ', np.min(np.array(img)))

	arr = np.array( img )
	arr = arr - np.min(arr)
	arrayInformation(arr)

	minValue = np.min( arr )
	if minValue < 0:
		arr = arr + abs( minValue )

	maxValue = np.max(arr)
	if maxValue > 255.0:
		arr = arr * ( 128.0 / maxValue)

	maxValue = np.max(arr)
	Bridson_Common.logDebug(__name__, "ReadMask Max value: ", maxValue)
	if maxValue > 1:
		arr = arr * (255.0 / maxValue)
	else:
		arr = arr * 255.0

	arrayInformation(arr)

	# arrayInformation( arr )
	# Need to invert the resulting array.
	# arr = Bridson_CreateMask.InvertMask( arr )

	Bridson_Common.logDebug(__name__, "\n\n*** Bridson_Common.readMask Processed Array ***")
	arrayInformation(arr)
	return arr

def writeMask(array, filename='BlurArrayImage.gif'):
	img = Image.fromarray(array)

	# Bridson_Common.logDebug(__name__, 'Bridson_Common.writeMask')
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.writeMask: ', np.array(img))
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.writeMask mode: ', img.mode)
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.writeMask shape: ', np.shape(np.array(img)))
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.writeMask Max: ', np.max(np.array(img)))
	# Bridson_Common.logDebug(__name__, 'Bridson_Common.writeMask Min: ', np.min(np.array(img)))
	#
	img.save(filename)

def imageInformation(img):

	Bridson_Common.logDebug(__name__, 'Bridson_Common.imageInformation: ', np.array(img))
	Bridson_Common.logDebug(__name__, 'Bridson_Common.imageInformation mode: ', img.mode)
	Bridson_Common.logDebug(__name__, 'Bridson_Common.imageInformation shape: ', np.shape(np.array(img)))
	Bridson_Common.logDebug(__name__, 'Bridson_Common.imageInformation Max: ', np.max(np.array(img)))
	Bridson_Common.logDebug(__name__, 'Bridson_Common.imageInformation Min: ', np.min(np.array(img)))


def arrayInformation(arr):
	Bridson_Common.logDebug(__name__, 'Bridson_Common.arrayInformation: ' , arr)
	Bridson_Common.logDebug(__name__, 'Bridson_Common.arrayInformation shape: ', np.shape(arr))
	Bridson_Common.logDebug(__name__, 'Bridson_Common.arrayInformation Max: ', np.max(arr))
	Bridson_Common.logDebug(__name__, 'Bridson_Common.arrayInformation Min: ', np.min(arr))

def triangleHistogram(vertices, faces, indexLabel):
	areaValues = []
	for face in faces:
		faceArea = Bridson_Common.findArea(vertices[face[0]], vertices[face[1]], vertices[face[2]])
		if faceArea == 0:
			Bridson_Common.logDebug(__name__, 'Face', face, 'with vertices', vertices[face[0]], vertices[face[1]], vertices[face[2]], 'has an area of', faceArea)
		areaValues.append( Bridson_Common.findArea(vertices[face[0]], vertices[face[1]], vertices[face[2]]) )


	# hist, bins, _ = plt.hist(areaValues, bins=8)

	# histogram on log scale.
	# Use non-equal bin sizes, such that they look equal on log scale.
	# logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

	Bridson_Common.logDebug(__name__, 'Area Values - ' + " Min: " + str(min(areaValues))  +  " Max: "+ str(max(areaValues)))
	if min(areaValues) == 0:
		Bridson_Common.logDebug(__name__, 'Minimum area is ZERO.')
	else:
		Bridson_Common.logDebug(__name__, 'Area Ratio: ' + "{:.4e}".format(max(areaValues) / min(areaValues)))

	if min(areaValues) < 1.0e-15:
		Bridson_Common.logDebug(__name__, " ****************** Min Area less than 1e-15: GUESS that trifinder will NOT be valid. ********************")
	else:
		Bridson_Common.logDebug(__name__, " ****************** Min Area greater than 1e-15: Guess that trifinder will be valid *****************")

	if Bridson_Common.debug:
		plt.figure()
		n, bins, patches = plt.hist(x=areaValues, alpha=0.7, rwidth=0.5, bins=1000)
		# plt.hist(x=areaValues,  bins=logbins)
		# plt.grid(axis='y', alpha=0.75)
		plt.xlabel('Area Values - ' + " Min: " + str(min(areaValues))  +  " Max: "+ str(max(areaValues)))
		plt.ylabel('Frequency')
		plt.title('Flattened Area Histogram ' + str(indexLabel) + 'Triangle Count: ' + str(len(areaValues)))
		# plt.text(23, 45, r'$\mu=15, b=3$')
		plt.xscale('log')
		# maxfreq = n.max()
		# Set a clean upper y-axis limit.
		# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


# https://rosettacode.org/wiki/Find_the_intersection_of_two_lines#Python ** Works.
def line_intersect(segment1, segment2):
	"""
	returns a (x, y) tuple or None if there is no intersection
	Expecting the following input:
	segment1: [(Ax1, Ay1),( Ax2, Ay2)]
	segment2: [(Bx1, By1), (Bx2, By2)]
	"""
	(Ax1, Ay1), (Ax2, Ay2) = segment1
	(Bx1, By1), (Bx2, By2) = segment2
	d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
	if d:
		uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
		uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
	else:
		return
	if not (0 <= uA <= 1 and 0 <= uB <= 1):
		return
	x = Ax1 + uA * (Ax2 - Ax1)
	y = Ay1 + uA * (Ay2 - Ay1)

	return x, y




if __name__ == "__main__":
	h = findTriangleHeight((0,0), (0,3), (4,0))
	if np.min( h ) < 5:
		Bridson_Common.logDebug(__name__, 'Min less than threshold: ', np.min(h))
	Bridson_Common.logDebug(__name__, h)

	segment_one = ((38.5, 59.605199292382274), (38.5, 57.42400566413322))
	segment_two = ((38.5, 62.5), (38.99359 , 60.497322))

	(a,b), (c,d) = segment_one
	(e,f), (g,h) = segment_two
	print ("Intersection:", line_intersect(segment_one, segment_two))

