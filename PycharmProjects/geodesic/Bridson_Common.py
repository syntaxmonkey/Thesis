import math
import Bridson_Common
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np
import numpy.linalg as la # https://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
import Bridson_CreateMask
import matplotlib.pyplot as plt
import inspect
import os

seedValue = 11

SLIC0=False
compactnessSLIC=1

bulkGeneration = True
debug=False
if bulkGeneration == True:
	displayMesh = False
else:
	displayMesh = True
diagnostic=False # Generally avoid dispalying meshes.  Only count the number of successful trifinder generations.
highlightEdgeTriangle=False # Highlight the edge triangle that contains the exterior point of the vertical lines.
drawSLICRegions = False

sortExteriorPoints=True
lineSkip = 1
lineCullingDistanceFactor = 2
allowBlankRegion=True

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
lineAngle = 125


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

colourArray = ['r', 'b', 'm']
colourArray = ['b', 'b', 'b']

mergeScale = 1  # How much to scale the contour lines before merging.
cropContours = True


def saveImage(filename, postFix, fig):
	if os.path.exists("./output") == True:
		if os.path.isdir("./output") == False:
			exit(-1)
	else:
		os.mkdir("./output")


	# Save the figures to files: https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
	actualFileName = "./output/" + filename + "_segments_" + str(Bridson_Common.segmentCount) + "_regionPixels_" + str(Bridson_Common.targetRegionPixelCount) + "_" + postFix + ".png"
	fig.savefig( actualFileName )
	if Bridson_Common.bulkGeneration: # Delete the figures when we are bulk generating.
		plt.close(fig=fig)

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

