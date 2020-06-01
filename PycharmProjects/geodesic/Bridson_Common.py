import math
import Bridson_Common
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np
import numpy.linalg as la # https://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
import Bridson_CreateMask

seedValue = 1

debug = False

def convertAxesBarycentric(x, y, sourceTriang, targetTriang, triFinder, Originalsamples, TargetSamples):
	# Convert the coordinates on one axes to the cartesian axes on the second axes.
	# Convert from triang to triang2.

	tri = triFinder(x, y)

	print("Source Tri: ", tri)
	# print(triang.triangles[tri])
	face = []
	for vertex in sourceTriang.triangles[tri]:  # Create triangle from the coordinates.
		curVertex = Originalsamples[vertex]
		face.append([curVertex[0], curVertex[1]])
	bary1 = calculateBarycentric(face, (x, y))  # Calculate the barycentric coordinates.

	face2 = []
	for vertex in targetTriang.triangles[tri]:
		curVertex = TargetSamples[vertex]
		face2.append([curVertex[0], curVertex[1]])
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
	return tnew.dot(bnew)


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
		print(">>> Error calculating the area: ", a, b, c, s, s*(s-a)*(s-b)*(s-c), v1, v2, v3)
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
	print('Bridson_Common.blurArray: ', np.array(img))
	print('Bridson_Common.blurArray mode: ', img.mode)
	print('Bridson_Common.blurArray shape: ', np.shape(np.array(img)))
	print('Bridson_Common.blurArray Max: ', np.max(np.array(img)))
	print('Bridson_Common.blurArray Min: ', np.min(np.array(img)))
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


def readMask(filename='BlurArrayImage.gif'):
	print("*** Bridson_Common.readMask ***")
	img = Image.open( filename )
	# print('Bridson_Common.readMask: ', np.array(img))
	# print('Bridson_Common.readMask mode: ', img.mode)
	# print('Bridson_Common.readMask shape: ', np.shape(np.array(img)))
	# print('Bridson_Common.readMask Max: ', np.max(np.array(img)))
	# print('Bridson_Common.readMask Min: ', np.min(np.array(img)))

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
	print("ReadMask Max value: ", maxValue)
	if maxValue > 1:
		arr = arr * (255.0 / maxValue)
	else:
		arr = arr * 255.0

	arrayInformation(arr)

	# arrayInformation( arr )
	# Need to invert the resulting array.
	# arr = Bridson_CreateMask.InvertMask( arr )

	print("\n\n*** Bridson_Common.readMask Processed Array ***")
	arrayInformation(arr)
	return arr

def writeMask(array, filename='BlurArrayImage.gif'):
	img = Image.fromarray(array)
	print('Bridson_Common.writeMask')
	print('Bridson_Common.writeMask: ', np.array(img))
	print('Bridson_Common.writeMask mode: ', img.mode)
	print('Bridson_Common.writeMask shape: ', np.shape(np.array(img)))
	print('Bridson_Common.writeMask Max: ', np.max(np.array(img)))
	print('Bridson_Common.writeMask Min: ', np.min(np.array(img)))
	img.save(filename)

def imageInformation(img):

	print('Bridson_Common.imageInformation: ', np.array(img))
	print('Bridson_Common.imageInformation mode: ', img.mode)
	print('Bridson_Common.imageInformation shape: ', np.shape(np.array(img)))
	print('Bridson_Common.imageInformation Max: ', np.max(np.array(img)))
	print('Bridson_Common.imageInformation Min: ', np.min(np.array(img)))


def arrayInformation(arr):
	print('Bridson_Common.arrayInformation: ' , arr)
	print('Bridson_Common.arrayInformation shape: ', np.shape(arr))
	print('Bridson_Common.arrayInformation Max: ', np.max(arr))
	print('Bridson_Common.arrayInformation Min: ', np.min(arr))



if __name__ == "__main__":
	h = findTriangleHeight((0,0), (0,3), (4,0))
	if np.min( h ) < 5:
		print('Min less than threshold: ', np.min(h))
	print(h)