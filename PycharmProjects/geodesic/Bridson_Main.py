from Bridson_Delaunay import generateDelaunay, displayDelaunayMesh
from Bridson_sampling import Bridson_sampling, displayPoints, genSquarePerimeterPoints, calculateParameters
import numpy as np
import math
import matplotlib.pyplot as plt
import Bridson_createOBJFile
import os
import Bridson_CreateMask
import Bridson_ChainCode
from PIL import Image, ImageFont, ImageDraw, ImageFilter

def generatePointsDisplay(xrange, yrange, dradius):
	# Generate points and display
	points = Bridson_sampling(width=xrange, height=yrange, radius=dradius)
	displayPoints(points, xrange, yrange)


def generateDelaunayDisplay(xrange, yrange, dradius):
	points = Bridson_sampling(width=xrange, height=yrange, radius=dradius)
	displayDelaunayMesh(points, dradius)

def genSquareDelaunayDisplay(xrange, yrange, radius=0, pointCount=0, mask=[], border=[]):
	radius, pointCount = calculateParameters(xrange, yrange, radius, pointCount)

	points = genSquarePerimeterPoints(xrange, yrange, radius=radius, pointCount=pointCount)
	print(np.shape(points))

	# Merge border with square perimeter.
	points = np.append( points, border, axis=0)

	# Generate all the sample points.
	points = Bridson_sampling(width=xrange, height=yrange, radius=radius, existingPoints=points, mask=mask)
	print(np.shape(points))

	if len(mask) > 0:
		points = filterOutPoints(points, mask)

	tri = displayDelaunayMesh(points, radius, mask, xrange)
	return points, tri


def filterOutPoints(points, mask):
	# expects inverted mask.
	# Remove points that intersect with the mesh.
	newPoints = []
	for point in points:
		# Points that are '0' should be retained.
		if mask[int(point[0]), int(point[1])] == 0:
			newPoints.append(point)

	return np.array(newPoints)

def euclidean_distance(a, b):
	dx = a[0] - b[0]
	dy = a[1] - b[1]
	return math.sqrt(dx * dx + dy * dy)


def createMeshFile(samples, tri, radius, center ):
	# Produce the mesh file.  Flatten the mesh with BFF.  Extract the 2D from BFF flattened mesh.
	path = "../../boundary-first-flattening/build/"
	# Create object file for image.
	Bridson_createOBJFile.createObjFile2D(path, "test1.obj", samples, tri, radius, center, distance=euclidean_distance)


def BFFReshape():
	print("Reshaping with BFF")
	path = "../../boundary-first-flattening/build/"
	# os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=1 --normalizeUVs ")
	os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --flattenToDisk --normalizeUVs ")
	# os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=1 --normalizeUVs --nCones=" + str(perimeterSegments))
	# os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=1 --normalizeUVs --nCones=6")

def FlattenMesh():
	path = "../../boundary-first-flattening/build/"
	print("Extracting 2D image post BFF Reshaping")
	os.system(path + "extract.py test1_out.obj test1_out_flat.obj")


def cleanUpFiles():
	path = "../../boundary-first-flattening/build/"
	os.system("rm " + path + "test1.obj ")
	os.system("rm "+ path + "test1_out.obj")
	os.system("rm " + path + "test1_out_flat.obj")



if __name__ == '__main__':
	cleanUpFiles()
	dradius = 1.5
	xrange, yrange = 100, 100
	mask = Bridson_CreateMask.genLetter(xrange, yrange, character='Y')
	count, chain, chainDirection, border = Bridson_ChainCode.generateChainCode(mask, rotate=False)
	border = Bridson_ChainCode.generateBorder(border, dradius)

	invertedMask =  Bridson_CreateMask.InvertMask( mask )

	# print(invertedMask)
	plt.figure()
	plt.subplot(1, 1, 1, aspect=1)
	plt.title('Inverted Mask')
	plt.imshow(invertedMask)
	plt.plot([i[1] for i in border], [i[0] for i in border], 'og') # Plot the "boundaries" points as green dots.

	# generatePointsDisplay(xrange, yrange, dradius)
	# generateDelaunayDisplay(xrange, yrange, dradius)
	points, tri = genSquareDelaunayDisplay(xrange, yrange, radius=dradius, mask=invertedMask, border=border)





	fakeRadius = max(xrange,yrange)

	createMeshFile(points, tri, fakeRadius, (xrange/2.0, yrange/2.0))
	BFFReshape()
	FlattenMesh()
	plt.show()