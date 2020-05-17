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
import Bridson_MeshObj
import readOBJFile
import Bridson_readOBJFile
import SLIC
import pylab
import matplotlib
matplotlib.use("tkagg")


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


def SLICImage():
	startIndex = 0 # Index starts at 0.
	regionIndex = startIndex
	imageraster, regionMap = SLIC.callSLIC(segmentCount=40)
	stopIndex=startIndex+16

	plt.figure()
	ax3 = plt.subplot(2, 1, 1, aspect=1, label='Image regions')
	plt.title('Image regions')
	''' Draw Letter blob '''

	# blankRaster = np.zeros(np.shape(imageraster))
	# ax3 = plt.subplot2grid(gridsize, (0, 1), rowspan=1)
	# ax3.imshow(blankRaster)
	ax3.imshow(imageraster)
	ax3.grid()
	thismanager = pylab.get_current_fig_manager()
	thismanager.window.wm_geometry("+0+0")

	print("Keys:",regionMap.keys())
	return imageraster, regionMap


def processMask(mask, dradius, indexLabel):
	xrange, yrange = np.shape(mask)
	meshObj = Bridson_MeshObj.MeshObject(mask=mask, dradius=dradius, indexLabel=indexLabel)
	points = meshObj.points
	tri = meshObj.triangulation
	fakeRadius = max(xrange,yrange)

	createMeshFile(points, tri, fakeRadius, (xrange/2.0, yrange/2.0))
	BFFReshape()
	FlattenMesh()

	flatvertices, flatfaces = Bridson_readOBJFile.readFlatObjFile(path = "../../boundary-first-flattening/build/", filename="test1_out_flat.obj")
	flatMeshObj = Bridson_MeshObj.MeshObject(flatvertices=flatvertices, flatfaces=flatfaces, xrange=xrange, yrange=yrange, indexLabel=indexLabel)
	return flatMeshObj

def displayRegionRaster(regionRaster, index):
	plt.figure()
	ax = plt.subplot(1, 1, 1, aspect=1, label='Region Raster ' + str(index))
	plt.title('Region Raster ' + str(index))
	''' Draw Letter blob '''

	# blankRaster = np.zeros(np.shape(imageraster))
	# ax3 = plt.subplot2grid(gridsize, (0, 1), rowspan=1)
	# ax3.imshow(blankRaster)
	ax.imshow(regionRaster)
	ax.grid()


if __name__ == '__main__':
	cleanUpFiles()
	dradius = 1.5
	xrange, yrange = 100, 100

	if False:
		mask = Bridson_CreateMask.genLetter(xrange, yrange, character='Y')
		processMask(mask, dradius, 0)
	# meshObj = Bridson_MeshObj.MeshObject(mask=mask, dradius=dradius)

	'''
	# Original code for generating the Mesh with Mask.
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
	'''
	# points = meshObj.points
	# tri = meshObj.triangulation
	# fakeRadius = max(xrange,yrange)
	#
	# createMeshFile(points, tri, fakeRadius, (xrange/2.0, yrange/2.0))
	# BFFReshape()
	# FlattenMesh()
	#
	# flatvertices, flatfaces = Bridson_readOBJFile.readFlatObjFile(path = "../../boundary-first-flattening/build/", filename="test1_out_flat.obj")
	#
	# Create Mesh object for flattened file.
	# flatMeshObj = Bridson_MeshObj.MeshObject(flatvertices=flatvertices, flatfaces=flatfaces, xrange=xrange, yrange=yrange)

	imageraster, regionMap = SLICImage()

	if False:
		startIndex = 0
		stopIndex = 1
		for regionIndex in range(startIndex, stopIndex):
		# for regionIndex in regionMap.keys():
			print(">>>>>>>>>>> Start of cycle")
			# resetVariables()
			# try:
			print("Generating index:", regionIndex)
			raster, actualTopLeft = SLIC.createRegionRasters(regionMap, regionIndex)
			displayRegionRaster( raster[:], regionIndex )

	for index in range(9, 10):
		cleanUpFiles()
	# Generate the raster for the first region.
		raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
		print(raster)
		flatMeshObj = processMask(raster, dradius, index)
		flatMeshObj.DrawVerticalLines()


	plt.show()


