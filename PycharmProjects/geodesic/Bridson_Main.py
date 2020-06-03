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
import Bridson_Common
import random
import sys

random.seed(Bridson_Common.seedValue)
print("Seed was:", Bridson_Common.seedValue)
np.random.seed(Bridson_Common.seedValue)


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
	if Bridson_Common.normalizeUV:
		os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --flattenToDisk --normalizeUVs ")
	else:
		os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --flattenToDisk ")
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


def featureRemoval(mask, dradius, indexLabel):
	currentMask = mask[:]
	# print(currentMask)

	for i in range(10):
		# random.seed(Bridson_Common.seedValue)
		# print("Seed was:", Bridson_Common.seedValue)
		# np.random.seed(Bridson_Common.seedValue)

		# if Bridson_Common.debug:
		plt.figure()
		plt.subplot(1, 1, 1, aspect=1)
		plt.title('Newly Formed Mask ' + str(indexLabel + i / 10))
		plt.imshow(currentMask)
		thismanager = pylab.get_current_fig_manager()
		thismanager.window.wm_geometry("+0+560")
		currentMask = blankRow(currentMask, i)
		processMask(currentMask, dradius, indexLabel + i/10)


def blankRow(mask, row):
	newMask = np.array(mask)
	height = np.shape(newMask)[0]
	zeroMask = newMask!=0
	rowCount = 3

	firstRow = rowCount

	for i in range(height-1,0,-1):
		if np.max(zeroMask[i]) > 0:
			firstRow = i
			print("FirstRow: ", firstRow)
			break

	for i in range(rowCount):
		newMask[firstRow - i ] = 0
	return newMask





def processMask(mask, dradius, indexLabel):
	# invertedMask = Bridson_CreateMask.InvertMask(mask)
	# invertedMask = Bridson_Common.blurArray(mask, 3)
	mask5x = Bridson_CreateMask.InvertMask(mask)

	if Bridson_Common.debug:
		plt.figure()
		plt.subplot(1, 1, 1, aspect=1)
		plt.title('original Mask')
		plt.imshow(mask)
		thismanager = pylab.get_current_fig_manager()
		thismanager.window.wm_geometry("+0+560")

	xrange, yrange = np.shape(mask)
	# mask5x = Bridson_Common.blurArray(mask5x, 5)
	blurRadius = math.ceil(dradius)
	if blurRadius % 2 == 0:
		blurRadius = blurRadius + 3
	blurRadius = 5
	print("*** BlurRadius: ", blurRadius)

	mask5x = Bridson_Common.blurArray(mask5x, blurRadius)
	mask5x = Bridson_CreateMask.InvertMask(mask5x)

	if Bridson_Common.debug:
		plt.figure()
		plt.subplot(1, 1, 1, aspect=1)
		plt.title('blurred Mask')
		plt.imshow(mask5x)
		thismanager = pylab.get_current_fig_manager()
		thismanager.window.wm_geometry("+0+560")


	meshObj = Bridson_MeshObj.MeshObject(mask=mask5x, dradius=dradius, indexLabel=indexLabel)
	points = meshObj.points
	tri = meshObj.triangulation
	fakeRadius = max(xrange,yrange)

	createMeshFile(points, tri, fakeRadius, (xrange/2.0, yrange/2.0))

	# vertices, faces = Bridson_readOBJFile.readFlatObjFile(path="../../boundary-first-flattening/build/",
	#                                                               filename="test1.obj")
	# meshObj = Bridson_MeshObj.MeshObject(flatvertices=vertices, flatfaces=faces, xrange=xrange,
	#                                          yrange=yrange, indexLabel=indexLabel)
	BFFReshape()
	FlattenMesh()

	flatvertices, flatfaces = Bridson_readOBJFile.readFlatObjFile(path = "../../boundary-first-flattening/build/", filename="test1_out_flat.obj")

	Bridson_Common.triangleHistogram(flatvertices, flatfaces, indexLabel)

	newIndex = str(indexLabel) + ":" + str(indexLabel)
	flatMeshObj = Bridson_MeshObj.MeshObject(flatvertices=flatvertices, flatfaces=flatfaces, xrange=xrange, yrange=yrange, indexLabel=indexLabel)
	return meshObj, flatMeshObj

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
	dradius = 1.5 # 3 seems to be the maximum value.
	xrange, yrange = 10, 10

	# mask = Bridson_CreateMask.CreateCircleMask(xrange, yrange, 10)

	if False:
		mask = Bridson_CreateMask.genLetter(xrange, yrange, character='Y')
		meshObj, flatMeshObj = processMask(mask, dradius, 0)
		# flatMeshObj.DrawVerticalLines()
		# meshObj.DrawVerticalLines()
		# Transfer the lines from the FlatMesh to meshObj.
		# meshObj.TransferLinePoints( flatMeshObj )

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
		stopIndex = 2
		for regionIndex in range(startIndex, stopIndex):
		# for regionIndex in regionMap.keys():
			print(">>>>>>>>>>> Start of cycle")
			# resetVariables()
			# try:
			print("Generating index:", regionIndex)
			raster, actualTopLeft = SLIC.createRegionRasters(regionMap, regionIndex)
			displayRegionRaster( raster[:], regionIndex )

	if True:
			for index in range(3,4):
				# Generate the raster for the first region.
				raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
				# print(raster)
				# Bridson_Common.arrayInformation( raster )
				for i in range(1):
					Bridson_Common.writeMask(raster)
					meshObj, flatMeshObj = processMask(raster, dradius, index + i / 10)
				# flatMeshObj.DrawVerticalLines()

				# Transfer the lines from the FlatMesh to meshObj.
				# meshObj.TransferLinePoints( flatMeshObj )

	print("\n\n\n")
	if False:
		index = 999
		# File are in /Users/hengsun/Documents/Thesis/PycharmProjects/geodesic.
		mask = Bridson_Common.readMask()
		# mask = Bridson_Common.readMask(filename='BlurSquare.gif')
		# mask = Bridson_Common.readMask(filename='BlurTriangle.gif')
		meshObj, flatMeshObj = processMask(mask, dradius, index)
		# flatMeshObj.DrawVerticalLines()

	# Perform feature removal, row by row.
	if False:
		for index in range(3,4):
			# Generate the raster for the first region.
			# raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
			# print(raster)
			# Bridson_Common.arrayInformation( raster )
			# Bridson_Common.writeMask(raster)
			raster = Bridson_Common.readMask(filename='BlurTriangle.gif')
			print("Minimum value of Raster: " , np.min(raster))
			print("Maximum value of Raster: ", np.max(raster))
			raster = Bridson_CreateMask.InvertMask(raster)
			print("Minimum value of Inverted Raster: ", np.min(raster))
			print("Maximum value of Inverted Raster: ", np.max(raster))
			featureRemoval(raster, dradius, index)
			# flatMeshObj.DrawVerticalLines()

			# Transfer the lines from the FlatMesh to meshObj.
			# meshObj.TransferLinePoints( flatMeshObj )


	if False:
		for index in range(1,2):
			# Generate the raster for the first region.
			raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
			# print(raster) = SLIC.createRegionRasters(regionMap, index)
			# print(raster)
			# Bridson_Common.arrayInformation( raster )
			# Bridson_Common.writeMask(raster)
			featureRemoval(raster, dradius, index)
			# flatMeshObj.DrawVerticalLines()

			# Transfer the lines from the FlatMesh to meshObj.
			# meshObj.TransferLinePoints( flatMeshObj )


	plt.show()


