import matplotlib
matplotlib.use("tkagg")

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
import Bridson_Common
import random
import sys
import Bridson_FinishedImage
from skimage.segmentation import mark_boundaries
import datetime
import gc

import Bridson_StructTensor
from multiprocessing import Process, freeze_support, set_start_method, Pool
import multiprocessing as mp
import uuid
import traceback


# Redirect print statements to file.


random.seed(Bridson_Common.seedValue)
np.random.seed(Bridson_Common.seedValue)
# Bridson_Common.logDebug(__name__, "Seed was:", Bridson_Common.seedValue)
import subprocess

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
	Bridson_Common.logDebug(__name__, np.shape(points))
	# Bridson_Common.logDebug(__name__, np.shape(points))

	# Merge border with square perimeter.
	points = np.append( points, border, axis=0)

	# Generate all the sample points.
	points = Bridson_sampling(width=xrange, height=yrange, radius=radius, existingPoints=points, mask=mask)
	Bridson_Common.logDebug(__name__, np.shape(points))

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
	print("Creating Mesh File: ", Bridson_Common.test1obj)
	path = Bridson_Common.objPath
	# Create object file for image.
	# Bridson_createOBJFile.createObjFile2D(path, "test1.obj", samples, tri, radius, center, distance=euclidean_distance)
	Bridson_createOBJFile.createObjFile2D(path, Bridson_Common.test1obj, samples, tri, radius, center, distance=euclidean_distance)




def BFFReshape():
	# print("BFFReshape")
	# Using pipes to kill a hung process: https://stackoverflow.com/questions/41094707/setting-timeout-when-using-os-system-function
	Bridson_Common.logDebug(__name__, "Reshaping with BFF")
	path = Bridson_Common.objPath
	# os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=1 --normalizeUVs ")
	#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	''''''
	if Bridson_Common.normalizeUV:
		# os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --flattenToDisk --normalizeUVs ")
		# parameters = " " + path + "test1.obj " + path + "test1_out.obj --flattenToDisk --normalizeUVs "
		parameters = " " + path + Bridson_Common.test1obj + " " + path + Bridson_Common.test1_outobj  + " --flattenToDisk --normalizeUVs "
	else:
		# os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --flattenToDisk ")
		# parameters = " " + path + "test1.obj " + path + "test1_out.obj --flattenToDisk "
		parameters = " " + path + Bridson_Common.test1obj + " " + path + Bridson_Common.test1_outobj + " --flattenToDisk "

	#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

	print("Attempting BFF")
	p = subprocess.Popen(path + 'bff-command-line' + parameters, shell=True, stdout=subprocess.PIPE)
	try:
		p.wait(Bridson_Common.timeoutPeriod)
	except subprocess.TimeoutExpired:
		print("Killed BFF")
		p.kill()
		p.kill()
	print("Finished BFF")

	# This handles the case where BFF has a problem processing the mesh, e.g. the mesh has a manifold.
	if p.returncode != 0:
		print("Unexpected error in BFFReshape:", p.returncode)
		raise Exception

	# os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=1 --normalizeUVs --nCones=" + str(perimeterSegments))
	# os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=1 --normalizeUVs --nCones=6")

def FlattenMesh():
	# print("FlattenMesh")
	path = Bridson_Common.objPath
	Bridson_Common.logDebug(__name__, "Extracting 2D image post BFF Reshaping")
	# os.system(path + "extract.py test1_out.obj test1_out_flat.obj")
	os.system(path + "extract.py " + Bridson_Common.test1_outobj + " " + Bridson_Common.test1_out_flatobj)

def cleanUpFiles():
	path = Bridson_Common.objPath
	# os.system("rm " + path + "test1.obj ")
	# os.system("rm "+ path + "test1_out.obj")
	# os.system("rm " + path + "test1_out_flat.obj")
	os.system("rm " +  Bridson_Common.test1obj)
	os.system("rm "+  Bridson_Common.test1_outobj)
	os.system("rm " +  Bridson_Common.test1_out_flatobj)
	os.system("rm " +  Bridson_Common.chaincodefile)


def SLICImage(filename):
	startIndex = 0 # Index starts at 0.
	regionIndex = startIndex
	imageraster, regionMap, segments, regionIntensityMap, regionColourMap = SLIC.callSLIC(filename)
	print("CC")
	Bridson_Common.outputEnvironmentVariables()

	# stopIndex=startIndex+16
	print("CC1")
	fig = plt.figure()
	ax3 = fig.add_subplot(1, 1, 1, aspect=1, label='Image regions')
	plt.title('Image regions - ' + filename + ' - Segments: ' + str(Bridson_Common.segmentCount) )
	# fig.canvas.set_window_title('Image regions - ' + filename + ' - Segments: ' + str(Bridson_Common.segmentCount) + ' - regionPixels: ' + str(Bridson_Common.targetRegionPixelCount))
	''' Draw Letter blob '''

	# blankRaster = np.zeros(np.shape(imageraster))
	# ax3 = plt.subplot2grid(gridsize, (0, 1), rowspan=1)
	# ax3.imshow(blankRaster)
	# ax3.imshow( imageraster, cmap='Greys', norm=matplotlib.colors.Normalize())
	# ax3.imshow( imageraster, cmap='Greys' )
	print("B")
	regionRaster = imageraster / np.max(imageraster)   # Need to normalize the region intensity [0 ... 1.0] to display properly.
	# print("Raster:", displayRaster)
	ax3.imshow(mark_boundaries( regionRaster, segments, color=(255,0,0) ))
	ax3.grid()
	# plt.clf()

	print("C")
	if Bridson_Common.GreyscaleSLIC:
		Bridson_Common.saveImage(filename, "GreyscaleSLIC", fig)
	# plt.clf()
	# print("D")
	thismanager = pylab.get_current_fig_manager()
	thismanager.window.wm_geometry("+0+0")
	# print("E")
	Bridson_Common.logDebug(__name__, "SLIC Keys:" + str(regionMap.keys()) )
	return imageraster, regionMap, regionRaster, segments, regionIntensityMap, regionColourMap


def featureRemoval(mask, dradius, indexLabel):
	currentMask = mask[:]
	# Bridson_Common.logDebug(__name__, currentMask)

	for i in range(10):
		# random.seed(Bridson_Common.seedValue)
		# Bridson_Common.logDebug(__name__, "Seed was:", Bridson_Common.seedValue)
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
			Bridson_Common.logDebug(__name__, "FirstRow: ", firstRow)
			break

	for i in range(rowCount):
		newMask[firstRow - i ] = 0
	return newMask




def processMask(mask, dradius, indexLabel):
	# invertedMask = Bridson_CreateMask.InvertMask(mask)
	# invertedMask = Bridson_Common.blurArray(mask, 3)
	successful = False
	attempts = 0
	maxAttempts = 10
	blurRadius = 3

	while successful == False and attempts < maxAttempts:
		attempts += 1
		try:
			mask5x = Bridson_CreateMask.InvertMask(mask)

			if Bridson_Common.debug:
				plt.figure()
				plt.subplot(1, 1, 1, aspect=1)
				plt.title('original Mask')
				plt.imshow(mask)
				thismanager = pylab.get_current_fig_manager()
				thismanager.window.wm_geometry("+0+560")

			xrange, yrange = np.shape(mask)

			# Bridson_Common.determineRadius(xrange, yrange) # Set the region specific radius.

			Bridson_Common.logDebug(__name__, "*** BlurRadius: " , blurRadius)
			print("blurRadius:", blurRadius)
			mask5x = Bridson_Common.blurArray(mask5x, blurRadius)
			mask5x = Bridson_CreateMask.InvertMask(mask5x)
			# print("A")
			# print(distanceRaster)
			if Bridson_Common.debug:
			# if True:
				plt.figure()
				plt.subplot(1, 1, 1, aspect=1)
				plt.title('blurred Mask')
				plt.imshow(mask5x)
				thismanager = pylab.get_current_fig_manager()
				thismanager.window.wm_geometry("+0+560")
			# print("A")
		# try:
			a = datetime.datetime.now()
			meshObj = Bridson_MeshObj.MeshObject(mask=mask5x, dradius=dradius, indexLabel=indexLabel) # Create Mesh based on Mask.
			# print("B")
			b = datetime.datetime.now()
			points = meshObj.points

			tri = meshObj.triangulation
			d = datetime.datetime.now()
			fakeRadius = max(xrange,yrange)
			# print("C")

			createMeshFile(points, tri, fakeRadius, (xrange/2.0, yrange/2.0))

			# vertices, faces = Bridson_readOBJFile.readFlatObjFile(path="../../boundary-first-flattening/build/",
			#                                                               filename="test1.obj")
			# meshObj = Bridson_MeshObj.MeshObject(flatvertices=vertices, flatfaces=faces, xrange=xrange,
			#                                          yrange=yrange, indexLabel=indexLabel)
			f = datetime.datetime.now()
			BFFReshape()
			g = datetime.datetime.now()
			FlattenMesh()
			h = datetime.datetime.now()
			# flatvertices, flatfaces = Bridson_readOBJFile.readFlatObjFile(path = "../../boundary-first-flattening/build/", filename="test1_out_flat.obj")
			flatvertices, flatfaces = Bridson_readOBJFile.readFlatObjFile(path=Bridson_Common.objPath,
			                                                              filename=Bridson_Common.test1_out_flatobj)
			i = datetime.datetime.now()
			Bridson_Common.triangleHistogram(flatvertices, flatfaces, indexLabel)
			# print("D")
			newIndex = str(indexLabel) + ":" + str(indexLabel)
			flatMeshObj = Bridson_MeshObj.MeshObject(flatvertices=flatvertices, flatfaces=flatfaces, xrange=xrange, yrange=yrange, indexLabel=indexLabel) # Create Mesh based on OBJ file.
			j = datetime.datetime.now()
			# print("E")
			successful = flatMeshObj.trifinderGenerated
			if False:
				print("Generate Mesh from Mask:", (b-a).microseconds )
				print("Generate Triangulation on MeshObj:", (d-b).microseconds )
				print("CreateMeshFile:", (f-d).microseconds )
				print("BFFRefreshape:", (g-f).microseconds )
				print("FlattenMesh:", (h-g).microseconds )
				print("ReadObjFile:", (i-h).microseconds )
				print("Create Mesh from File:", (j-i).microseconds )
		except Exception as e:
			print("processMask main failure:", e)
			successful = False
		# print("G")
		if successful:
			print("Attempt ", attempts, " successful")
		else:
			print("Attempt ", attempts, " UNsuccessful")
		blurRadius += 2
		blurRadius = 7 if blurRadius > 7 else blurRadius # Limit the blurRadius to 9.

	# indexLabel="LineSeed"
	# lineReferencePointsObj = Bridson_MeshObj.MeshObject(mask=mask5x, dradius=dradius*Bridson_Common.lineRadiusFactor, indexLabel=indexLabel)

	return meshObj, flatMeshObj, successful

def displayRegionRaster(regionRaster, index):
	if Bridson_Common.displayMesh:
		plt.figure()
		ax = plt.subplot(1, 1, 1, aspect=1, label='Region Raster ' + str(index))
		plt.title('Region Raster ' + str(index))
		''' Draw Letter blob '''

		# blankRaster = np.zeros(np.shape(imageraster))
		# ax3 = plt.subplot2grid(gridsize, (0, 1), rowspan=1)
		# ax3.imshow(blankRaster)
		ax.imshow(regionRaster)
		ax.grid()



def drawRegionLines( filename, finishedImage, regionList):

	regionMap = finishedImage.regionMap
	maskRasterCollection = finishedImage.maskRasterCollection
	meshObjCollection = finishedImage.meshObjCollection

	successfulRegions = 0

	for index in regionList:
		print("(**** ", filename, " Starting Region: ", index, "of", len(regionMap.keys()), "  *****" )

		raster = maskRasterCollection[index]
		displayRegionRaster(raster, index)

		indexLabel = index # + i / 10
		# Bridson_Common.writeMask(raster)
		Bridson_Common.determineRadius(np.shape(raster)[0], np.shape(raster)[1])
		dradius = Bridson_Common.dradius
		meshObj, flatMeshObj, trifindersuccess = processMask(raster, dradius, indexLabel)


		# Find the intensity of this region.
		# Produce angle based on the intensity.
		# find the dx, dy.
		# Find two points that fulfill the dx,dy.
		# Calculate the barycentric coordinates in flat mesh coordinates.
		# Calculate the angle in flat mesh coordinates.
		if trifindersuccess:

			meshAngle = finishedImage.regionDirection[ index ]  # Obtain the angle that was calculated.
			flatAngle = int( flatMeshObj.calculateAngle( meshObj, desiredAngle=meshAngle ) )  # Determine the angle on the flat mesh.
			print("MeshAngle:", meshAngle, "Flat angle:", flatAngle)

			# if trifindersuccess:
			successfulRegions += 1
			print("Trifinder was successfully generated for region", index)
			if Bridson_Common.diagnostic == False:
				# Only draw the lines if the trifinder was successful generated.
				if Bridson_Common.linesOnFlat:
					if Bridson_Common.verticalLines:
						flatMeshObj.DrawAngleLinesExteriorSeed2(raster, angle=flatAngle )
						# flatMeshObj.DrawAngleLinesExteriorSeed3(raster, regionMap, index, intensity=regionIntensityMap[index], angle=flatAngle)
						flatMeshObj.checkLinePoints() # diagnostic check for empty lines.
					else:
						flatMeshObj.DrawAngleLinesExteriorSeed2()

					# Transfer the lines from the FlatMesh.
					meshObj.TransferLinePointsFromTarget(flatMeshObj)
				else:
					if Bridson_Common.verticalLines:
						meshObj.DrawAngleLinesExteriorSeed2()
					else:
						meshObj.DrawAngleLinesExteriorSeed2()

					flatMeshObj.TransferLinePointsFromTarget(meshObj)

			# if True:  # Rotate original image 90 CW.
			# print("About to call rotateClockwise90")
			meshObj.rotateClockwise90()
			# print("Rotated 90 clockwise")
			meshObjCollection[ index ] = meshObj
			# print("Save NoSLIC meshObj")
			# NoSLICmeshObjCollection[ index ] = meshObj
		else:
			print("Trifinder was NOT successfully generated for region ", index)




def indexValidation(filename):
	print(">>>>>>>>>>>>>>>> Calling SLIC for ", filename, "<<<<<<<<<<<<<<<<<<<<<<<")
	imageraster, regionMap, regionRaster, segments, regionIntensityMap, regionColourMap = SLICImage(filename)

	# imageShape = np.shape(regionRaster)
	# Bridson_Common.segmentCount = int((imageShape[0]*imageShape[1]) / Bridson_Common.targetRegionPixelCount)
	# print("Target Region Count:", Bridson_Common.segmentCount )
	# print("Image Shape:", imageShape )
	successfulRegions = 0
	# Create new image for contour lines.  Should be the same size as original image.
	finishedImageSLIC = Bridson_FinishedImage.FinishedImage()
	finishedImageSLIC.setTitle(filename)
	# print( "Region Raster: ", regionRaster )
	finishedImageSLIC.drawSLICRegions( regionRaster, segments )
	finishedImageSLIC.setTitle(filename)

	# finishedImage.setXLimit( 0, np.shape(imageraster)[0])
	finishedImageNoSLIC = Bridson_FinishedImage.FinishedImage()
	finishedImageNoSLIC.setTitle(filename + "_POST_")
	finishedImageNoSLIC.setXLimit(0, np.shape(regionRaster)[1])
	finishedImageNoSLIC.setYLimit(0, -np.shape(regionRaster)[0])

	# finishedImage.setXLimit( 0, np.shape(imageraster)[0])
	finishedImageNoSLICPRE = Bridson_FinishedImage.FinishedImage()
	finishedImageNoSLICPRE.setTitle(filename + "_PRE_")
	# finishedImageNoSLICPRE.setXLimit(0, np.shape(regionRaster)[1])
	# finishedImageNoSLICPRE.setYLimit(0, -np.shape(regionRaster)[0])


	maskRasterCollection = {}
	meshObjCollection = {}
	NoSLICmaskRasterCollection = {}
	# NoSLICmeshObjCollection = {}
	regionDirection = {}

	# for index in range(10,15):  # Interesting regions: 11, 12, 14
	# print("RegionMap keys:", regionMap.keys())
	if Bridson_Common.bulkGeneration == False:
		regionList = range(  54, 57)
		regionList = range(len(regionMap.keys()))
	else:
		regionList = range(len(regionMap.keys()) )
	# for index in range(5,10):
	# for index in range( len(regionMap.keys()) ):

	originalImage = Bridson_Common.readImagefile( filename )  # We read the image in as greyscale... should we?

	plt.figure()
	plt.imshow( originalImage )
	plt.title("Original Image")

	for index in regionList:
		# Generate the raster for the first region.
		raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
		maskRasterCollection[index] = raster.copy()  # Make a copy of the mask raster.
		NoSLICmaskRasterCollection[ index ] = raster.copy()
		regionDirection[ index ] = 0  # Default the direction to 0.

	# Set the variables



	##### Process PRE image
	if Bridson_Common.generatePREImage:
		finishedImageNoSLICPRE.setMaps(regionMap.copy(), regionRaster.copy(), maskRasterCollection.copy(), meshObjCollection.copy(), regionIntensityMap.copy())
		finishedImageNoSLICPRE.shiftRastersMeshObj(regionMap, regionRaster, originalImage)
		finishedImageNoSLICPRE.calculateRegionDirection(regionList)
		finishedImageNoSLICPRE.generateRAG(filename, segments, regionColourMap)
		# Generate the PRE version of the image.
		drawRegionLines(filename, finishedImageNoSLICPRE, regionList)
		finishedImageNoSLICPRE.shiftLinePoints()
		finishedImageNoSLICPRE.cropCullLines()
		finishedImageNoSLICPRE.genLineAdjacencyMap()
		finishedImageNoSLICPRE.mergeLines()
		for index in meshObjCollection.keys():
			finishedImageNoSLICPRE.drawRegionContourLines(index, drawSLICRegions=False)
		Bridson_Common.saveImage(filename, "NoSlic_PRE", finishedImageNoSLICPRE.fig)


	###### Process finishedImageSLIC
	finishedImageSLIC.setMaps(regionMap, regionRaster, maskRasterCollection, meshObjCollection, regionIntensityMap)
	# Create region raster in the actual location.
	print("0A")
	finishedImageSLIC.shiftRastersMeshObj(regionMap, regionRaster, originalImage)
	print("0B")
	# Calculate RAG (Region Adjacency Graph)
	finishedImageSLIC.calculateRegionDirection(regionList)
	finishedImageSLIC.generateRAG(filename, segments, regionColourMap)
	finishedImageSLIC.adjustRegionAngles2()
	drawRegionLines( filename, finishedImageSLIC, regionList )
	finishedImageSLIC.shiftLinePoints()
	# At this point, we need can attempt to merge the lines between each region.
	# Still have a problem with the coordinates though.
	print("About to cropCullLines")
	finishedImageSLIC.cropCullLines()
	print("About to genLineAdjacencyMap")
	finishedImageSLIC.genLineAdjacencyMap()
	print("About to mergeLines")
	finishedImageSLIC.mergeLines()


	print("0D")


		# for index in meshObjCollection.keys():
		# 	finishedImageNoSLICPRE.drawRegionContourLines(index, drawSLICRegions=False)
		# Bridson_Common.saveImage(filename, "NoSLIC_PRE", finishedImageNoSLICPRE.fig)


	print("About to copyFromOther")
	finishedImageNoSLIC.copyFromOther( finishedImageSLIC )
	print("0I")

	for index in meshObjCollection.keys():
		# Draw the region contour lines onto the finished image.
		# meshObj = meshObjCollection[ index ]
		# print("About to call drawRegionContourLines on SLIC image.")
		# Bridson_Common.debug = True
		finishedImageSLIC.drawRegionContourLines( index,  drawSLICRegions=True )
		# print("About to call drawRegionContourLines on NON SLIC image.")
		finishedImageNoSLIC.drawRegionContourLines( index,  drawSLICRegions=False )
		# finishedImageNoSLICPRE.drawRegionContourLines(index, drawSLICRegions=False)
	print("0J")

	# finishedImageNoSLIC.highLightEdgePoints( drawSLICRegions=False )
	# finishedImageSLIC.highLightEdgePoints(drawSLICRegions=True )

	Bridson_Common.saveImage(filename, "WithSLIC", finishedImageSLIC.fig )
	Bridson_Common.saveImage(filename, "NoSLIC_POST", finishedImageNoSLIC.fig)
	# Bridson_Common.saveImage(filename, "NoSlic_PRE", finishedImageNoSLICPRE.fig)

	print("0K")
	print("Successful Regions: ", successfulRegions)
	print("Total Regions: ", len(regionMap.keys()) )
	print("||||||||||||||||||||||||||||||||||")
	print("")

		# Obtain the points from the

			# Apply BFF again.
			# BFFReshape()
			# FlattenMesh()
			#
			# flatvertices, flatfaces = Bridson_readOBJFile.readFlatObjFile(path="../../boundary-first-flattening/build/",	filename="test1_out_flat.obj")
			# newIndex = indexLabel + 0.0012345
			# Bridson_Common.triangleHistogram(flatvertices, flatfaces, newIndex)


			# flatMeshObj2 = Bridson_MeshObj.MeshObject(flatvertices=flatvertices, flatfaces=flatfaces, xrange=xrange,
			#                                          yrange=yrange, indexLabel=newIndex)
			# successful = flatMeshObj2.trifinderGenerated
			# Transfer the lines from the FlatMesh to meshObj.
			# flatMeshObj2.TransferLinePointsFromTarget(flatMeshObj)



# Validate that barycentric works.
	# meshObj2, flatMeshObj2 = processMask(raster, dradius, index + i / 10 + 0.0012345)
	# flatMeshObj2.TransferLinePoints(flatMeshObj)

def wrapper(filename, segmentCount, compactness, cnn):
	gc.collect()
	baseName = 'f' + str(uuid.uuid4().hex)
	Bridson_Common.test1obj = baseName + '.obj'
	Bridson_Common.test1_outobj = baseName + '_out.obj'
	Bridson_Common.test1_out_flatobj = baseName + '_out_flat.obj'
	Bridson_Common.chaincodefile = baseName + '_chaincode.txt'

	# cleanUpFiles()
	random.seed(Bridson_Common.seedValue)
	np.random.seed(Bridson_Common.seedValue)
	# Bridson_Common.targetRegionPixelCount = targetPixel
	Bridson_Common.segmentCount = segmentCount
	# Set the seed each time.
	Bridson_Common.compactnessSLIC = compactness
	Bridson_Common.semanticSegmentation = cnn
	try:
		indexValidation(filename)
		# print('Handling file ', filename + Bridson_Common.test1obj )
	# sys.stdout.close()
	except Exception as e:
		print("Exception calling indexValidation for filename:", filename, e)
		print("Error details:", sys.exc_info()[0])
		exc_info = sys.exc_info()
		traceback.print_exception(*exc_info)
		del exc_info


	plt.close("all")
	gc.collect()
	cleanUpFiles()
	# sys.exit(0)


if __name__ == '__main__':
	# Create unique names for output files.
	dradius = Bridson_Common.dradius # 3 seems to be the maximum value.
	xrange, yrange = 10, 10

	# mask = Bridson_CreateMask.CreateCircleMask(xrange, yrange, 10)


	# regionMap = []
	images = []
	# images.append('SimpleR.png')
	# images.append('SimpleC.png')
	# images.append('simpleHorizon.png')
	# images.append('FourSquares.png')
	# images.append('SimpleSquare.jpg')
	# images.append("FourCircles.png")
	# images.append("SimpleSquares.png")
	images.append('simpleTriangle.png')
	images.append('Stripes.png')

	# Batch A.
	# images.append('RedApple.jpg')
	# images.append('Sunglasses.jpg')
	# images.append('TapeRolls.jpg')
	# images.append('Massager.jpg')
	# images.append('eyeball.jpg')
	# images.append('truck.jpg')
	# images.append('cat1.jpg')

	# Original
	# images.append('dog2.jpg')

	# Batch B.
	# images.append('moon1.jpg')
	# images.append('popsicle.jpg')
	# images.append('rainbow.jpg')
	# images.append('fishhead.jpg')
	# images.append('bald-eagle.jpg')
	# images.append('grapes.jpg')
	# images.append('green-tree-frog.jpg')
	# images.append('hand.jpg')


	# Batch C.
	# images.append('alex-furgiuele-UkH7L-aag8A-unsplash_Cactus.jpg')
	# images.append('meritt-thomas-Ao09kk2ovB0-unsplash_Cupcake.jpg')
	# images.append('aleksandra-antic-Xnqj9FvHycM-unsplash_Cupcake.jpg')
	# images.append('faris-mohammed-oAlRgZXsXUI-unsplash_Eggs.jpg')
	# images.append('ruben-rodriguez-GFZZmRbyPFQ-unsplash_Egg.jpg')
	# images.append('gryffyn-m-rpm07rS8Rl8-unsplash_Shoe.jpg')
	# images.append('dan-gold-N7RiDzfF2iw-unsplash_VWBeetle.jpg')
	# images.append('herson-rodriguez-w8CcH9Md4vE-unsplash_Van.jpg')
	# images.append('lucia-lua-ramirez-lG0AHN1Gapw-unsplash_Bus.jpg')
	# images.append('pawel-czerwinski-xt1tPXqOdcc-unsplash_TrafficLight.jpg')
	# images.append('joshua-hoehne-WPrTKRw8KRQ-unsplash_StopSign.jpg')
	# images.append('devvrat-jadon-WLNkAHCjYOw-unsplash_Hammer.jpg')
	# images.append('magic-bowls-3QGtPOqeBEQ-unsplash.jpg')
	# images.append('ruslan-keba-G5tOIWFZqFE-unsplash_RubiksCube.jpg')

	# Batch D
	images.append('david-dibert-Huza8QOO3tc-unsplash.jpg')
	images.append('everyday-basics-i0ROGKijuek-unsplash.jpg')
	images.append('imani-bahati-LxVxPA1LOVM-unsplash.jpg')
	images.append('kaitlyn-ahnert-3iQ_t2EXfsM-unsplash.jpg')
	images.append('luis-quintero-qKspdY9XUzs-unsplash.jpg')
	images.append('miguel-andrade-nAOZCYcLND8-unsplash.jpg')
	images.append('mr-o-k--ePHy6jg_7c-unsplash.jpg')
	images.append('valentin-lacoste-GcepdU3MyKE-unsplash.jpg')

	semanticSegmentation = ['none', 'mask_rcnn',  'both']
	# semanticSegmentation = ['none']
	# semanticSegmentation = ['none', 'mask_rcnn', 'deeplabv3', 'both']
	# semanticSegmentation = ['mask_rcnn', 'deeplabv3', 'both']
	# semanticSegmentation = ['both', 'none']
	# semanticSegmentation.append('none')


	if Bridson_Common.bulkGeneration == False:
		images = ['david-dibert-Huza8QOO3tc-unsplash.jpg']
		semanticSegmentation = ['mask_rcnn']
		semanticSegmentation = ['none']

	# percentages = [0.05, 0.1, 0.15, 0.2]
	targetPixels = [  400, 800]
	# targetPixels = [  800 ]
	targetPixels = [3200]
	if Bridson_Common.bulkGeneration:
		# segmentCounts = [100, 200]
		segmentCounts = [200, 300, 400]
		# segmentCounts = [200]
		compactnessList = [ 0.1, 0.25, 0.5]
		if Bridson_Common.SLIC0:
			compactnessList = [0.01]
		else:
			compactnessList = [ 20, 40]
		# compactnessList = [1]
	else:
		segmentCounts = [200]
		compactnessList = [ 5 ]

	if Bridson_Common.smallBatch:
		images = []
		images.append('simpleTriangle.png')
		# images.append('Stripes.png')
		images.append('kaitlyn-ahnert-3iQ_t2EXfsM-unsplash.jpg')
		images.append('valentin-lacoste-GcepdU3MyKE-unsplash.jpg')
		images.append('david-dibert-Huza8QOO3tc-unsplash.jpg')
		semanticSegmentation = ['none']
		segmentCounts = [200]
		compactnessList = [20]

	variables = []
	for filename in images:
		# sys.stdout = open("./output/" + filename + ".log", "w")
		# for targetPixel in targetPixels:
		for segmentCount in segmentCounts:
			for compactness in compactnessList:
				for cnn in semanticSegmentation:
					variables.append( (filename, segmentCount, compactness, cnn) )

					if Bridson_Common.bulkGeneration == False:
						baseName = 'f' + str(uuid.uuid4().hex)
						Bridson_Common.test1obj = baseName + '.obj'
						Bridson_Common.test1_outobj = baseName + '_out.obj'
						Bridson_Common.test1_out_flatobj = baseName + '_out_flat.obj'

						# Bridson_Common.targetRegionPixelCount = targetPixel
						Bridson_Common.segmentCount = segmentCount
						# Set the seed each time.
						Bridson_Common.compactnessSLIC=compactness
						Bridson_Common.semanticSegmentation = cnn
						random.seed(Bridson_Common.seedValue)
						np.random.seed(Bridson_Common.seedValue)
						try:
							indexValidation(filename)
							if Bridson_Common.bulkGeneration:
								plt.close("all")
								gc.collect()
							# sys.stdout.close()
						except Exception as e:
							print("Exception calling indexValidation:", e)
							print("Error details:", sys.exc_info()[0])
							exc_info = sys.exc_info()
							traceback.print_exception(*exc_info)
							del exc_info

	if Bridson_Common.bulkGeneration:
		# coreCount = mp.cpu_count() - 4
		coreCount = Bridson_Common.coreCount
		'''
			Have to implement this looping mechanism.
			For some reason, if we provide the whole list to the pool, the pool eventually hangs or all the chidren crash.
		'''
		blocks = coreCount*2
		iterations = int(len(variables) / blocks + 1)

		print("CoreCount:", coreCount)
		mp.set_start_method('forkserver')  # Valid options are 'fork', 'spawn', and 'forkserver'
		for iteration in range( iterations ):
			iterationVariables = variables[iteration*blocks:(iteration+1)*blocks].copy()
			print(iterationVariables)
			with Pool(processes=coreCount) as pool:
				pool.starmap( wrapper, iterationVariables )
			pool.terminate()

	print("*************************************************")
	print("Finished")
	print("*************************************************")

	if Bridson_Common.bulkGeneration:
		print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
		print(">>>>>>>>>>>>>>>>>>>>>>> Main Exit <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
		print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
		# exit(0)
		os.system('rm *_chaincode.txt')
		sys.exit(0)
	else:
		plt.show()


