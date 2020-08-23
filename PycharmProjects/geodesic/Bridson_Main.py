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
import Bridson_FinishedImage
from skimage.segmentation import mark_boundaries

random.seed(Bridson_Common.seedValue)
np.random.seed(Bridson_Common.seedValue)
# Bridson_Common.logDebug(__name__, "Seed was:", Bridson_Common.seedValue)


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
	path = "../../boundary-first-flattening/build/"
	# Create object file for image.
	Bridson_createOBJFile.createObjFile2D(path, "test1.obj", samples, tri, radius, center, distance=euclidean_distance)




def BFFReshape():
	Bridson_Common.logDebug(__name__, "Reshaping with BFF")
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
	Bridson_Common.logDebug(__name__, "Extracting 2D image post BFF Reshaping")
	os.system(path + "extract.py test1_out.obj test1_out_flat.obj")


def cleanUpFiles():
	path = "../../boundary-first-flattening/build/"
	os.system("rm " + path + "test1.obj ")
	os.system("rm "+ path + "test1_out.obj")
	os.system("rm " + path + "test1_out_flat.obj")




def SLICImage(filename):
	startIndex = 0 # Index starts at 0.
	regionIndex = startIndex
	imageraster, regionMap, segments, regionIntensityMap = SLIC.callSLIC(filename)


	stopIndex=startIndex+16

	fig = plt.figure()
	ax3 = fig.add_subplot(1, 1, 1, aspect=1, label='Image regions')
	plt.title('Image regions - ' + filename + ' - Segments: ' + str(Bridson_Common.segmentCount) + ' - regionPixels: ' + str(Bridson_Common.targetRegionPixelCount))
	''' Draw Letter blob '''

	# blankRaster = np.zeros(np.shape(imageraster))
	# ax3 = plt.subplot2grid(gridsize, (0, 1), rowspan=1)
	# ax3.imshow(blankRaster)
	# ax3.imshow( imageraster, cmap='Greys', norm=matplotlib.colors.Normalize())
	# ax3.imshow( imageraster, cmap='Greys' )

	regionRaster = imageraster / np.max(imageraster)   # Need to normalize the region intensity [0 ... 1.0] to display properly.
	# print("Raster:", displayRaster)
	ax3.imshow(mark_boundaries( regionRaster, segments, color=(255,0,0) ))
	ax3.grid()

	Bridson_Common.saveImage(filename, "GreyscaleSLIC", fig)

	thismanager = pylab.get_current_fig_manager()
	thismanager.window.wm_geometry("+0+0")

	Bridson_Common.logDebug(__name__, "SLIC Keys:" + str(regionMap.keys()) )
	return imageraster, regionMap, regionRaster, segments, regionIntensityMap


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

	while True:
		attempts += 1
		if successful or attempts > 10:
			break
		else:
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
			Bridson_Common.logDebug(__name__, "*** BlurRadius: " , blurRadius)

			mask5x = Bridson_Common.blurArray(mask5x, blurRadius)
			mask5x = Bridson_CreateMask.InvertMask(mask5x)



			# print(distanceRaster)
			if Bridson_Common.debug:
				plt.figure()
				plt.subplot(1, 1, 1, aspect=1)
				plt.title('blurred Mask')
				plt.imshow(mask5x)
				thismanager = pylab.get_current_fig_manager()
				thismanager.window.wm_geometry("+0+560")


			meshObj = Bridson_MeshObj.MeshObject(mask=mask5x, dradius=dradius, indexLabel=indexLabel) # Create Mesh based on Mask.
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
			flatMeshObj = Bridson_MeshObj.MeshObject(flatvertices=flatvertices, flatfaces=flatfaces, xrange=xrange, yrange=yrange, indexLabel=indexLabel) # Create Mesh based on OBJ file.
			successful = flatMeshObj.trifinderGenerated
			if successful:
				print("Attempt ", attempts, " successful")
			else:
				print("Attempt ", attempts, " UNsuccessful")

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



def indexValidation(filename):
	imageraster, regionMap, regionRaster, segments, regionIntensityMap = SLICImage(filename)
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
	finishedImageNoSLIC.setTitle(filename)
	finishedImageNoSLIC.setXLimit(0, np.shape(regionRaster)[1])
	finishedImageNoSLIC.setYLimit(0, -np.shape(regionRaster)[0])

	maskRasterCollection = {}
	meshObjCollection = {}
	NoSLICmaskRasterCollection = {}
	NoSLICmeshObjCollection = {}

	# for index in range(10,15):  # Interesting regions: 11, 12, 14
	for index in [7,8]:
	# for index in range( len(regionMap.keys()) ):
		print("(***************** ", filename, " Starting Region: ", index, "of", Bridson_Common.segmentCount, "  *************************" )

		# Generate the raster for the first region.
		raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
		# print("Raster:", raster)
		maskRasterCollection[index] = raster.copy()  # Make a copy of the mask raster.
		NoSLICmaskRasterCollection[ index ] = raster.copy()

		displayRegionRaster(raster, index)

		# Bridson_Common.logDebug(__name__, raster)
		# Bridson_Common.arrayInformation( raster )
		for i in range(1):
			indexLabel = index + i / 10
			Bridson_Common.writeMask(raster)
			meshObj, flatMeshObj, trifindersuccess = processMask(raster, dradius, indexLabel)

			if trifindersuccess:
				successfulRegions += 1
				print("Trifinder was successfully generated for region ", index)
				if Bridson_Common.diagnostic == False:
					# Only draw the lines if the trifinder was successful generated.
					if Bridson_Common.linesOnFlat:
						if Bridson_Common.verticalLines:
							# flatMeshObj.DrawVerticalLines()
							# flatMeshObj.DrawVerticalLinesSeededFrom(LineSeedPointsObj, meshObj) # Draw lines based on seed from tertiary mesh.
							# flatMeshObj.DrawVerticalLinesExteriorSeed2() # Draw lines using exterior points as line seed.
							flatMeshObj.DrawAngleLinesExteriorSeed2(angle=0)
							# flatMeshObj.DrawAngleLinesExteriorSeed2(angle=90)
						else:
							# flatMeshObj.DrawHorizontalLines()
							# flatMeshObj.DrawHorizontalLinesExteriorSeed() # Draw lines using exterior points as line seed.
							flatMeshObj.DrawAngleLinesExteriorSeed2()
						# Transfer the lines from the FlatMesh to meshObj.
						meshObj.TransferLinePointsFromTarget(flatMeshObj)
					else:
						if Bridson_Common.verticalLines:
							# meshObj.DrawVerticalLines()
							# meshObj.DrawVerticalLinesSeededFrom(LineSeedPointsObj, meshObj) # Draw lines based on seed from tertiary mesh.
							# meshObj.DrawVerticalLinesExteriorSeed() # Draw lines using exterior points as line seed.
							meshObj.DrawAngleLinesExteriorSeed2()
						else:
							# meshObj.DrawHorizontalLines()
							# meshObj.DrawHorizontalLinesExteriorSeed() # Draw lines using exterior points as line seed.
							meshObj.DrawAngleLinesExteriorSeed2()

						flatMeshObj.TransferLinePointsFromTarget(meshObj)
				if True:  # Rotate original image 90 CW.
					meshObj.rotateClockwise90()

				# At this point, we have transferred the lines from the flattened mesh to the original mesh.
				meshObjCollection[ index ] = meshObj
				NoSLICmeshObjCollection[ index ] = meshObj
			else:
				print("Trifinder was NOT successfully generated for region ", index)


	# meshObj.diagnosticExterior()
			# flatMeshObj.diagnosticExterior()

		# finishedImage.drawRegionContourLines(regionMap, index, meshObj)


	# At this point, we need can attempt to merge the lines between each region.
	# Still have a problem with the coordinates though.
	finishedImageSLIC.mergeLines(regionMap, regionRaster, maskRasterCollection, meshObjCollection, regionIntensityMap)
	finishedImageNoSLIC.setCollections( maskRasterCollection, meshObjCollection , finishedImageSLIC.regionConnectivity, finishedImageSLIC.distanceRasters)

	for index in meshObjCollection.keys():
		# Draw the region contour lines onto the finished image.
		meshObj = meshObjCollection[ index ]
		finishedImageSLIC.drawRegionContourLines(regionMap, index, meshObj, regionIntensityMap[index], drawSLICRegions=True )
		finishedImageNoSLIC.drawRegionContourLines(regionMap, index, meshObj, regionIntensityMap[index], drawSLICRegions=False )
		print("Done drawing contour lines")


	Bridson_Common.saveImage( filename, "WithSLIC", finishedImageSLIC.fig )
	Bridson_Common.saveImage(filename, "NoSLIC", finishedImageNoSLIC.fig)

	print("Successful Regions: ", successfulRegions)
	print("Total Regions: ", len(regionMap.keys()) )


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


if __name__ == '__main__':
	cleanUpFiles()
	dradius = Bridson_Common.dradius # 3 seems to be the maximum value.
	xrange, yrange = 10, 10

	# mask = Bridson_CreateMask.CreateCircleMask(xrange, yrange, 10)

	if False:
		mask = Bridson_CreateMask.genLetter(xrange, yrange, character='Y')
		meshObj, flatMeshObj = processMask(mask, dradius, 0)
		# flatMeshObj.DrawVerticalLines()
		# meshObj.DrawVerticalLines()
		# Transfer the lines from the FlatMesh to meshObj.
		# meshObj.TransferLinePointsFromTarget( flatMeshObj )

	# meshObj = Bridson_MeshObj.MeshObject(mask=mask, dradius=dradius)


	'''
	# Original code for generating the Mesh with Mask.
		count, chain, chainDirection, border = Bridson_ChainCode.generateChainCode(mask, rotate=False)
		border = Bridson_ChainCode.generateBorder(border, dradius)

		invertedMask =  Bridson_CreateMask.InvertMask( mask )

		# Bridson_Common.logDebug(__name__, invertedMask)
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

	# imageraster, regionMap = SLICImage()
	regionMap = []

	if False:
		startIndex = 0
		stopIndex = 2
		for regionIndex in range(startIndex, stopIndex):
		# for regionIndex in regionMap.keys():
			Bridson_Common.logDebug(__name__, ">>>>>>>>>>> Start of cycle")
			# resetVariables()
			# try:
			Bridson_Common.logDebug(__name__, "Generating index:", regionIndex)
			raster, actualTopLeft = SLIC.createRegionRasters(regionMap, regionIndex)
			displayRegionRaster( raster[:], regionIndex )

	if False:
			for index in range(1,2):
				# Generate the raster for the first region.
				raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
				# Bridson_Common.logDebug(__name__, raster)
				# Bridson_Common.arrayInformation( raster )
				for i in range(1):
					Bridson_Common.writeMask(raster)
					meshObj, flatMeshObj = processMask(raster, dradius, index + i / 10)
					flatMeshObj.DrawVerticalLines()

				# Transfer the lines from the FlatMesh to meshObj.
				meshObj.TransferLinePointsFromTarget( flatMeshObj )

	images = []
	# images.append('SimpleR.png')
	# images.append('SimpleC.png')
	# images.append('simpleTriangle.png')
	# images.append('simpleHorizon.png')
	# images.append('FourSquares.png')
	# # images.append('SimpleSquare.jpg')
	# images.append("FourCircles.png")
	# images.append('Stripes.png')

	# Batch A.
	# images.append('RedApple.jpg')
	# images.append('Sunglasses.jpg')
	# images.append('TapeRolls.jpg')
	# images.append('Massager.jpg')
	images.append('eyeball.jpg')
	# images.append('truck.jpg')
	# images.append('cat1.jpg')

	# Original
	# images.append('dog2.jpg')

	# percentages = [0.05, 0.1, 0.15, 0.2]
	# targetPixels = [  400, 800, 1600 ]
	targetPixels = [800]
	for filename in images:
		for targetPixel in targetPixels:
			Bridson_Common.targetRegionPixelCount = targetPixel
			# Set the seed each time.
			random.seed(Bridson_Common.seedValue)
			np.random.seed(Bridson_Common.seedValue)
			indexValidation(filename)

	Bridson_Common.logDebug(__name__, "------------------------------------------")
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
		for index in range(5,6):
			# Generate the raster for the first region.
			# raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
			# Bridson_Common.logDebug(__name__, raster)
			# Bridson_Common.arrayInformation( raster )
			# Bridson_Common.writeMask(raster)
			raster = Bridson_Common.readMask(filename='BlurTriangle.gif')
			Bridson_Common.logDebug(__name__, "Minimum value of Raster: " , np.min(raster))
			Bridson_Common.logDebug(__name__, "Maximum value of Raster: ", np.max(raster))
			raster = Bridson_CreateMask.InvertMask(raster)
			Bridson_Common.logDebug(__name__, "Minimum value of Inverted Raster: ", np.min(raster))
			Bridson_Common.logDebug(__name__, "Maximum value of Inverted Raster: ", np.max(raster))
			featureRemoval(raster, dradius, index)
			# flatMeshObj.DrawVerticalLines()

			# Transfer the lines from the FlatMesh to meshObj.
			# meshObj.TransferLinePointsFromTarget( flatMeshObj )


	if False:
		for index in range(1,2):
			# Generate the raster for the first region.
			raster, actualTopLeft = SLIC.createRegionRasters(regionMap, index)
			# raster = SLIC.createRegionRasters(regionMap, index)
			# Bridson_Common.logDebug(__name__, raster)
			# Bridson_Common.arrayInformation( raster )
			# Bridson_Common.writeMask(raster)
			featureRemoval(raster, dradius, index)
			# flatMeshObj.DrawVerticalLines()

			# Transfer the lines from the FlatMesh to meshObj.
			# meshObj.TransferLinePointsFromTarget( flatMeshObj )


	if Bridson_Common.bulkGeneration:
		exit(0)
	else:
		plt.show()


