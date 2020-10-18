# import the necessary packages
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import numpy as np
from PIL import Image
import Bridson_Common
from skimage import exposure, filters
import cv2 as cv
import Bridson_ImageModify
import SemanticSegmentation
import os
import Bridson_ColourOperations



from ChainCodeGenerator import generateChainCode, writeChainCodeFile

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

def saveImage(filename, postFix, image):
	if os.path.exists("./output") == True:
		if os.path.isdir("./output") == False:
			exit(-1)
	else:
		os.mkdir("./output")

	try:
		# Save the figures to files: https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
		actualFileName = "./output/" + filename + "_segments_" + str(Bridson_Common.segmentCount)
		if Bridson_Common.SLIC0:
			actualFileName = actualFileName + "_compactness_SLIC0"
		else:
			actualFileName = actualFileName + "_compactness_" + str(Bridson_Common.compactnessSLIC)
		actualFileName = actualFileName + "_cnn_" + Bridson_Common.semanticSegmentation + "_semanticRatio_" + str(Bridson_Common.semanticSegmentationRatio) + "_" + postFix + ".png"
		cv.imwrite(actualFileName, image)
	except Exception as e:
		print("Error saving file:", e)

def segmentImage(imageName, numSegments):
	# print("SLIC Generating segments:", numSegments)
	# load the image and convert it to a floating point data type
	# image = img_as_float(io.imread(imageName))
	originalImage = cv.imread( imageName )
	print("EE")
	postFix = ''

	# if Bridson_Common.semanticSegmentation == 'mask_rcnn':
	# 	cnn = SemanticSegmentation.Mask_RCNN()
	# 	image = cnn.processImage(imageName)
	# 	saveImage(imageName, 'MASKRCNN_SEMANTIC', image)
	# 	postFix = postFix + 'MASKRCNN_'
	# 	# image = np.asarray(image)
	# 	del cnn
	# elif Bridson_Common.semanticSegmentation == 'deeplabv3':
	# 	cnn = SemanticSegmentation.Deeplabv3()
	# 	image = cnn.processImage(imageName)
	# 	saveImage(imageName, 'DEEPLABV3_SEMANTIC', image)
	# 	postFix = postFix + 'DEEPLABV3_'
	# 	# image = np.asarray(image)
	# 	del cnn
	# elif Bridson_Common.semanticSegmentation == 'both':
	# 	cnn = SemanticSegmentation.Mask_RCNN()
	# 	image1 = cnn.processImage(imageName)
	# 	del cnn
	# 	cnn = SemanticSegmentation.Deeplabv3()
	# 	image2 = cnn.processImage(imageName)
	# 	del cnn
	# 	image = cv.addWeighted(image1, 0.5, image2, 0.5, 0.0)
	# 	saveImage(imageName, 'BOTH_SEMANTIC', image)
	# 	postFix = postFix + 'BOTH_'
	# else:
		# Bridson_Common.semanticSegmentation == 'none':
		# image = io.imread(imageName)
	image = cv.imread( imageName )
	print("EE1")
	# Alpha mix the the
	imageRatio = 1 - Bridson_Common.semanticSegmentationRatio

	image = cv.addWeighted(originalImage, imageRatio, image, Bridson_Common.semanticSegmentationRatio, 0.0)
	print("AA Max and Min of greyscaleImage:", np.amax(image), np.amin(image))
	image = image / np.amax(image)
	image = image * 255
	image = np.asarray(image, dtype=np.uint8)
	print("BB Max and Min of greyscaleImage:", np.amax(image), np.amin(image))
	print("EE2")
	postFix = postFix + 'OVERLAY'
	saveImage(imageName, postFix, image)
	if Bridson_Common.bulkGeneration == False:
		plt.figure()
		plt.title(imageName + 'Semantic Segmentation')
		plt.imshow(image)
	print("EE2")
	if Bridson_Common.increaseContrast:
		image = np.asarray( Bridson_ImageModify.increaseContrast( Image.fromarray(image),  Bridson_Common.contrastFactor ) )

	if Bridson_Common.SLICGrey:
		# image = cv.cvtColor( Image.fromarray(image), cv.COLOR_RGB2GRAY)
		# image = np.asarray( image )
		image = np.asarray( Image.fromarray(image).convert('L') )
	# imageArr = np.asarray( image )
	# image = np.copy(image)
	print("EE3")
	print("Shape:",np.shape(image))
	print("Type:",type(image))
	print("EE4")
	if Bridson_Common.Median:
		# try:
		image = filters.median( image )
	print("CC Max and Min of greyscaleImage:", np.amax(image), np.amin(image))
		# except Exception as e:
		# 	print(np.shape(image))
		# 	print(type(image))
		# 	print(">>>>>>>>>>>>>> Error converting to Median:", e)
	# print("Image:", image)
	# loop over the number of segments
	# for numSegments in (100, 200, 300):
	# for numSegments in (100,):
		# apply SLIC and extract (approximately) the supplied number
		# of segments
	# segments = slic(image, n_segments=numSegments, sigma=5, compactness=11, slic_zero=False, enforce_connectivity=True)

	if Bridson_Common.EqualizeHistogram:
		# try:
		image = exposure.equalize_adapthist(image)
		# except Exception as e:
		# 	print(np.shape(image))
		# 	print(type(image))
		# 	print(">>>>>>>>>>>>> Error equalizing histogram:", e)
	print("DD Max and Min of greyscaleImage:", np.amax(image), np.amin(image))

	# if False:
	# 	image = cv.cvtColor(image, cv.COLOR_RGB2Lab)
	#
	# 	plt.figure()
	# 	plt.imshow(image)
	# 	plt.title('CIELab image')


	if Bridson_Common.SLIC0 == True:
		segments = slic(image, n_segments=numSegments, sigma=3, compactness=Bridson_Common.compactnessSLIC, slic_zero=True, enforce_connectivity=True, max_iter=Bridson_Common.SLICIterations)
	else:
		segments = slic(image, n_segments=numSegments, sigma=3, compactness=Bridson_Common.compactnessSLIC,  enforce_connectivity=True, max_iter=Bridson_Common.SLICIterations)
	print("EE Max and Min of greyscaleImage:", np.amax(image), np.amin(image))
	# print("SLIC:segmentImage setting segment Count")
	# Bridson_Common.segmentCount = len(segments)
	# Bridson_Common.logDebug(__name__, type(segments))
	# regionIndex = 16
	# # show the output of SLIC
	# fig = plt.figure("Superpixels -- %d segments - file%s" % (numSegments, imageName))
	# ax = fig.add_subplot(3, 3, 1)
	# ax.imshow(mark_boundaries(image, segments, color=(1,0,0), mode='inner', background_label=regionIndex))
	# Bridson_Common.logDebug(__name__, "shape: ", np.shape(segments))
	# # Bridson_Common.logDebug(__name__, segments)
	# # plt.axis("off")
	#
	# raster, regionMap = catalogRegions(segments)
	# # fig = plt.figure("Raster --")
	# ax = fig.add_subplot(3, 3,  2)
	# regionImage = Image.fromarray(np.uint8(raster), 'L')
	# ax.imshow(raster)
	# # ax.imshow(mark_boundaries(regionImage, segments, color=(1,0,0), mode='inner', background_label=regionIndex ))
	#
	# baseAxis = 3
	# for i in range(7):
	# 	regionRaster = createRegionRasters(regionMap, regionIndex)
	# 	ax = fig.add_subplot(3, 3,  baseAxis)
	# 	Bridson_Common.logDebug(__name__, regionRaster)
	# 	ax.imshow(regionRaster)
	# 	regionIndex += 1
	# 	baseAxis += 1


	# show the plots
	# plt.show()
	# Bridson_Common.logDebug(__name__, regionMap)
	return image, segments



# Generate image region Intensity Hashmap.




def catalogRegions( segments, regionIntensityMap ):
	'''
	Create a map of region that maps to list of (x,y) coordinates.
	Create a raster image of the individual
	:param segments: Segmentation information returned from SLIC.
	:return: Dictionary of segments.  Map will contain all the x,y coordinates for each segment.

	'''
	# print("Segments:", segments)
	x, y = np.shape(segments)
	raster = np.zeros((x,y))
	regionMap = {}
	for i in range(x):
		for j in range(y):
			regionLabel = segments[i][j]
			# print("regionLabel:", regionLabel)
			# raster[i][j] = (regionLabel * 234867)%256
			raster[i][j] = regionIntensityMap[ regionLabel ]
			region = regionMap.get(regionLabel)
			if region == None:
				regionMap[regionLabel] = [(i,j)]
			else:
				region.append((i,j))

	return raster, regionMap


def calculateTopLeft( regionCoordinates ):
	topLeft = (np.max(regionCoordinates), np.max(regionCoordinates))
	bottomRight = (np.min(regionCoordinates), np.min(regionCoordinates))
	for coord in regionCoordinates:
		# Bridson_Common.logDebug(__name__, coord)
		x, y = coord
		if x < topLeft[0]:
			topLeft = (x, topLeft[1])
		if y < topLeft[1]:
			topLeft = (topLeft[0], y)
		if x > bottomRight[0]:
			bottomRight = (x, bottomRight[1])
		if y > bottomRight[1]:
			bottomRight = (bottomRight[0], y)

	return topLeft, bottomRight

def createRegionRasters(regionMap, region=0):
	# For each region, create a raster.
	actualTopLeft=[0,0]
	# for regionLabel, regionCoordinates in regionMap.items():
	# 	Bridson_Common.logDebug(__name__, regionLabel)

	regionCoordinates = regionMap.get(region)

	if regionCoordinates != None:
		topLeft, bottomRight = calculateTopLeft(regionCoordinates)

		actualTopLeft[0]=topLeft[1]
		actualTopLeft[1]=topLeft[0]

		deltax = bottomRight[0] - topLeft[0] + 10
		deltay = bottomRight[1] - topLeft[1] + 10
		deltax = deltay = max(deltax,deltay) # Size of the raster region.

		# deltax, deltay = bottomRight[0] - topLeft[0]+3, bottomRight[1] - topLeft[1]+3

		shiftx, shifty = topLeft[0]-5, topLeft[1]-5 # This is the amount to shift the actual coordinates such that they fit onto the raster.
		# shiftx, shifty=topLeft[0]-1, topLeft[1]-1

		# print( 'TopLeft:', topLeft, 'BottomRight:',  bottomRight)
		# print( 'DeltaX:', deltax, 'DeltaY:', deltay)
		# print( 'Shiftx:', shiftx, 'Shifty:', shifty)
		raster = np.zeros((deltax,deltay))


		for coord in regionCoordinates:
			x, y = coord
			x = x - shiftx
			y = y - shifty
			# Bridson_Common.logDebug(__name__, 'Relative coords:', x, y)
			raster[x][y] = 255

		# print("Raster:", raster)
		# Highlight the topLeft pixel.
		# raster[topLeft[0]-shiftx-5][topLeft[1]-shifty-5] = 127
	else:
		raster = None


	return raster, actualTopLeft





def getSegmentLabels(segments):
	x, y = np.shape(segments)
	# raster = np.zeros((x,y))
	segmentLabels = {}
	for i in range(x):
		for j in range(y):
			segmentLabels[ segments[i][j] ] = 1
	return segmentLabels

def generateImageIntensityHashmap( greyImage, segments):
	# Generate the average intensity for each region.
	# Create array for the arverge intensity for each region.
	segmentLabels = getSegmentLabels( segments )

	regionIntensityMap = {}
	# Initialize the regionIntensityMap
	for regionIndex in segmentLabels:
		regionIntensityMap[ regionIndex ] = 0

	x, y = np.shape(segments)
	if Bridson_Common.intensityMapType == 'average':
		# Generate the average for region intensity.
		# raster = np.zeros((x,y))
		for i in range(x):
			for j in range(y):
				regionIntensityMap[ segments[i][j] ] += greyImage[i][j]


		# Calculate the average for each regionIndex.
		for regionIndex in segmentLabels:
			regionIntensityMap[ regionIndex ] = regionIntensityMap[ regionIndex ] / np.count_nonzero( segments == regionIndex ) # Calculate the average intensity for each region.
	else:
		# Generate the median for region intensity.
		for i in range(x):
			for j in range(y):
				regionIntensityMap[ segments[i][j] ] = []

		for i in range(x):
			for j in range(y):
				regionIntensityMap[ segments[i][j] ].append( greyImage[i][j] )

		# Calculate the median for each regionIndex.
		for regionIndex in segmentLabels:
			regionIntensityMap[ regionIndex ] = np.median( regionIntensityMap[ regionIndex ])

	return regionIntensityMap

def generateImageColourHashmap( colourImage, segments):
	# Generate the average colour for each region.  Convert the RGB to CIELab.
	# colourImage = cv.cvtColor(colourImage, cv.COLOR_RGB2Lab)  # Convert to CIELab

	# Generate the average intensity for each region.
	# Create array for the arverge intensity for each region.
	segmentLabels = getSegmentLabels( segments )

	regionColourMap = {}
	# Initialize the regionIntensityMap
	for regionIndex in segmentLabels:
		regionColourMap[ regionIndex ] = [0,0,0]

	x, y = np.shape(segments)
	# raster = np.zeros((x,y))
	for i in range(x):
		for j in range(y):
			regionColourMap[ segments[i][j] ] += colourImage[i][j]

	# Calculate the average for each regionIndex.
	for regionIndex in segmentLabels:
		regionColourMap[ regionIndex ] = ( regionColourMap[ regionIndex ] / np.count_nonzero( segments == regionIndex ) ).astype(int) # Calculate the average intensity for each region.
		# print(regionIndex, " colour ", regionColourMap[ regionIndex ])
		regionColourMap[ regionIndex ] = Bridson_ColourOperations.convertRGB_CIE( regionColourMap[ regionIndex ] )

	return regionColourMap




def calculateSegmentCount(imageFile):
	image = io.imread(imageFile)
	print("calculateSegmentCount Image shape:", np.shape(image))
	imageShape = np.shape(image)
	# Bridson_Common.targetRegionPixelCount = int((imageShape[0]*Bridson_Common.targetPercent * imageShape[1]*Bridson_Common.targetPercent))
	Bridson_Common.segmentCount = int( (imageShape[0]*imageShape[1]) / Bridson_Common.targetRegionPixelCount )
	print("calculateSegmentCount SegmentCount:", Bridson_Common.segmentCount)
	print("calculateSegmentCount Target Pixel Count:", Bridson_Common.targetRegionPixelCount)


def callSLIC(filename):
	images = []
	# images.append('dog2.jpg')
	# images.append('SimpleSquare.jpg')

	# for imageFile in images:
	# calculateSegmentCount( filename )
	segmentCount = Bridson_Common.segmentCount
	for numSegments in (segmentCount,):
		image, segments = segmentImage(filename, numSegments)
		print("DD")
		# newImage = np.copy(image)
		print("DD1")
		# greyscaleImage = np.asarray( Image.fromarray(image).convert('L') )
		# print('GreyScale:', greyscaleImage)
		# regionIndex = 16
		# show the output of SLIC
		# fig = plt.figure("Superpixels -- %d segments - file%s" % (numSegments, image))
		# ax = fig.add_subplot(3, 3, 1)
		# ax.imshow(mark_boundaries(image, segments, color=(1,0,0), mode='inner', background_label=regionIndex))
		# Bridson_Common.logDebug(__name__, "shape: ", np.shape(segments)) # HSC
		# Bridson_Common.logDebug(__name__, segments)
		# plt.axis("off")
		print("DD2")
		originalImage = cv.imread(filename)
		# originalImage = np.asarray( originalImage )
		print("DD3")
		regionColourMap = generateImageColourHashmap(originalImage, segments)
		print("DD4")
		# print("regionColourMap:", regionColourMap)

		print("DD5")
		# print("Type of image:", type(image))
		print("YY Max and Min of image:", np.amax(image), np.amin(image))
		if Bridson_Common.intensityMapUserMergeCNN:
			image = np.asarray(image, dtype=np.uint8) # Need to convert to uint8.  Image.fromarray will throw an error otherwise.
			greyscaleImage = np.asarray( Image.fromarray(image).convert('L') )
		else:
			greyscaleImage = cv.cvtColor(originalImage, cv.COLOR_BGR2GRAY)

		print("ZZ Max and Min of greyscaleImage:", np.amax(greyscaleImage), np.amin(greyscaleImage))
		# Generate region intensity HashMap.
		regionIntensityMap = generateImageIntensityHashmap(greyscaleImage, segments)

		# print("DD6")
		# print("Region Intensity Map:", regionIntensityMap)
		raster, regionMap = catalogRegions(segments, regionIntensityMap)
		# print("DD7")
		# fig = plt.figure("Raster --")
		# ax = fig.add_subplot(3, 3,  2)
		# regionImage = Image.fromarray(np.uint8(raster), 'L')
		# ax.imshow(raster)
		# ax.imshow(mark_boundaries(regionImage, segments, color=(1,0,0), mode='inner', background_label=regionIndex ))

	# raster = raster.astype(int)  # Convert the raster to integer.
	return raster, regionMap, segments, regionIntensityMap, regionColourMap




if __name__ == '__main__':
	debug = True
	images = []

	# images.append( 'raptor.jpg' )
	# image = 'raptor.jpg'
	# segmentImage(image)
	# image = 'truck.jpg'
	# images.append('eyeball.jpg')
	# image = 'eyeball.jpg'
	# images.append('eyeball.jpg')
	# image='tree.jpg'
	# images.append('tree.jpg')
	# image='cat1.jpg'
	# images.append('cat1.jpg')
	# image='cat2.jpg'
	# images.append('cat2.jpg')
	# image='cat3.jpg'
	# images.append('cat3.jpg')
	# image='cat4.jpg'
	# images.append('cat4.jpg')
	# image='dog1.jpg'
	# images.append('dog1.jpg')
	# image='dog2.jpg'
	images.append('dog2.jpg')
	# image='dog3.jpg'
	# images.append('dog3.jpg')
	for imageFile in images:

		for numSegments in (40,):
			image, segments = segmentImage(imageFile, numSegments)
			newImage = np.copy(image)
			regionIndex = 1
			if debug == True:
				# show the output of SLIC
				fig = plt.figure("Superpixels -- %d segments - file%s" % (numSegments, image))
				ax = fig.add_subplot(3, 3, 1)
				ax.imshow(mark_boundaries(image, segments, color=(1,0,0), mode='inner', background_label=regionIndex))
				# Bridson_Common.logDebug(__name__, "shape: ", np.shape(segments))
				# Bridson_Common.logDebug(__name__, segments)
				# plt.axis("off")

				raster, regionMap = catalogRegions(segments)
				# fig = plt.figure("Raster --")
				ax = fig.add_subplot(3, 3,  2)
				regionImage = Image.fromarray(np.uint8(raster), 'L')
				ax.imshow(raster)
				# ax.imshow(mark_boundaries(regionImage, segments, color=(1,0,0), mode='inner', background_label=regionIndex ))

				baseAxis = 3
				for i in range(7):
					regionRaster, actualTopLeft = createRegionRasters(regionMap, regionIndex)
					Bridson_Common.logDebug(__name__, "Actual Top Left:", actualTopLeft)
					ax = fig.add_subplot(3, 3,  baseAxis)
					# Bridson_Common.logDebug(__name__, regionRaster)
					ax.imshow(regionRaster)
					regionIndex += 1
					baseAxis += 1

			else:
				# genMeshFromRaster(segments[0])

				None


	plt.show()

