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

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

def segmentImage(imageName):
	# load the image and convert it to a floating point data type
	image = img_as_float(io.imread(imageName))



	# loop over the number of segments
	# for numSegments in (100, 200, 300):
	for numSegments in (100,):
		# apply SLIC and extract (approximately) the supplied number
		# of segments
		segments = slic(image, n_segments=numSegments, sigma=5, compactness=10, slic_zero=False, enforce_connectivity=False)

		regionIndex = 22
		# show the output of SLIC
		fig = plt.figure("Superpixels -- %d segments - file%s" % (numSegments, imageName))
		ax = fig.add_subplot(3, 3, 1)
		ax.imshow(mark_boundaries(image, segments, color=(1,0,0), mode='inner', background_label=regionIndex))
		print("shape: ", np.shape(segments))
		# print(segments)
		# plt.axis("off")

		raster, regionMap = catalogRegions(segments)
		# fig = plt.figure("Raster --")
		ax = fig.add_subplot(3, 3,  2)
		regionImage = Image.fromarray(np.uint8(raster), 'L')
		ax.imshow(raster)
		ax.imshow(mark_boundaries(regionImage, segments, color=(1,0,0), mode='inner', background_label=regionIndex ))

		baseAxis = 3
		for i in range(7):
			regionRaster = createRegionRasters(regionMap, regionIndex)
			ax = fig.add_subplot(3, 3,  baseAxis)
			print(regionRaster)
			ax.imshow(regionRaster)
			regionIndex += 1
			baseAxis += 1




	# show the plots
	# plt.show()
	print(regionMap)
	return (segments)


def catalogRegions(segments):
	'''

	:param segments: Segmentation information returned from SLIC.
	:return: Dictionary of segments.  Map will contain all the x,y coordinates for each segment.
	'''

	x, y = np.shape(segments)
	raster = np.zeros((x,y))
	regionMap = {}
	for i in range(x):
		for j in range(y):
			regionLabel = segments[i][j]
			raster[i][j] = regionLabel
			region = regionMap.get(regionLabel)
			if region == None:
				regionMap[regionLabel] = [(i,j)]
			else:
				region.append((i,j))

	return raster, regionMap


def createRegionRasters(regionMap, region=0):
	# For each region, create a raster.

	# for regionLabel, regionCoordinates in regionMap.items():
	# 	print(regionLabel)

	regionCoordinates = regionMap.get(region)

	if regionCoordinates != None:
		topLeft = (np.max(regionCoordinates), np.max(regionCoordinates))
		bottomRight = (np.min(regionCoordinates), np.min(regionCoordinates))
		for coord in regionCoordinates:
			print(coord)
			x, y = coord
			if x < topLeft[0]:
				topLeft = (x,topLeft[1])
			if y < topLeft[1]:
				topLeft = (topLeft[0], y)
			if x > bottomRight[0]:
				bottomRight = (x, bottomRight[1])
			if y > bottomRight[1]:
				bottomRight = (bottomRight[0],y)

		deltax = bottomRight[0] - topLeft[0] + 10
		deltay = bottomRight[1] - topLeft[1] + 10
		shiftx, shifty = topLeft[0]-5, topLeft[1]-5

		print('TopLeft:', topLeft, 'BottomRight:',  bottomRight)
		print('DeltaX:', deltax, 'DeltaY:', deltay)
		raster = np.zeros((deltax,deltay))


		for coord in regionCoordinates:
			x, y = coord
			x = x - shiftx
			y = y - shifty
			print('Relative coords:', x, y)
			raster[x][y] = 1
	else:
		raster = None


	return raster






if __name__ == '__main__':
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
	for image in images:
		segments = segmentImage(image)
		# raster = segmentCheck(segments)



	plt.show()