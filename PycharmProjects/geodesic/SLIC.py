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

from ChainCodeGenerator import generateChainCode, writeChainCodeFile

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

def segmentImage(imageName, numSegments):
	# load the image and convert it to a floating point data type
	image = img_as_float(io.imread(imageName))


	# loop over the number of segments
	# for numSegments in (100, 200, 300):
	# for numSegments in (100,):
		# apply SLIC and extract (approximately) the supplied number
		# of segments
	segments = slic(image, n_segments=numSegments, sigma=5, compactness=10, slic_zero=False, enforce_connectivity=False)

	# print(type(segments))
	# regionIndex = 16
	# # show the output of SLIC
	# fig = plt.figure("Superpixels -- %d segments - file%s" % (numSegments, imageName))
	# ax = fig.add_subplot(3, 3, 1)
	# ax.imshow(mark_boundaries(image, segments, color=(1,0,0), mode='inner', background_label=regionIndex))
	# print("shape: ", np.shape(segments))
	# # print(segments)
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
	# 	print(regionRaster)
	# 	ax.imshow(regionRaster)
	# 	regionIndex += 1
	# 	baseAxis += 1


	# show the plots
	# plt.show()
	# print(regionMap)
	return image, segments


def catalogRegions(segments):
	'''
	Create a map of region that maps to list of (x,y) coordinates.
	Create a raster image of the individual
	:param segments: Segmentation information returned from SLIC.
	:return: Dictionary of segments.  Map will contain all the x,y coordinates for each segment.

	'''

	x, y = np.shape(segments)
	raster = np.zeros((x,y))
	regionMap = {}
	for i in range(x):
		for j in range(y):
			regionLabel = segments[i][j]
			raster[i][j] = (regionLabel * 234867)%256
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
			raster[x][y] = 255
	else:
		raster = None


	return raster


'''
def genMeshFromRaster(raster):
	global triang, triang2, trifinder, trifinder2, ax1, ax2, Originalsamples, Flatsamples, Originalfaces, Flatfaces, polygon1, polygon2, xsize, ysize, perimeterSegments, startingR, angle, bprocess, breset, bangle, dimension, character, letterRatio, letterDimension

	generateBlob = True

	xsize, ysize = np.size(raster)

	if generateBlob:
		# letterDimension = int(dimension / letterRatio)
		letterDimension = letterDimension
		xsize = ysize = dimension

		# letter = genLetter(boxsize=letterDimension, character=character, blur=0)
		letter = raster
		count, chain, chainDirection, border = generateChainCode(letter)
		print('ChainDirection:', len(chainDirection), chainDirection)
		# writeChainCodeFile('./', 'testChainCode.txt', chainDirection)
		writeChainCodeFile('./', 'chaincode.txt', chainDirection)
		print(len(chainDirection))
		perimeterSegments = len(chainDirection)
		startingR = perimeterSegments / 10

		startTime = int(round(time.time() * 1000))
		samples = poisson_disc_samples(width=xsize, height=ysize, r=10, k=k, segments=perimeterSegments)
		# samples = poisson_disc_samples(width=xsize, height=ysize, r=4, k=k, segments=len(chainDirection))
		endTime = int(round(time.time() * 1000))
	else:
		xsize = ysize = dimension
		type = 'cRecurve3'

		if type == 'normal':
			# Attempted 'H'
			perimeterSegments = 82 # The value
			os.system('cp chaincodecopy.txt chaincode.txt')
		elif type == 'pentagram':
			# Pentagram
			perimeterSegments = 40 # The value # Pentagram.
			os.system('cp pentagram.txt chaincode.txt')
		elif type == 'star':
			# star
			perimeterSegments = 26*5 # Now works.  The angles were wrong.  Had to fix.
			os.system('cp star.txt chaincode.txt')
		elif type == 'square':
			# star
			perimeterSegments = 26*4 #
			os.system('cp square.txt chaincode.txt')
		elif type == 'square2':
			# star
			perimeterSegments = 26*4 #
			os.system('cp square2.txt chaincode.txt')
		elif type == 'square3':
			# star
			perimeterSegments = 26*4 #
			os.system('cp square3.txt chaincode.txt')
		elif type == 'c':
			# star
			perimeterSegments = 93 #
			os.system('cp c.txt chaincode.txt')
		elif type == 'cRecurve2':
			# star
			perimeterSegments = 81 #
			os.system('cp cRecurve2.txt chaincode.txt')
		elif type == 'cRecurve3':
			# star
			perimeterSegments = 80 #
			os.system('cp cRecurve3.txt chaincode.txt')

		startingR = perimeterSegments / 10
		startTime = int(round(time.time() * 1000))
		samples = poisson_disc_samples(width=xsize, height=ysize, r=20, k=k, segments=perimeterSegments)
		endTime = int(round(time.time() * 1000))
		raster = [[0 for i in range(xsize)] for j in range(ysize)]
		for coords in samples:
			x, y = coords
			xint = int(x)
			yint = int(y)
			raster[xint][yint] = int(255)
		letter = raster

	samples = np.array(samples)  # Need to convert to np array to have proper slicing.
	print("Execution time: %d ms" % (endTime - startTime))


	if not genVoronoi:
		print(raster)
		plt.imshow(raster)
	else:

		#voronoi_plot_2d(vor)
		tri = Delaunay(samples)  # Generate the triangles from the vertices.

		# Produce the mesh file.  Flatten the mesh with BFF.  Extract the 2D from BFF flattened mesh.
		path = "../../boundary-first-flattening/build/"
		# Create object file for image.
		createOBJFile.createObjFile2D(path, "test1.obj", samples, tri, radius, center, distance=euclidean_distance)
		# Reshape with BFF.
		print("Reshaping with BFF")
		os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=1 --normalizeUVs")
		# Extract the flattened version of the image.

		print("Extracting 2D image post BFF Reshaping")
		os.system(path + "extract.py test1_out.obj test1_out_flat.obj")



		# Read the OBJ file and produce the new mesh entries.
		Originalsamples, Originalfaces, Flatsamples, Flatfaces = readOBJFile.readObjFile(path, "test1_out.obj")

		# exit(1)
		#Originalfaces = list(Originalfaces)
		min_radius = .001

		# Create the grid.
		gridsize = (3,2)
		fig = plt.figure(figsize=(12,8))

		# Draw Letter blob
		axdraw = plt.subplot2grid(gridsize, (0,0))
		
		# letter = genLetter(boxsize=100, character='P')
		# # print(np.shape(letter))
		# print("Letter:", letter)
		axdraw.imshow(letter)

		# Buttons 
		axcut = plt.axes([0.8, 0.8, 0.1, 0.075]) # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes. [ left, bottom, width, height ]
		bprocess = Button(axcut, 'Process', color='red', hovercolor='green')
		bprocess.on_clicked(reset)

		axcut = plt.axes([0.8, 0.65, 0.1, 0.075]) # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes. [ left, bottom, width, height ]
		breset = Button(axcut, 'Reset', color='red', hovercolor='green')
		breset.on_clicked(clearDots) # Bind event for reset button.

		axcut = plt.axes([0.65, 0.65, 0.1,
		                  0.075])  # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes. [ left, bottom, width, height ]
		bangle =  Slider(axcut, 'Angle: ' + str(angle), 0, 360, valfmt='%1.2f', dragging=True)
		bangle.set_val(angle)
		bangle.on_changed(updateAngle)

		# Mesh drawings 
		ax1 = plt.subplot2grid(gridsize, (1, 0), rowspan=2)
		ax1.set_xlim([-dimension*0.2, dimension*1.2])
		ax1.set_ylim([-dimension * 0.2, dimension * 1.2])

		ax2 = plt.subplot2grid(gridsize, (1, 1), rowspan=2)
		ax2.set_xlim([-dimension*0.2, dimension*1.2])
		ax2.set_ylim([-dimension * 0.2, dimension * 1.2])

		# First subplot
		triang = Triangulation(Originalsamples[:, 0], Originalsamples[:, 1], triangles=Originalfaces)
		# ax1 = plt.subplot(121, aspect='equal') # Create first subplot.
		ax1.triplot(triang, color='grey')

		# triang.set_mask(np.hypot(Originalsamples[:, 0][triang.triangles].mean(axis=1), Originalsamples[:, 1][triang.triangles].mean(axis=1)) < min_radius)
		trifinder = triang.get_trifinder()
		polygon1 = Polygon([[0, 0], [0, 0]], facecolor='y')  # dummy data for xs,ys
		update_polygon(-1, polygon1)

		ax1.add_patch(polygon1)
		plt.gcf().canvas.mpl_connect('motion_notify_event', motion_notify)
		plt.gcf().canvas.mpl_connect('button_press_event', on_click)
		plt.gcf().canvas.mpl_connect('key_press_event', press) # Imported from DottedLine

		# plt.gcf().canvas.mpl_connect('button_press_event', motion_notify1) # https://matplotlib.org/3.1.1/users/event_handling.html

		# Second subplot
		# print(Flatfaces)
		Flatsamples = Flatsamples * dimension
		triang2 = Triangulation(Flatsamples[:, 0], Flatsamples[:, 1], triangles=Flatfaces)
		# ax2 = plt.subplot(122, aspect='equal')  # Create first subplot.
		ax2.triplot(triang2, color='grey')

		triang2.set_mask(np.hypot(Flatsamples[:, 0][triang2.triangles].mean(axis=1),
		                          Flatsamples[:, 1][triang2.triangles].mean(axis=1)) < min_radius)

		
		#	This block, we are trying to find the normals of the triangles.  Not successful yet.
		
		# tempPoints = np.vstack([Flatsamples[:, 0], Flatsamples[:, 1]]).T
		# tempTri = Delaunay(tempPoints)
		# np.unique(tempTri.simplices.ravel())
		# print('*** CoPlanar:', tempTri.coplanar)

		if False:
			trifinder2 = triang2.get_trifinder()
			polygon2 = Polygon([[0, 0], [0, 0]], facecolor='y')  # dummy data for xs,ys
			update_polygon2(-1, polygon2)
			ax2.add_patch(polygon2)


	#plt.imshow(raster)
	plt.gray()
	# Paint()
	# plt.show()
	return plt

'''



def callSLIC():
	images = []
	images.append('dog2.jpg')

	for imageFile in images:

		for numSegments in (100,):
			image, segments = segmentImage(imageFile, numSegments)
			newImage = np.copy(image)
			regionIndex = 16
			# show the output of SLIC
			# fig = plt.figure("Superpixels -- %d segments - file%s" % (numSegments, image))
			# ax = fig.add_subplot(3, 3, 1)
			# ax.imshow(mark_boundaries(image, segments, color=(1,0,0), mode='inner', background_label=regionIndex))
			print("shape: ", np.shape(segments))
			# print(segments)
			# plt.axis("off")

			raster, regionMap = catalogRegions(segments)
			# fig = plt.figure("Raster --")
			# ax = fig.add_subplot(3, 3,  2)
			# regionImage = Image.fromarray(np.uint8(raster), 'L')
			# ax.imshow(raster)
			# ax.imshow(mark_boundaries(regionImage, segments, color=(1,0,0), mode='inner', background_label=regionIndex ))

	return raster, regionMap




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

		for numSegments in (100,):
			image, segments = segmentImage(imageFile, numSegments)
			newImage = np.copy(image)
			regionIndex = 16
			if debug == True:
				# show the output of SLIC
				fig = plt.figure("Superpixels -- %d segments - file%s" % (numSegments, image))
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
				# ax.imshow(mark_boundaries(regionImage, segments, color=(1,0,0), mode='inner', background_label=regionIndex ))



				baseAxis = 3
				for i in range(7):
					regionRaster = createRegionRasters(regionMap, regionIndex)
					ax = fig.add_subplot(3, 3,  baseAxis)
					print(regionRaster)
					ax.imshow(regionRaster)
					regionIndex += 1
					baseAxis += 1

			else:
				# genMeshFromRaster(segments[0])

				None


	plt.show()