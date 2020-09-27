from random import random
from math import cos, sin, floor, sqrt, pi, ceil, fabs
import time
import matplotlib
matplotlib.use('tkagg')
from driver import genCircleCoords
import numpy as np

import matplotlib.pyplot as plt # For displaying array as image

from scipy.spatial import Voronoi, voronoi_plot_2d # For generating Voronoi graphs - https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.spatial.Voronoi.html
from scipy.spatial import Delaunay # For generating Delaunay - https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.html

import createOBJFile
import readOBJFile
import os

from matplotlib.tri import Triangulation # https://matplotlib.org/3.1.1/gallery/event_handling/trifinder_event_demo.html
from matplotlib.patches import Polygon # https://matplotlib.org/3.1.1/gallery/event_handling/trifinder_event_demo.html

import numpy.linalg as la # https://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points

from DottedLine import press, handleDottedLine, handleMove, createDottedLine

from ObjectBlob import Paint

from matplotlib.widgets import Button, Slider # https://matplotlib.org/3.1.1/gallery/widgets/buttons.html

from FilledLetter import genLetter, genSquareRaster

from ChainCodeGenerator import generateChainCode, writeChainCodeFile

from FindMeshBoundary import generateBoundaryPoints, findTopBottom, findLeftRight
from circularLines import generateCircularPoints

from SLIC import callSLIC, createRegionRasters
import Bridson_Common

def euclidean_distance(a, b):
	dx = a[0] - b[0]
	dy = a[1] - b[1]
	return sqrt(dx * dx + dy * dy)


def poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=random, segments=5):
	global radius, center, dynamic_ratio

	centerx = xsize / 2
	centery = ysize / 2
	center = (centerx, centery)
	radius = xsize / 2

	def insert_coords(p, grid):
		grid_x, grid_y = grid_coords(p)
		grid[grid_x + grid_y * grid_width] = p

	def grid_coords(p):
		return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

	def fits(p, gx, gy):
		dist = distance(p, center)
		if useDynamicRatio:
			newp=[gx, gy]
			gdist = distance(newp,center)


		if dist >= radius: # Case if point is outside of the circle.
			return False

		if useDynamicRatio:
			if gdist >= radius:
				return False

			oriHeight = sqrt(radius * radius - dist * dist)
			newHeight = sqrt(radius*radius - gdist*gdist)

			deltax = dist-gdist
			deltaHeight = newHeight - oriHeight
			slope = sqrt(deltax*deltax + deltaHeight*deltaHeight)
			dynamic_ratio = fabs(deltax / slope)
			#print(dynamic_ratio)

		yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
		for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
			for y in yrange:
				#print(p, x, y, grid_width)
				g = grid[x + y * grid_width]
				if g is None:
					continue
				if useDynamicRatio:
					if distance(p, g) <= r*dynamic_ratio:
						return False
				else:
					if distance(p, g) <= r:
						return False
		return True




	print("Segments: ", segments)
	# Generate the list of circle Perimeters.
	circlePerimeter = genCircleCoords(width, height, center, radius, segments=segments )
	print("Circle Perimeter:", circlePerimeter)
	print("Length of Circle Perimeter:", len(circlePerimeter))

	# Find the actual r
	'''
		This is a very interesting value.  If we use a static value, it can make for a very dense grid.
		However, if we use the actual distance between the perimeter points, it is inversely proportional to the number of segments.
	'''
	r = xsize
	for i in range(len(circlePerimeter)-1):
		r = min(distance(circlePerimeter[i], circlePerimeter[i+1]), r)
	r = int(min(distance(circlePerimeter[0], circlePerimeter[-1]), r))
	print("Segment based r: ", r)

	if useSegmentRadius:
		r = int(min(distance(circlePerimeter[0], circlePerimeter[-1]), r))
	else:
		r = r * rRatio

	if r == 0: # Kludge for radius.
		print("Kludging r to 1")
		r = 1

	print("Actual r: ", r)


	tau = 2 * pi
	cellsize = r / sqrt(2)

	print("CellSize:", cellsize)
	grid_width = int(ceil(width / cellsize))
	grid_height = int(ceil(height / cellsize))
	#if grid == None:
	grid = [None] * (grid_width * grid_height)

	for coords in circlePerimeter:
		insert_coords(coords, grid)

	'''
		The algorithm assumes the canvas is blank and needs to always insert a single value.  
		We trick the algorithm by inserting the last perimeter coordinate as the first value.
		The other perimeter values have already been inserted into the grid.
		
		Because the algorithm assumes that the canvas is blank, it ALWAYS retains the first p.
	'''
	if not forceCenter:
		# remove last item.
		grid = grid[:-1]
		p = circlePerimeter[-1]
	else:
		p = [centerx, centery]
	#p = width * random(), height * random()
	#queue = [p]
	queue = [p]
	grid_x, grid_y = grid_coords(p)
	grid[grid_x + grid_y * grid_width] = p

	while queue:
		qi = int(random() * len(queue))
		qx, qy = queue[qi]
		queue[qi] = queue[-1]
		queue.pop()
		for _ in range(k):
			alpha = tau * random()
			if useDynamicRatio:
				d = dynamic_ratio * r * sqrt(3 * random() + 1)  # Make sure we use dynamic ratio.
			else:
				d = r * sqrt(3 * random() + 1)
			px = qx + d * cos(alpha)
			py = qy + d * sin(alpha)
			if not (0 <= px < width and 0 <= py < height):
				continue
			p = [px, py]
			grid_x, grid_y = grid_coords(p)
			if not fits(p, grid_x, grid_y):
				continue
			queue.append(p)
			grid[grid_x + grid_y * grid_width] = p
	#print(grid)
	return [p for p in grid if p is not None]


def poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=random, segments=5):
	global radius, center, dynamic_ratio

	centerx = xsize / 2
	centery = ysize / 2
	center = (centerx, centery)
	radius = xsize / 2

	def insert_coords(p, grid):
		grid_x, grid_y = grid_coords(p)
		grid[grid_x + grid_y * grid_width] = p

	def grid_coords(p):
		return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

	def fits(p, gx, gy):
		dist = distance(p, center)
		if useDynamicRatio:
			newp = [gx, gy]
			gdist = distance(newp, center)

		if dist >= radius:  # Case if point is outside of the circle.
			return False

		if useDynamicRatio:
			if gdist >= radius:
				return False

			oriHeight = sqrt(radius * radius - dist * dist)
			newHeight = sqrt(radius * radius - gdist * gdist)

			deltax = dist - gdist
			deltaHeight = newHeight - oriHeight
			slope = sqrt(deltax * deltax + deltaHeight * deltaHeight)
			dynamic_ratio = fabs(deltax / slope)
		# print(dynamic_ratio)

		yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
		for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
			for y in yrange:
				# print(p, x, y, grid_width)
				g = grid[x + y * grid_width]
				if g is None:
					continue
				if useDynamicRatio:
					if distance(p, g) <= r * dynamic_ratio:
						return False
				else:
					if distance(p, g) <= r:
						return False
		return True

	print("Segments: ", segments)
	# Generate the list of circle Perimeters.
	circlePerimeter = genCircleCoords(width, height, center, radius, segments=segments)
	print("Circle Perimeter:", circlePerimeter)
	print("Length of Circle Perimeter:", len(circlePerimeter))

	# Find the actual r
	'''
		This is a very interesting value.  If we use a static value, it can make for a very dense grid.
		However, if we use the actual distance between the perimeter points, it is inversely proportional to the number of segments.
	'''
	r = xsize
	for i in range(len(circlePerimeter) - 1):
		r = min(distance(circlePerimeter[i], circlePerimeter[i + 1]), r)
	r = int(min(distance(circlePerimeter[0], circlePerimeter[-1]), r))
	print("Segment based r: ", r)

	if useSegmentRadius:
		r = int(min(distance(circlePerimeter[0], circlePerimeter[-1]), r))
	else:
		r = r * rRatio

	if r == 0:  # Kludge for radius.
		print("Kludging r to 1")
		r = 1

	print("Actual r: ", r)

	tau = 2 * pi
	cellsize = r / sqrt(2)

	print("CellSize:", cellsize)
	grid_width = int(ceil(width / cellsize))
	grid_height = int(ceil(height / cellsize))
	# if grid == None:
	grid = [None] * (grid_width * grid_height)

	for coords in circlePerimeter:
		insert_coords(coords, grid)

	'''
		The algorithm assumes the canvas is blank and needs to always insert a single value.  
		We trick the algorithm by inserting the last perimeter coordinate as the first value.
		The other perimeter values have already been inserted into the grid.

		Because the algorithm assumes that the canvas is blank, it ALWAYS retains the first p.
	'''
	if not forceCenter:
		# remove last item.
		grid = grid[:-1]
		p = circlePerimeter[-1]
	else:
		p = [centerx, centery]
	# p = width * random(), height * random()
	# queue = [p]
	queue = [p]
	grid_x, grid_y = grid_coords(p)
	grid[grid_x + grid_y * grid_width] = p

	while queue:
		qi = int(random() * len(queue))
		qx, qy = queue[qi]
		queue[qi] = queue[-1]
		queue.pop()
		for _ in range(k):
			alpha = tau * random()
			if useDynamicRatio:
				d = dynamic_ratio * r * sqrt(3 * random() + 1)  # Make sure we use dynamic ratio.
			else:
				d = r * sqrt(3 * random() + 1)
			px = qx + d * cos(alpha)
			py = qy + d * sin(alpha)
			if not (0 <= px < width and 0 <= py < height):
				continue
			p = [px, py]
			grid_x, grid_y = grid_coords(p)
			if not fits(p, grid_x, grid_y):
				continue
			queue.append(p)
			grid[grid_x + grid_y * grid_width] = p
	# print(grid)
	return [p for p in grid if p is not None]

'''
	variables
'''
useSegmentRadius = False # When set to True, will use the minimum distance between perimeter points.  False will use 60% of perimeter point distance.
forceCenter = False # When set to True, will force inject the center point onto canvas.
genVoronoi = True
useDynamicRatio = False


dimension = 100

character = 'A'
letterRatio = 4 # How much to shrink the leter when generating chaincode.
targetChainCodesSegments = 100
letterDimension = 40
generateDots = False
# letterRatio = int (dimension / letterDimension )

circleSegments = 72

spacing = 10
segmentLength = 5

angle = 136
center = 0
radius = 0
rRatio = 1
k = 50
ax1 = None
ax2 = None
xsize = dimension # Should be multiple of 20.
ysize = dimension # Should be multiple of 20.
dynamic_ratio = 2

dt = None
tempLine = None
ax1lines = []
ax2lines = []
additionalPoints = []




def update_polygon(tri, polygon):
	if tri == -1:
		points = [0, 0, 0]
	else:
		points = triang.triangles[tri]
	# print("Update polygon Points: ", points)
	xs = triang.x[points]
	ys = triang.y[points]
	polygon.set_xy(np.column_stack([xs, ys]))


def update_polygon2(tri, polygon):
	if tri == -1:
		points = [0, 0, 0]
	else:
		points = triang2.triangles[tri]
	# print("Update polygon2 Points: ", points)
	xs = triang2.x[points]
	ys = triang2.y[points]
	polygon.set_xy(np.column_stack([xs, ys]))


def motion_notify(event):

	if event.inaxes == ax1:
		tri = trifinder(event.xdata, event.ydata)
		handleMove(event, ax1)
	elif event.inaxes == ax2:
		tri = trifinder2(event.xdata, event.ydata) # Make the event handler check both images.
		handleMove(event, ax2)
	else:
		tri = -1

	# update_polygon(tri, polygon1)
	# update_polygon2(tri, polygon2) # alpha - force an update on the other mesh.
	plt.title('In triangle %i' % tri)
	fig = plt.gcf()
	fig.canvas.set_window_title('In triangle %i' % tri)
	event.canvas.draw()


def convertAxesBarycentric(x, y, triang, triang2, triFinder, Originalsamples, Flatsamples):
	# Convert the coordinates on one axes to the cartesian axes on the second axes.

	tri = triFinder(x, y)
	# print(tri)
	# print(triang.triangles[tri])
	face = []
	for vertex in triang.triangles[tri]:  # Create triangle from the coordinates.
		curVertex = Originalsamples[vertex]
		face.append([curVertex[0], curVertex[1]])
	bary1 = calculateBarycentric(face, (x, y))  # Calculate the barycentric coordinates.

	face2 = []
	for vertex in triang2.triangles[tri]:
		curVertex = Flatsamples[vertex]
		face2.append([curVertex[0], curVertex[1]])
	cartesian = get_cartesian_from_barycentric(bary1, face2)

	return cartesian

def replicateDots(linePoints, axTarget, triFinder, triang, triang2, Originalsamples, Flatsamples, generateDots=True, actualTopLeft=(0,0), scale=1):
	global additionalPoints
	# print('Replicate Dots')
	# print(linePoints)
	replicatedPoints = []
	shiftx, shifty = actualTopLeft

	if generateDots == True:
		# Generate barycentric cartesian points on the second axes.

		for linePoint in linePoints:
			# ax1.plot(event.xdata, event.ydata, 'go')

			# print('Replicate Dots Data: ', linePoint.get_xdata(orig=True), linePoint.get_ydata(orig=True))
			x = linePoint.get_xdata(orig=True)[0]
			y = linePoint.get_ydata(orig=True)[0]

			cartesian = convertAxesBarycentric(x, y, triang, triang2, triFinder, Originalsamples, Flatsamples)

			dot, = axTarget.plot(cartesian[0]+shiftx, cartesian[1]+shifty, color='green', marker='.')
			additionalPoints.append(dot)
	else:
		for linePoint in linePoints:
			x = linePoint.get_xdata(orig=True)
			y = linePoint.get_ydata(orig=True)

			newLine = []
			for i in range(len(x)):
				tempX = x[i]
				tempY = y[i]

				cartesian = convertAxesBarycentric(tempX, tempY, triang, triang2, triFinder, Originalsamples, Flatsamples)
				cartesian[0] = cartesian[0]*scale+shiftx
				cartesian[1] = cartesian[1]*scale+shifty
				newLine.append(cartesian)

			replicatedPoints = np.array(newLine)
			line, = axTarget.plot(replicatedPoints[:, 0], replicatedPoints[:, 1], 'r')
			additionalPoints.append(line)


def clearDots(event):
	global ax1lines, ax2lines, additionalPoints

	# print('ax1lines:', ax1lines)
	for lines in ax1lines:
		for point in lines:
			point.remove()

	# print('ax2lines:', ax2lines)
	for lines in ax2lines:
		for point in lines:
			point.remove()

	for points in additionalPoints:
		points.remove()

	ax1lines =  []
	ax2lines = []
	additionalPoints = []
	print('ax1lines:', ax1lines)
	event.canvas.draw()




def on_click(event):
	global dt, tempLine, ax1lines, ax2lines
	print('on click event: ', event.inaxes)
	# https://stackoverflow.com/questions/41824662/how-to-plot-a-dot-each-time-at-the-point-the-mouse-is-clicked-in-matplotlib
	# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
	if event.inaxes == ax1:
		# print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
		linePoints = handleDottedLine(event, event.inaxes)
		if len(linePoints) > 0:
			ax1lines.append(linePoints)
			replicateDots(linePoints, ax2, trifinder, triang, triang2, Originalsamples, Flatsamples)

		event.canvas.draw()
	elif event.inaxes == ax2:
		linePoints = handleDottedLine(event, event.inaxes)
		if len(linePoints) > 0:
			ax2lines.append(linePoints)
			replicateDots(linePoints, ax1, trifinder2, triang2, triang, Flatsamples, Originalsamples)

		event.canvas.draw()


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

def circularLines():
	global angle, spacing, dimension, ax1, circleSegments, ax2, trifinder, triang, triang2, Originalsamples, Flatsamples, generateDots, radius

	circleValues = generateCircularPoints(angle, dimension, spacing, circleSegments, ax1, radius)
	# print(circleValues)
	if generateDots:
		for circle in circleValues:
			circlePoints = []
			for point in circle:
				if trifinder(point[0], point[1]) != -1:
					dot, = ax1.plot(point[0], point[1], '.r' )
				# dot, = ax.plot(linePoint[0], linePoint[1], '.r')
					circlePoints.append(dot)
			ax1lines.append(circlePoints)
			replicateDots(circlePoints, ax2, trifinder, triang, triang2, Originalsamples, Flatsamples, generateDots=generateDots)
	else:
		for circle in circleValues:
			validPoints = []
			for point in circle:
				if trifinder(point[0], point[1]) != -1:
					validPoints.append(point)

			validPoints = np.array(validPoints)
			circleLine, = ax1.plot(validPoints[:, 0], validPoints[:, 1], 'r')

			replicateDots([circleLine], ax2, trifinder, triang, triang2, Originalsamples, Flatsamples, generateDots=generateDots)


def segmentParallelLines( actualTopLeft, ax ):
	xAxisValues, yAxisValues = generateBoundaryPoints(angle, dimension, spacing)
	segmentParallelLineFilter(xAxisValues, yAxisValues, actualTopLeft, ax)

def segmentParallelLineFilter(xAxisValues, yAxisValues, actualTopLeft, axTarget):
	global spacing, segmentLength, generateDots

	''' Actually draw the lines and dots. '''
	dottedLines = []

	for line in xAxisValues:
		startPoint, endPoint = line
		distance = euclidean_distance(startPoint, endPoint)
		segments = distance / spacing  # Determine the number of segments required.
		incrementX = ( endPoint[0] - startPoint[0] ) / segments
		incrementY = ( endPoint[1] - startPoint[1] ) / segments
		top, bottom = findTopBottom(startPoint, endPoint, incrementX, incrementY, distance, trifinder)
		if not (top[0] == -1 or top[1] == -1 or bottom[0] == -1 or bottom[1] == -1):
			line = [top, bottom]
			dottedLines.append( line )

	for line in yAxisValues:
		startPoint, endPoint = line
		distance = euclidean_distance(startPoint, endPoint)
		segments = distance / spacing  # Determine the number of segments required.
		incrementX = ( endPoint[0] - startPoint[0] ) / segments
		incrementY = ( endPoint[1] - startPoint[1] ) / segments
		top, bottom = findTopBottom(startPoint, endPoint, incrementX, incrementY, distance, trifinder)
		if not (top[0] == -1 or top[1] == -1 or bottom[0] == -1 or bottom[1] == -1):
			line = [top, bottom]
			dottedLines.append( line )

	# print(dottedLines)
		# print(bottom)

	lineEndPoints = np.array(dottedLines)
	for line in lineEndPoints:
		# print(line)
		# ax1.plot(line[:,0], line[:,1], marker='.') # For confirmation.

		# Create the lines for the points.
		linePoints = createDottedLine(ax1, line[0], line[1], segmentLength=segmentLength, generateDots=generateDots)
		ax1lines.append(linePoints)
		replicateDots(linePoints, ax2, trifinder, triang, triang2, Originalsamples, Flatsamples, generateDots=generateDots)
		replicateDots(linePoints, axTarget, trifinder, triang, triang2, Originalsamples, Flatsamples, generateDots=generateDots, actualTopLeft=actualTopLeft, scale=0.5)

def parallelLines():
	global angle, spacing, dimension

	xAxisValues, yAxisValues = generateBoundaryPoints(angle, dimension, spacing)
	parallelLineFilter(xAxisValues, yAxisValues)

def parallelLineFilter(xAxisValues, yAxisValues):
	global spacing, segmentLength, generateDots

	''' Actually draw the lines and dots. '''
	dottedLines = []

	for line in xAxisValues:
		startPoint, endPoint = line
		distance = euclidean_distance(startPoint, endPoint)
		segments = distance / spacing  # Determine the number of segments required.
		incrementX = ( endPoint[0] - startPoint[0] ) / segments
		incrementY = ( endPoint[1] - startPoint[1] ) / segments
		top, bottom = findTopBottom(startPoint, endPoint, incrementX, incrementY, distance, trifinder)
		if not (top[0] == -1 or top[1] == -1 or bottom[0] == -1 or bottom[1] == -1):
			line = [top, bottom]
			dottedLines.append( line )

	for line in yAxisValues:
		startPoint, endPoint = line
		distance = euclidean_distance(startPoint, endPoint)
		segments = distance / spacing  # Determine the number of segments required.
		incrementX = ( endPoint[0] - startPoint[0] ) / segments
		incrementY = ( endPoint[1] - startPoint[1] ) / segments
		top, bottom = findTopBottom(startPoint, endPoint, incrementX, incrementY, distance, trifinder)
		if not (top[0] == -1 or top[1] == -1 or bottom[0] == -1 or bottom[1] == -1):
			line = [top, bottom]
			dottedLines.append( line )

	# print(dottedLines)
		# print(bottom)

	lineEndPoints = np.array(dottedLines)
	for line in lineEndPoints:
		# print(line)
		# ax1.plot(line[:,0], line[:,1], marker='.') # For confirmation.

		# Create the lines for the points.
		linePoints = createDottedLine(ax1, line[0], line[1], segmentLength=segmentLength, generateDots=generateDots)
		ax1lines.append(linePoints)
		replicateDots(linePoints, ax2, trifinder, triang, triang2, Originalsamples, Flatsamples, generateDots=generateDots)





# linePoints = generateLinePoints(top, bottom)

def reset(event):
	global triang, triang2, trifinder, trifinder2, ax1, ax2, Originalsamples, Flatsamples, Originalfaces, Flatfaces, polygon1, polygon2, perimeterSegments, startingR, angle
	triang = []
	triang2 = []
	trifinder = []
	trifinder2 = []
	ax1 = None
	ax2 = None
	Originalsamples = []
	Flatsamples = []
	Originalfaces = []
	Flatfaces = []
	polygon1 = None
	polygon2 = None
	genMesh()


def updateAngle(val):
	global angle
	angle = val
	print('New angle: ', angle)

def addButtons(plt):
	''' Buttons '''
	# axcut = plt.axes([0.8, 0.8, 0.1,
	#                   0.075])  # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes. [ left, bottom, width, height ]
	# bprocess = Button(axcut, 'Process', color='red', hovercolor='green')
	# bprocess.on_clicked(reset)

	# axcut = plt.axes([0.8, 0.65, 0.1,
	#                   0.075])  # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes. [ left, bottom, width, height ]
	# breset = Button(axcut, 'Reset', color='red', hovercolor='green')
	# breset.on_clicked(clearDots)  # Bind event for reset button.

	# axcut = plt.axes([0.65, 0.65, 0.1,
	#                   0.075])  # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes. [ left, bottom, width, height ]
	# bangle = Slider(axcut, 'Angle: ' + str(angle), 0, 360, valinit=angle, valfmt='%1.2f', dragging=True)
	# bangle.on_changed(updateAngle)


def genMesh():
	global triang, triang2, trifinder, trifinder2, ax1, ax2, Originalsamples, Flatsamples, Originalfaces, Flatfaces, polygon1, polygon2, xsize, ysize, perimeterSegments, startingR, angle, bprocess, breset, bangle, dimension, character, letterRatio, letterDimension

	generateBlob = True

	if generateBlob:
		# letterDimension = int(dimension / letterRatio)
		letterDimension = letterDimension
		xsize = ysize = dimension
		# letter = genLetter(boxsize=dimension, character=character)
		letter = genLetter(boxsize=letterDimension, character=character, blur=0)
		count, chain, chainDirection, border = generateChainCode(letter)
		print('ChainDirection:', len(chainDirection), chainDirection)
		# writeChainCodeFile('./', 'testChainCode.txt', chainDirection)

		writeChainCodeFile('', Bridson_Common.chaincodefile, chainDirection)
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
		os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=1 --normalizeUVs ")
		# os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=1 --normalizeUVs --nCones=" + str(perimeterSegments))
		# os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=1 --normalizeUVs --nCones=6")
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

		''' Draw Letter blob '''
		axdraw = plt.subplot2grid(gridsize, (0,0))
		# letter = genLetter(boxsize=100, character='P')
		# # print(np.shape(letter))
		# print("Letter:", letter)
		axdraw.imshow(letter)

		''' Buttons '''
		# axcut = plt.axes([0.8, 0.8, 0.1, 0.075]) # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes. [ left, bottom, width, height ]
		# bprocess = Button(axcut, 'Process', color='red', hovercolor='green')
		# bprocess.on_clicked(reset)

		# axcut = plt.axes([0.8, 0.65, 0.1, 0.075]) # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes. [ left, bottom, width, height ]
		# breset = Button(axcut, 'Reset', color='red', hovercolor='green')
		# breset.on_clicked(clearDots) # Bind event for reset button.

		# axcut = plt.axes([0.65, 0.65, 0.1,
		#                   0.075])  # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes. [ left, bottom, width, height ]
		# bangle =  Slider(axcut, 'Angle: ' + str(angle), 0, 360, valfmt='%1.2f', dragging=True)
		# bangle.set_val(angle)
		# bangle.on_changed(updateAngle)

		''' Mesh drawings '''
		ax1 = plt.subplot2grid(gridsize, (1, 0), rowspan=2)
		ax1.set_xlim([-dimension*0.2, dimension*1.2])
		ax1.set_ylim([-dimension * 0.2, dimension * 1.2])
		ax1.grid()

		ax2 = plt.subplot2grid(gridsize, (1, 1), rowspan=2)
		ax2.set_xlim([-dimension*0.2, dimension*1.2])
		ax2.set_ylim([-dimension * 0.2, dimension * 1.2])
		ax2.grid()

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

		''' This block, we are trying to find the normals of the triangles.  Not successful yet. '''
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


def genMeshFromRaster():
	global triang, triang2, trifinder, trifinder2, ax1, ax2, Originalsamples, Flatsamples, Originalfaces, Flatfaces, polygon1, polygon2, xsize, ysize, perimeterSegments, startingR, angle, bprocess, breset, bangle, dimension, character, letterRatio, letterDimension

	startIndex = 1
	regionIndex = startIndex
	imageraster, regionMap = callSLIC(segmentCount=40)
	stopIndex=startIndex+71

	print("Keys:",regionMap.keys())
	# Create the grid.
	gridsize = (3, 2)
	fig = plt.figure(figsize=(12, 8))
	''' Draw Letter blob '''
	axdraw = plt.subplot2grid(gridsize, (0, 0))

	blankRaster = np.zeros(np.shape(imageraster))
	ax3 = plt.subplot2grid(gridsize, (0, 1), rowspan=1)
	ax3.imshow(blankRaster)
	ax3.grid()

	# for regionIndex in range(startIndex, stopIndex):
	for regionIndex in regionMap.keys():
		print(">>>>>>>>>>> Start of cycle")
		resetVariables()
		# try:
		print("Generating index:", regionIndex)
		raster, actualTopLeft = createRegionRasters(regionMap, regionIndex)

		generateBlob = True

		axdraw.imshow(raster)

		# return
		if generateBlob:
			# letterDimension = int(dimension / letterRatio)
			# letterDimension = letterDimension
			# xsize = ysize = dimension
			# letter = genLetter(boxsize=dimension, character=character)
			# letter = genLetter(boxsize=letterDimension, character=character, blur=0)
			letter = raster
			print("Size of Raster: ", np.shape(letter))
			letterDimension=dimension=xsize = ysize = np.max(np.shape(letter))

			count, chain, chainDirection, border = generateChainCode(letter)

			print('ChainDirection:', len(chainDirection), chainDirection)
			# writeChainCodeFile('./', 'testChainCode.txt', chainDirection)
			writeChainCodeFile('', Bridson_Common.chaincodefile, chainDirection)
			print(len(chainDirection))
			perimeterSegments = len(chainDirection)
			startingR = perimeterSegments / 20

			startTime = int(round(time.time() * 1000))
			samples = poisson_disc_samples(width=xsize, height=ysize, r=10, k=k, segments=perimeterSegments)
			# samples = poisson_disc_samples(width=xsize, height=ysize, r=4, k=k, segments=len(chainDirection))
			endTime = int(round(time.time() * 1000))


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

			''' Mesh drawings '''
			ax1 = plt.subplot2grid(gridsize, (1, 0), rowspan=2)
			ax1.set_xlim([-dimension*0.2, dimension*1.2])
			ax1.set_ylim([-dimension * 0.2, dimension * 1.2])
			ax1.set_xlim([-xsize*0.2, xsize*1.2])
			ax1.set_ylim([-ysize * 0.2, ysize * 1.2])
			ax1.grid()

			ax2 = plt.subplot2grid(gridsize, (1, 1), rowspan=2)
			ax2.set_xlim([-dimension*0.2, dimension*1.2])
			ax2.set_ylim([-dimension * 0.2, dimension * 1.2])
			ax2.set_xlim([-xsize*0.2, xsize*1.2])
			ax2.set_ylim([-ysize * 0.2, ysize * 1.2])
			ax2.grid()


			# ax3 = plt.subplot2grid(gridsize, (0,1), rowspan=1)
			# ax3.imshow(blankRaster)
			# ax3.grid()

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
			# Flatsamples = Flatsamples * dimension
			Flatsamples = Flatsamples * xsize
			triang2 = Triangulation(Flatsamples[:, 0], Flatsamples[:, 1], triangles=Flatfaces)
			# ax2 = plt.subplot(122, aspect='equal')  # Create first subplot.
			ax2.triplot(triang2, color='grey')

			triang2.set_mask(np.hypot(Flatsamples[:, 0][triang2.triangles].mean(axis=1),
			                          Flatsamples[:, 1][triang2.triangles].mean(axis=1)) < min_radius)

			''' This block, we are trying to find the normals of the triangles.  Not successful yet. '''
			# tempPoints = np.vstack([Flatsamples[:, 0], Flatsamples[:, 1]]).T
			# tempTri = Delaunay(tempPoints)
			# np.unique(tempTri.simplices.ravel())
			# print('*** CoPlanar:', tempTri.coplanar)

			if False:
				trifinder2 = triang2.get_trifinder()
				polygon2 = Polygon([[0, 0], [0, 0]], facecolor='y')  # dummy data for xs,ys
				update_polygon2(-1, polygon2)
				ax2.add_patch(polygon2)

			print("********** Calling segmentParallelLines")
			segmentParallelLines(actualTopLeft, ax3)
		# except:
		# 	print("Something went wrong with index ", regionIndex)

	# parallelLines()
	#plt.imshow(raster)
	plt.gray()
	# Paint()
	# plt.show()
	return plt


def genMeshFromRaster2():
	global triang, triang2, trifinder, trifinder2, ax1, ax2, Originalsamples, Flatsamples, Originalfaces, Flatfaces, polygon1, polygon2, xsize, ysize, perimeterSegments, startingR, angle, bprocess, breset, bangle, dimension, character, letterRatio, letterDimension

	startIndex = 0 # Index starts at 0.
	regionIndex = startIndex
	imageraster, regionMap = callSLIC(segmentCount=40)
	stopIndex=startIndex+16

	print("Keys:",regionMap.keys())
	# Create the grid.
	gridsize = (3, 2)
	fig = plt.figure(figsize=(12, 8))
	''' Draw Letter blob '''
	axdraw = plt.subplot2grid(gridsize, (0, 0))

	blankRaster = np.zeros(np.shape(imageraster))
	ax3 = plt.subplot2grid(gridsize, (0, 1), rowspan=1)
	# ax3.imshow(blankRaster)
	ax3.imshow(imageraster)
	ax3.grid()

	for regionIndex in range(startIndex, stopIndex):
	# for regionIndex in regionMap.keys():
		print(">>>>>>>>>>> Start of cycle")
		resetVariables()
		# try:
		print("Generating index:", regionIndex)
		raster, actualTopLeft = createRegionRasters(regionMap, regionIndex)

		generateBlob = True

		axdraw.imshow(raster)

		# return
		if generateBlob:
			# letterDimension = int(dimension / letterRatio)
			# letterDimension = letterDimension
			# xsize = ysize = dimension
			# letter = genLetter(boxsize=dimension, character=character)
			# letter = genLetter(boxsize=letterDimension, character=character, blur=0)
			letter = raster
			print("Size of Raster: ", np.shape(letter))
			letterDimension=dimension=xsize = ysize = np.max(np.shape(letter))

			count, chain, chainDirection, border = generateChainCode(letter, rotate=False, angle=90)

			print('ChainDirection:', len(chainDirection), chainDirection)
			# writeChainCodeFile('./', 'testChainCode.txt', chainDirection)
			writeChainCodeFile('', Bridson_Common.chaincodefile, chainDirection)
			print(len(chainDirection))
			perimeterSegments = len(chainDirection)
			startingR = perimeterSegments / 20

			startTime = int(round(time.time() * 1000))
			samples = poisson_disc_samples(width=xsize, height=ysize, r=10, k=k, segments=perimeterSegments)
			# samples = poisson_disc_samples(width=xsize, height=ysize, r=4, k=k, segments=len(chainDirection))
			endTime = int(round(time.time() * 1000))


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

			''' Mesh drawings '''
			ax1 = plt.subplot2grid(gridsize, (1, 0), rowspan=2)
			ax1.set_xlim([-dimension*0.2, dimension*1.2])
			ax1.set_ylim([-dimension * 0.2, dimension * 1.2])
			ax1.set_xlim([-xsize*0.2, xsize*1.2])
			ax1.set_ylim([-ysize * 0.2, ysize * 1.2])
			ax1.grid()

			ax2 = plt.subplot2grid(gridsize, (1, 1), rowspan=2)
			ax2.set_xlim([-dimension*0.2, dimension*1.2])
			ax2.set_ylim([-dimension * 0.2, dimension * 1.2])
			ax2.set_xlim([-xsize*0.2, xsize*1.2])
			ax2.set_ylim([-ysize * 0.2, ysize * 1.2])
			ax2.grid()


			# ax3 = plt.subplot2grid(gridsize, (0,1), rowspan=1)
			# ax3.imshow(blankRaster)
			# ax3.grid()

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
			# Flatsamples = Flatsamples * dimension
			Flatsamples = Flatsamples * xsize
			triang2 = Triangulation(Flatsamples[:, 0], Flatsamples[:, 1], triangles=Flatfaces)
			# ax2 = plt.subplot(122, aspect='equal')  # Create first subplot.
			ax2.triplot(triang2, color='grey')

			triang2.set_mask(np.hypot(Flatsamples[:, 0][triang2.triangles].mean(axis=1),
			                          Flatsamples[:, 1][triang2.triangles].mean(axis=1)) < min_radius)

			''' This block, we are trying to find the normals of the triangles.  Not successful yet. '''
			# tempPoints = np.vstack([Flatsamples[:, 0], Flatsamples[:, 1]]).T
			# tempTri = Delaunay(tempPoints)
			# np.unique(tempTri.simplices.ravel())
			# print('*** CoPlanar:', tempTri.coplanar)

			if False:
				trifinder2 = triang2.get_trifinder()
				polygon2 = Polygon([[0, 0], [0, 0]], facecolor='y')  # dummy data for xs,ys
				update_polygon2(-1, polygon2)
				ax2.add_patch(polygon2)

			print("********** Calling segmentParallelLines")
			segmentParallelLines(actualTopLeft, ax3)
		# except:
		# 	print("Something went wrong with index ", regionIndex)

	# parallelLines()
	#plt.imshow(raster)
	plt.gray()
	# Paint()
	# plt.show()
	return plt

def resetVariables():
	global triang, triang2, trifinder, trifinder2, ax1, ax2, Originalsamples, Flatsamples, Originalfaces, Flatfaces, polygon1, polygon2, xsize, ysize, perimeterSegments, startingR, angle, bprocess, breset, bangle, dimension, character, letterRatio, letterDimension
	triang = triang2 = trifinder= trifinder2= ax1= ax2= Originalsamples= Flatsamples= Originalfaces= Flatfaces= polygon1= polygon2= xsize= ysize= perimeterSegments= startingR= bprocess= breset= bangle= dimension= character= letterRatio= letterDimension=None



if __name__ == '__main__':

	# Original
	'''
	genMesh()

	# circularLines()
	parallelLines()
	# addButtons(plt)
	plt.show()
	'''
	# regionIndex = 18
	# imageraster, regionMap = callSLIC()
	# raster, actualTopLeft = createRegionRasters(regionMap, regionIndex)

	# raster = genSquareRaster(20)
	# print(raster)
	# print(raster)
	genMesh()
	resetVariables()
	# genMeshFromRaster2()
	# genMeshFromRaster()
	# parallelLines()
	print('Done')
	plt.show()


