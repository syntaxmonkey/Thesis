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

import pylab


def euclidean_distance(a, b):
	dx = a[0] - b[0]
	dy = a[1] - b[1]
	return sqrt(dx * dx + dy * dy)


def poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=random, segments=5):
	global radius
	global center
	global dynamic_ratio

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



	centerx = xsize / 2
	centery = ysize / 2
	center = (centerx, centery)
	radius = xsize / 2 - 1


	print("Segments: ", segments)
	# Generate the list of circle Perimeters.
	circlePerimeter = genCircleCoords(width, height, center, radius, segments=segments )
	#print(circlePerimeter)

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

	print("Actual r: ", r)


	tau = 2 * pi
	cellsize = r / sqrt(2)

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


'''
	Settings.
	
'''
useSegmentRadius = True # When set to True, will use the minimum distance between perimeter points.  False will use 60% of perimeter point distance.
forceCenter = False # When set to True, will force inject the center point onto canvas.
genVoronoi = True
useDynamicRatio = False

center = 0
radius = 0
rRatio = 1
k = 50
xsize = 100 # Should be multiple of 20.
ysize = 100 # Should be multiple of 20.
perimeterSegments = 60
startingR = perimeterSegments / 10
dynamic_ratio = 2
startTime = int(round(time.time() * 1000))
samples = poisson_disc_samples(width=xsize, height=ysize, r=20, k=k, segments=perimeterSegments)
endTime = int(round(time.time() * 1000))

samples = np.array(samples) # Need to convert to np array to have proper slicing.

print("Execution time: %d ms" % (endTime - startTime))

raster = [[0 for i in range(xsize)] for j in range(ysize)]

for coords in samples:
	x, y = coords
	xint = int(x)
	yint = int(y)
	raster[xint][yint] = int(255)

def update_polygon(tri, polygon):
    if tri == -1:
        points = [0, 0, 0]
    else:
        points = triang.triangles[tri]
    xs = triang.x[points]
    ys = triang.y[points]
    polygon.set_xy(np.column_stack([xs, ys]))


def motion_notify1(event):
    if event.inaxes is None:
        tri = -1
    else:
        tri = trifinder(event.xdata, event.ydata)
    update_polygon(tri, polygon1)
    plt.title('In triangle %i' % tri)
    fig = pylab.gcf()
    fig.canvas.set_window_title('In triangle %i' % tri)

    event.canvas.draw()

def motion_notify2(event):
    if event.inaxes is None:
        tri = -1
    else:
        tri = trifinder(event.xdata, event.ydata)
    update_polygon(tri, polygon2)
    plt.title('In triangle %i' % tri)
    fig = pylab.gcf()
    fig.canvas.set_window_title('In triangle %i' % tri)
    event.canvas.draw()



if not genVoronoi:
	print(raster)
	plt.imshow(raster)
else:
	#vor = Voronoi(samples)

	#voronoi_plot_2d(vor)
	tri = Delaunay(samples)  # Generate the triangles from the vertices.


	# Produce the mesh file.  Flatten the mesh with BFF.  Extract the 2D from BFF flattened mesh.
	path = "../../boundary-first-flattening/build/"
	# Create object file for image.
	createOBJFile.createObjFile2D(path, "test1.obj", samples, tri, radius, center, distance=euclidean_distance)
	# Reshape with BFF.
	print("Reshaping with BFF")
	os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=0.3")
	# Extract the flattened version of the image.
	print()
	print("Extracting 2D image post BFF Reshaping")
	os.system(path + "extract.py test1_out.obj test1_out_flat.obj")


	#print(samples[:,])
	min_radius = 0.25
	# https://matplotlib.org/3.1.1/gallery/event_handling/trifinder_event_demo.html
	triang = Triangulation(samples[:, 0], samples[:, 1])
	triang.x[0]=triang.x[0]*1
	triang.set_mask(np.hypot(samples[:, 0][triang.triangles].mean(axis=1), samples[:, 1][triang.triangles].mean(axis=1)) < min_radius)
	trifinder = triang.get_trifinder()

	# First subplot
	polygon1 = Polygon([[0, 0], [0, 0]], facecolor='y')  # dummy data for xs,ys
	update_polygon(-1, polygon1)
	plt.subplot(121, aspect='equal') # Create first subplot.
	#print(samples[0], tri.simplices[0])
	#plt.triplot(samples[:, 0], samples[:, 1], tri.simplices.copy()) # tri.simplices are indeces to the points.  They represent the three vertices that form a facet.
	plt.triplot(triang, 'bo-')
	#plt.plot(samples[:, 0], samples[:, 1], 'o')
	# https://matplotlib.org/3.1.1/gallery/event_handling/trifinder_event_demo.html
	plt.gca().add_patch(polygon1)
	#plt.gcf().canvas.mpl_connect('motion_notify_event', motion_notify1)
	plt.gcf().canvas.mpl_connect('button_press_event', motion_notify1) # https://matplotlib.org/3.1.1/users/event_handling.html


	readOBJFile.readObjFile(path, "test1_out.obj")

	# Second subplot
	polygon2 = Polygon([[0, 0], [0, 0]], facecolor='y')  # dummy data for xs,ys
	update_polygon(-1, polygon2)
	plt.subplot(122, aspect='equal')  # Create first subplot.
	plt.triplot(triang, 'go-')
	plt.gca().add_patch(polygon2)
	#plt.gcf().canvas.mpl_connect('motion_notify_event', motion_notify2)
	plt.gcf().canvas.mpl_connect('button_press_event', motion_notify2) # https://matplotlib.org/3.1.1/users/event_handling.html





#plt.imshow(raster)
plt.gray()
plt.show()
