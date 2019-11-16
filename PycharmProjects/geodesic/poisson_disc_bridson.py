from random import random
from math import cos, sin, floor, sqrt, pi, ceil
import time
import matplotlib
matplotlib.use('tkagg')
from driver import genCircleCoords
import numpy as np

import matplotlib.pyplot as plt # For displaying array as image

from scipy.spatial import Voronoi, voronoi_plot_2d # For generating Voronoi graphs - https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.spatial.Voronoi.html
from scipy.spatial import Delaunay # For generating Delaunay - https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.html

import createOBJFile
import os

def euclidean_distance(a, b):
	dx = a[0] - b[0]
	dy = a[1] - b[1]
	return sqrt(dx * dx + dy * dy)


def poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=random, segments=5):

	def insert_coords(p, grid):
		grid_x, grid_y = grid_coords(p)
		grid[grid_x + grid_y * grid_width] = p

	def grid_coords(p):
		return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

	def fits(p, gx, gy):
		if distance(p, center) > radius: # Case if point is outside of the circle.
			return False
		yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
		for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
			for y in yrange:
				#print(p, x, y, grid_width)
				g = grid[x + y * grid_width]
				if g is None:
					continue
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
useSegmentRadius = False # When set to True, will use the minimum distance between perimeter points.  False will use 60% of perimeter point distance.
forceCenter = False # When set to True, will force inject the center point onto canvas.
genVoronoi = True

path = "../../boundary-first-flattening/build/"

rRatio = 1
k = 100
xsize = 100 # Should be multiple of 20.
ysize = 100 # Should be multiple of 20.
startTime = int(round(time.time() * 1000))
samples = poisson_disc_samples(width=xsize, height=ysize, r=10, k=k, segments=11*4)
endTime = int(round(time.time() * 1000))

samples = np.array(samples) # Need to convert to np array to have proper slicing.

print("Execution time: %d ms" % (endTime - startTime))

raster = [[0 for i in range(xsize)] for j in range(ysize)]

for coords in samples:
	x, y = coords
	xint = int(x)
	yint = int(y)
	raster[xint][yint] = int(255)


if not genVoronoi:
	print(raster)
	plt.imshow(raster)
else:
	vor = Voronoi(samples)
	#voronoi_plot_2d(vor)
	tri = Delaunay(samples)
	#print(samples[:,])

	#print(samples[0], tri.simplices[0])
	plt.triplot(samples[:, 0], samples[:, 1], tri.simplices.copy()) # tri.simplices are indeces to the points.  They represent the three vertices that form a facet.
	plt.plot(samples[:, 0], samples[:, 1], 'o')


# Create object file for image.
createOBJFile.createObjFile2D(path, "test1.obj", samples, tri)

# Reshape with BFF.
print("Reshaping with BFF")
os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=0.3")
# Extract the flattened version of the image.
print()
print("Extracting 2D image post BFF Reshaping")
os.system(path + "extract.py test1_out.obj test1_out_flat.obj")

#plt.imshow(raster)
plt.gray()
plt.show()
