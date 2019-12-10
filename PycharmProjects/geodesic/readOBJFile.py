import math
import numpy as np


# We want to the read the file and try to vertex mappings are retained.
def readObjFile(path, filename):
	f = open(path + filename, "r+")

	Initsamples = np.array([])
	Flatsamples = np.array([])

	fileContent = f.read().split("\n");
	f.close()

	count = 0


	for line in fileContent:
		print(line)
		count += 1

		# Line can be one of the following types: v, vt, or f.
		# v is a vertex in 3D
		# vt is a vertex in 2D
		# f is a facet definition.  It will contain 3 vertices that form a triangle.  There are pairings that map from the 3D vertex to the 2D vertex.

		# TODO:
		# Create arrays for the 3D and 2D vertices.
		# Create mapping 3D to 2D vertices by using the f lines.

		# The vertex references are combined.

		'''
			1. Create list of vertices.  Either prefix of v or vt.
			
			What we need is a list of 3D coordinates and a list of 2D coordinates.
			The 3D coordinates will represent the original mesh (v)
			The 2D coordinates will represent the flattened mesh (vt)
			
			The vertices will be interleaved.  So the indeces are common between the 3D and 2D coordinates. 
			
			
			2. Create list of facets.
			These will be prefixed with an f, then followed by three pairs of indeces in the format of <v>/<vt>.
			
			These represent the triangles.
			
			3. Create the Triangulation objects for the 3D Mesh and 2D mesh.
			The Triangulation objects contain x, y, and triangle data structures: https://www.programcreek.com/python/example/91951/matplotlib.tri.Triangulation
			
			
			
		'''


	print("Line count " + str(count))
	return fileContent

