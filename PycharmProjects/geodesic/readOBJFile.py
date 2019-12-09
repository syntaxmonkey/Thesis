import math
import numpy as np


# We want to the read the file and try to vertex mappings are retained.
def readObjFile(path, filename):
	f = open(path + filename, "r+")

	samples = np.array([])

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


	print("Line count " + str(count))
	return fileContent

