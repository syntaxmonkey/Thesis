import math
import numpy as np


# We want to the read the file and try to vertex mappings are retained.
def readObjFile(path, filename):
	f = open(path + filename, "r+")

	Originalsamples = np.array([])
	Flatsamples = np.array([])
	IndexMap = dict()
	Originalfaces = np.array([], dtype=np.int)
	Flatfaces = np.array([], dtype=np.int)

	fileContent = f.read().split("\n");
	f.close()

	globalIndex = 1
	originalIndex = 1
	flatIndex = 1
	faceCount = 0

	for line in fileContent:
		# print(line)
		lineSplit = line.split()
		# print(lineSplit)
		if (len(lineSplit) > 0):
			lineType = lineSplit[0]
			if (lineType == 'v'):
				# This is a vertex for original mesh.
				Originalsamples = np.concatenate((Originalsamples, np.array(lineSplit[1:])))
				IndexMap[str(globalIndex)] = [originalIndex, 0]
				globalIndex += 1
				originalIndex += 1

			elif (lineType == 'vt'):
				# This is a vertex for flattened mesh.
				Flatsamples = np.concatenate((Flatsamples, np.array(lineSplit[1:])))
				IndexMap[str(globalIndex)] = [0, flatIndex]
				globalIndex += 1
				flatIndex += 1

			elif (lineType == 'f'):
				faceCount += 1
				# Now we need to create the faces.
				# The face entries f v/vt v/vt v/vt
				v1, vf1 = lineSplit[1].split("/")
				v2, vf2 = lineSplit[2].split("/")
				v3, vf3 = lineSplit[3].split("/")

				# So the v and vt are interleaved throughout the file.  We need a way to figure out the un-interleaved index of the vertices.
				# We need a way to map from f index to separated f index.

				# newFace = ['f', IndexMap[v1][0], IndexMap[v2][0], IndexMap[v3][0] ]
				# newFlatFace = ['f', IndexMap[vf1][1], IndexMap[vf2][1], IndexMap[vf3][1] ]

				newFace = [int(v1), int(v2), int(v3) ]
				newFlatFace = [ int(vf1), int(vf2), int(vf2) ]

				Originalfaces = np.concatenate((Originalfaces, np.array(newFace)))
				Flatfaces = np.concatenate((Flatfaces, np.array(newFlatFace)))

			else:
				raise("Unkown line type: ", line)

	Initsamples = np.reshape(Originalsamples, (originalIndex-1,3))
	# Initsamples = np.array(Initsamples)
	Flatsamples = np.reshape(Flatsamples, (flatIndex-1, 2))
	# Flatsamples = np.array(Flatsamples)
	# print(Initsamples)
	# print(len(Initsamples))
	# print(Flatsamples)
	# print("originalCount: ", originalIndex)
	# print("flatCount: ", flatIndex)
	Originalfaces = np.reshape(Originalfaces, (faceCount, 3))
	print(Originalfaces)
	Flatfaces = np.reshape(Flatfaces, (faceCount, 3))
	# print(Flatfaces)
	# print(IndexMap)

		# Line can be one of the following types: v, vt, or f.
		# v is a vertex in 3D
		# vt is a vertex in 2D
		# f is a facet definition.  It will contain 3 vertices that form a triangle.  There are pairings that map from the 3D vertex to the 2D vertex.

		# TODO:
		# Create arrays for the 3D and 2D vertices.
		# Create mapping 3D to 2D vertices by using the f lines.

		# The vertex references are combined.


			# 1. Create list of vertices.  Either prefix of v or vt.
			#
			# What we need is a list of 3D coordinates and a list of 2D coordinates.
			# The 3D coordinates will represent the original mesh (v)
			# The 2D coordinates will represent the flattened mesh (vt)
			#
			# The vertices will be interleaved.  So the indeces are common between the 3D and 2D coordinates.
			#
			#
			# 2. Create list of facets.
			# These will be prefixed with an f, then followed by three pairs of indeces in the format of <v>/<vt>.
			#
			# These represent the triangles.
			#
			# 3. Create the Triangulation objects for the 3D Mesh and 2D mesh.
			# The Triangulation objects contain x, y, and triangle data structures: https://www.programcreek.com/python/example/91951/matplotlib.tri.Triangulation
			#
			#

	return Originalsamples, Originalfaces, Flatsamples, Flatfaces







if __name__ == '__main__':
	path = "../../boundary-first-flattening/build/"
	readObjFile(path, "test1_out.obj")