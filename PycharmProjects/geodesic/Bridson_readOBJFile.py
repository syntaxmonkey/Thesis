import math
import Bridson_Common
import numpy as np
import random




# We want to the read the file and try to vertex mappings are retained.
def readFlatObjFile(path, filename):
	f = open(path + filename, "r+")

	Flatsamples = np.array([])
	OriginalIndexMap = dict()
	FlatIndexMap = dict()

	#Originalfaces = []
	Flatfaces = np.array([], dtype=np.int)

	fileContent = f.read().split("\n");
	f.close()

	globalIndex = 0
	originalIndex = 0
	flatIndex = 0
	faceCount = 0

	for line in fileContent:
		# Bridson_Common.logDebug(__name__, line)
		lineSplit = line.split()
		# Bridson_Common.logDebug(__name__, lineSplit)
		if (len(lineSplit) > 0):
			lineType = lineSplit[0]
			if (lineType == 'v'):
				globalIndex += 1
				originalIndex += 1

				# This is a vertex for original mesh.
				Flatsamples = np.concatenate((Flatsamples, [float(i) for i in lineSplit[1:]] ))
				flatIndex += 1
			elif (lineType == 'f'):
				faceCount += 1
				# Now we need to create the faces.
				# The face entries f v/vt v/vt v/vt
				# Handle 3d case.
				# v1, vf1 = lineSplit[1].split("/")
				# v2, vf2 = lineSplit[2].split("/")
				# v3, vf3 = lineSplit[3].split("/")

				# Handle 2d case.
				vf1 = lineSplit[1]
				vf2 = lineSplit[2]
				vf3 = lineSplit[3]

				# So the v and vt are interleaved throughout the file.  We need a way to figure out the un-interleaved index of the vertices.
				# We need a way to map from f index to separated f index.
				newFlatFace = [ int(vf1)-1, int(vf2)-1, int(vf3)-1 ]

				# Originalfaces = np.concatenate((Originalfaces, newFace))
				Flatfaces = np.concatenate((Flatfaces, newFlatFace))

			else:
				raise("Unkown line type: ", line)

	Flatsamples = np.reshape(Flatsamples, (flatIndex, 2))
	Flatfaces = np.reshape(Flatfaces, (faceCount, 3))

	# Bridson_Common.logDebug(__name__, "GlobalIndex: ", globalIndex)
	# Shift Original coordinates to positive quadrant.
	xmin = abs(np.min(Flatsamples[:, 0]))
	ymin = abs(np.min(Flatsamples[:, 1]))
	Flatsamples[:, 0] = Flatsamples[:, 0] + xmin
	Flatsamples[:, 1] = Flatsamples[:, 1] + ymin

	Bridson_Common.logDebug(__name__, "Flatface: ", len(Flatfaces))
	Bridson_Common.logDebug(__name__, "Max of faces: ", np.max(Flatfaces))


	# Create Original Triangle <--> Flat Triangle map
	indexOri = 1
	# for triangle in Originalfaces:
	# 	Bridson_Common.logDebug(__name__, triangle)
	# 	Flattriangle = OriginalIndexMap[tuple(triangle)]
	# 	Bridson_Common.logDebug(__name__, "Flat triangle: ", Flattriangle)
	# 	indexFlat = np.where(Flatfaces == OriginalIndexMap[tuple(triangle)])
	#
	# 	Bridson_Common.logDebug(__name__, "IndexFlat: ", indexFlat)
	# 	Bridson_Common.logDebug(__name__, Flatfaces[indexFlat[:, 0]])
	#
	# 	OriginalSampleMap[indexOri] = indexFlat[0][0]
	#
	# 	indexOri += 1
	# 	break;


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

	Bridson_Common.logDebug(__name__, "Reading OBJ file ****************************************************")
	#ValidateSamples(Flatsamples, Flatfaces)
	Bridson_Common.logDebug(__name__, "Done OBJ file ****************************************************")
	return  Flatsamples, Flatfaces

def createObjFile2D(path, filename, samples, triangleValues, radius, center, distance):

	f = open(path + filename, "w+")

	radius2 = radius+1
	#Bridson_Common.logDebug(__name__, radius2)
	# Output the vertices.
	Vcount=0
	for coords in samples:
		#Bridson_Common.logDebug(__name__, coords)
		dist = distance(center, coords) / radius2
		# Bridson_Common.logDebug(__name__, "Dist:", dist)
		#Bridson_Common.logDebug(__name__, dist*dist)
		#zvalue = math.sqrt(radius2 - coords[0]*coords[0] - coords[1]*coords[1])
		# if dist > 1.0:
		# 	dist = 1.0
		# dist=dist*.75
		# zvalue = math.sqrt(1 - dist*dist) * radius2
		zvalue = 0
		#f.write("v %f %f %f\r\n" % (coords[0], coords[1], 0) )
		f.write("v %f %f %f\r\n" % (coords[0], coords[1], zvalue))
		# Bridson_Common.logDebug(__name__, "v %f %f %f\r\n" % (coords[0], coords[1], zvalue))
		Vcount += 1
	Bridson_Common.logDebug(__name__, "Vertex Count:", Vcount)

	averageArea = Bridson_Common.findAverageArea(triangleValues.triangles, samples)

	Bridson_Common.logDebug(__name__, "createObj File Average Area:",averageArea)

	Fcount = 0
	# Output the facets.
	# for facet in triangleValues.simplices.copy():
	for facet in triangleValues.triangles:
		# Bridson_Common.logDebug(__name__, "Facet:", facet)
		area = Bridson_Common.findArea(samples[facet[0]], samples[facet[1]], samples[facet[2]])
		if area < averageArea / 100.0:
			Bridson_Common.logDebug(__name__, "Removing Triangle Area:", area)
			Bridson_Common.logDebug(__name__, "f %d %d %d\r\n" % (facet[0]+1, facet[1]+1, facet[2]+1))
		else:
			f.write("f %d %d %d\r\n" % (facet[0] + 1, facet[1] + 1, facet[2] + 1))  # The facet indeces start at 1, not at 0.  Need to increment index.

		Fcount+=1

	Bridson_Common.logDebug(__name__, "Facet Count:", Fcount)
	Bridson_Common.logDebug(__name__, "Facet - Vertex:", Fcount - Vcount)
	f.close()


def createObjFile3D(path, filename, samples, triangleValues, radius, center, distance):
	f = open(path + filename, "w+")

	radius2 = radius+1
	#Bridson_Common.logDebug(__name__, radius2)
	# Output the vertices.
	for coords in samples:
		#Bridson_Common.logDebug(__name__, coords)
		#dist = distance(center, coords) / radius2
		#Bridson_Common.logDebug(__name__, dist*dist)
		#zvalue = math.sqrt(radius2 - coords[0]*coords[0] - coords[1]*coords[1])
		#zvalue = math.sqrt(1 - dist*dist) * radius2
		#f.write("v %f %f %f\r\n" % (coords[0], coords[1], 0) )
		Bridson_Common.logDebug(__name__, "Writing coords: ", coords)
		f.write("v %f %f %f\r\n" % (coords[0], coords[1], coords[2]))

	# Output the facets.
	for facet in triangleValues.simplices.copy():
		f.write("f %d %d %d\r\n" % (facet[0]+1, facet[1]+1, facet[2]+1)) # The facet indeces start at 1, not at 0.  Need to increment index.

	f.close()