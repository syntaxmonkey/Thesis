import math


def createObjFile2D(path, filename, samples, triangleValues, radius, center, distance):
	f = open(path + filename, "w+")

	radius2 = radius+1
	#print(radius2)
	# Output the vertices.
	count=0
	for coords in samples:
		#print(coords)
		dist = distance(center, coords) / radius2
		# print("Dist:", dist)
		#print(dist*dist)
		#zvalue = math.sqrt(radius2 - coords[0]*coords[0] - coords[1]*coords[1])
		# if dist > 1.0:
		# 	dist = 1.0
		# dist=dist*.75
		zvalue = math.sqrt(1 - dist*dist) * radius2
		#f.write("v %f %f %f\r\n" % (coords[0], coords[1], 0) )
		f.write("v %f %f %f\r\n" % (coords[0], coords[1], zvalue))
		count += 1

	# Output the facets.
	for facet in triangleValues.simplices.copy():
		f.write("f %d %d %d\r\n" % (facet[0]+1, facet[1]+1, facet[2]+1)) # The facet indeces start at 1, not at 0.  Need to increment index.

	f.close()


def createObjFile3D(path, filename, samples, triangleValues, radius, center, distance):
	f = open(path + filename, "w+")

	radius2 = radius+1
	#print(radius2)
	# Output the vertices.
	for coords in samples:
		#print(coords)
		#dist = distance(center, coords) / radius2
		#print(dist*dist)
		#zvalue = math.sqrt(radius2 - coords[0]*coords[0] - coords[1]*coords[1])
		#zvalue = math.sqrt(1 - dist*dist) * radius2
		#f.write("v %f %f %f\r\n" % (coords[0], coords[1], 0) )
		print("Writing coords: ", coords)
		f.write("v %f %f %f\r\n" % (coords[0], coords[1], coords[2]))

	# Output the facets.
	for facet in triangleValues.simplices.copy():
		f.write("f %d %d %d\r\n" % (facet[0]+1, facet[1]+1, facet[2]+1)) # The facet indeces start at 1, not at 0.  Need to increment index.

	f.close()