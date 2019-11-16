

def createObjFile2D(filename, samples, triangleValues):
	f = open(filename, "w+")

	# Output the vertices.
	for coords in samples:
		#print(coords)
		f.write("v %f %f %f\r\n" % (coords[0], coords[1], 0) )

	# Output the facets.
	for facet in triangleValues.simplices.copy():
		f.write("f %d %d %d\r\n" % (facet[0]+1, facet[1]+1, facet[2]+1)) # The facet indeces start at 1, not at 0.  Need to increment index.

	f.close()