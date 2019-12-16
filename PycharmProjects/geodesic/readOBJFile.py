import math
import numpy as np
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt # For displaying array as image
from matplotlib.patches import Polygon # https://matplotlib.org/3.1.1/gallery/event_handling/trifinder_event_demo.html
import pylab

# We want to the read the file and try to vertex mappings are retained.
def readObjFile(path, filename):
	f = open(path + filename, "r+")

	Originalsamples = np.array([])
	Flatsamples = np.array([])
	IndexMap = dict()
	Originalfaces = np.array([], dtype=np.int)
	#Originalfaces = []
	Flatfaces = np.array([], dtype=np.int)

	fileContent = f.read().split("\n");
	f.close()

	globalIndex = 0
	originalIndex = 0
	flatIndex = 0
	faceCount = 0

	for line in fileContent:
		# print(line)
		lineSplit = line.split()
		# print(lineSplit)
		if (len(lineSplit) > 0):
			lineType = lineSplit[0]
			if (lineType == 'v'):
				globalIndex += 1
				originalIndex += 1

				# This is a vertex for original mesh.
				# Originalsamples = np.concatenate((Originalsamples, np.array(lineSplit[1:])))
				Originalsamples = np.concatenate((Originalsamples, [float(i) for i in lineSplit[1:]] ))
				IndexMap[str(globalIndex)] = [originalIndex, 0]

			elif (lineType == 'vt'):
				globalIndex += 1
				flatIndex += 1

				# This is a vertex for flattened mesh.
				# Flatsamples = np.concatenate((Flatsamples, np.array(lineSplit[1:])))
				Flatsamples = np.concatenate((Flatsamples, [float(i) for i in lineSplit[1:]] ))
				IndexMap[str(globalIndex)] = [0, flatIndex]

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

				newFace = [int(v1)-1, int(v2)-1, int(v3)-1 ]
				newFlatFace = [ int(vf1)-1, int(vf2)-1, int(vf3)-1 ]

				Originalfaces = np.concatenate((Originalfaces, newFace))
				#Originalfaces.append( newFace )
				Flatfaces = np.concatenate((Flatfaces, newFlatFace))

			else:
				raise("Unkown line type: ", line)

	Originalsamples = np.reshape(Originalsamples, (originalIndex,3))
	# Initsamples = np.array(Initsamples)
	Flatsamples = np.reshape(Flatsamples, (flatIndex, 2))
	# Flatsamples = np.array(Flatsamples)
	# print(Initsamples)
	# print(len(Initsamples))
	# print(Flatsamples)
	# print("originalCount: ", originalIndex)
	# print("flatCount: ", flatIndex)
	Originalfaces = np.reshape(Originalfaces, (faceCount, 3))
	# print(Originalfaces)
	Flatfaces = np.reshape(Flatfaces, (faceCount, 3))
	# print(Flatfaces)
	# print(IndexMap)
	print("GlobalIndex: ", globalIndex)
	print("Originalsamples: ", len(Originalsamples))
	print("Originalfaces: ", len(Originalfaces))
	print("Max of faces: ", np.max(Originalfaces))
	print("Flatfaces: ", len(Flatfaces))
	#print(Originalsamples[559])

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




def update_polygon(tri, polygon):
    if tri == -1:
        points = [0, 0, 0]
    else:
        points = triang.triangles[tri]
    xs = triang.x[points]
    ys = triang.y[points]
    polygon.set_xy(np.column_stack([xs, ys]))


def update_polygon2(tri, polygon):
    if tri == -1:
        points = [0, 0, 0]
    else:
        points = triang2.triangles[tri]
    xs = triang2.x[points]
    ys = triang2.y[points]
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
        tri = trifinder2(event.xdata, event.ydata)
    update_polygon2(tri, polygon2)
    plt.title('In triangle %i' % tri)
    fig = pylab.gcf()
    fig.canvas.set_window_title('In triangle %i' % tri)
    event.canvas.draw()



if __name__ == '__main__':
	path = "../../boundary-first-flattening/build/"
	Originalsamples, Originalfaces, Flatsamples, Flatfaces = readObjFile(path, "test1_out.obj")

	#Originalfaces = list(Originalfaces)
	min_radius = .25

	# First subplot
	triang = Triangulation(Originalsamples[:, 0], Originalsamples[:, 1], triangles=Originalfaces)
	plt.subplot(121, aspect='equal') # Create first subplot.
	plt.triplot(triang, 'b-')

	triang.set_mask(np.hypot(Originalsamples[:, 0][triang.triangles].mean(axis=1), Originalsamples[:, 1][triang.triangles].mean(axis=1)) < min_radius)
	trifinder = triang.get_trifinder()

	polygon1 = Polygon([[0, 0], [0, 0]], facecolor='y')  # dummy data for xs,ys
	update_polygon(-1, polygon1)

	plt.gca().add_patch(polygon1)
	plt.gcf().canvas.mpl_connect('motion_notify_event', motion_notify1)
	#plt.gcf().canvas.mpl_connect('button_press_event', motion_notify1) # https://matplotlib.org/3.1.1/users/event_handling.html


	# Second subplot
	print(Flatfaces)
	triang2 = Triangulation(Flatsamples[:, 0], Flatsamples[:, 1], triangles=Flatfaces)
	plt.subplot(122, aspect='equal')  # Create first subplot.
	plt.triplot(triang2, 'b-')

	triang2.set_mask(np.hypot(Flatsamples[:, 0][triang2.triangles].mean(axis=1), Flatsamples[:, 1][triang2.triangles].mean(axis=1)) < min_radius)
	trifinder2 = triang2.get_trifinder()

	polygon2 = Polygon([[0, 0], [0, 0]], facecolor='y')  # dummy data for xs,ys
	update_polygon2(-1, polygon2)

	plt.gca().add_patch(polygon2)
	plt.gcf().canvas.mpl_connect('motion_notify_event', motion_notify2)
	#plt.gcf().canvas.mpl_connect('button_press_event', motion_notify2) # https://matplotlib.org/3.1.1/users/event_handling.html

	plt.show()

