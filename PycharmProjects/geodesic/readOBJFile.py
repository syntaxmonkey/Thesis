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
	OriginalIndexMap = dict()
	FlatIndexMap = dict()

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


			elif (lineType == 'vt'):
				globalIndex += 1
				flatIndex += 1

				# This is a vertex for flattened mesh.
				# Flatsamples = np.concatenate((Flatsamples, np.array(lineSplit[1:])))
				Flatsamples = np.concatenate((Flatsamples, [float(i) for i in lineSplit[1:]] ))


			elif (lineType == 'f'):
				faceCount += 1
				# Now we need to create the faces.
				# The face entries f v/vt v/vt v/vt
				v1, vf1 = lineSplit[1].split("/")
				v2, vf2 = lineSplit[2].split("/")
				v3, vf3 = lineSplit[3].split("/")

				# So the v and vt are interleaved throughout the file.  We need a way to figure out the un-interleaved index of the vertices.
				# We need a way to map from f index to separated f index.
				newFace = [int(v1)-1, int(v2)-1, int(v3)-1 ]
				newFlatFace = [ int(vf1)-1, int(vf2)-1, int(vf3)-1 ]

				Originalfaces = np.concatenate((Originalfaces, newFace))
				#Originalfaces.append( newFace )
				Flatfaces = np.concatenate((Flatfaces, newFlatFace))

				# Create index maps.
				# OriginalIndexMap[tuple(newFace)] = tuple(newFlatFace)
				# FlatIndexMap[tuple(newFlatFace)] = tuple(newFace)

			else:
				raise("Unkown line type: ", line)

	Originalsamples = np.reshape(Originalsamples, (originalIndex,3))
	Flatsamples = np.reshape(Flatsamples, (flatIndex, 2))
	Originalfaces = np.reshape(Originalfaces, (faceCount, 3))
	Flatfaces = np.reshape(Flatfaces, (faceCount, 3))

	print("GlobalIndex: ", globalIndex)
	print("Originalsamples: ", len(Originalsamples))
	# Shift Original coordinates to positive quadrant.
	xmin = abs(np.min(Originalsamples[:, 0]))
	ymin = abs(np.min(Originalsamples[:, 1]))
	Originalsamples[:, 0] = Originalsamples[:, 0] + xmin
	Originalsamples[:, 1] = Originalsamples[:, 1] + ymin

	print("Originalfaces: ", len(Originalfaces))
	print("Max of faces: ", np.max(Originalfaces))
	print("Flatfaces: ", len(Flatfaces))


	OriginalSampleMap = np.array([None] * len(Originalsamples))
	FlatSamplesMap = np.array([None]*len(Flatsamples))

	# Create Original Triangle <--> Flat Triangle map
	indexOri = 1
	# for triangle in Originalfaces:
	# 	print(triangle)
	# 	Flattriangle = OriginalIndexMap[tuple(triangle)]
	# 	print("Flat triangle: ", Flattriangle)
	# 	indexFlat = np.where(Flatfaces == OriginalIndexMap[tuple(triangle)])
	#
	# 	print("IndexFlat: ", indexFlat)
	# 	print(Flatfaces[indexFlat[:, 0]])
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

	print("Reading OBJ file ****************************************************")
	ValidateSamples(Flatsamples, Flatfaces)
	print("Done OBJ file ****************************************************")
	return Originalsamples, Originalfaces, Flatsamples, Flatfaces



def ValidateSamples(samples, faces):
	# We will go through the list of points and confirm there are no duplicate values.
	existingPoints = {}
	duplicate = False
	for point in samples:
		point = tuple(point)
		# print('Checking: ', point)
		if existingPoints.get(point, None) == None:
			existingPoints[point] = 1
		else:
			print('duplicatePoint: ', point)
			duplicate = True

	if duplicate:
		print("Failed point checking.")
	print("Done checking points.")

	print("Checking Faces: ***************************")
	existingFaces = {}
	duplicate = False
	failedFaces = []
	for face in faces:
		face = tuple(np.sort(face))
		# print('Checking:', face)
		if existingFaces.get(face, None) == None:
			existingFaces[face] = 1
		else:
			print('duplicateFace: ', face)
			duplicate = True
	if duplicate:
		print("Failed Face checking.")
	print("Done Checking Faces: ***************************")

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
	elif event.inaxes == ax2:
		tri = trifinder2(event.xdata, event.ydata) # Make the event handler check both images.
	else:
		tri = -1

	update_polygon(tri, polygon1)
	update_polygon2(tri, polygon2) # alpha - force an update on the other mesh.
	plt.title('In triangle %i' % tri)
	fig = pylab.gcf()
	fig.canvas.set_window_title('In triangle %i' % tri)
	event.canvas.draw()

def motion_notify2(event):
	# No longer required.  Combinted into motion notify1.
	if event.inaxes is None:
		tri = -1
	else:
		tri = trifinder2(event.xdata, event.ydata)

	update_polygon(tri, polygon1) # alpha - force an update on the other mesh.
	update_polygon2(tri, polygon2)
	plt.title('In triangle %i' % tri)
	fig = pylab.gcf()
	fig.canvas.set_window_title('In triangle %i' % tri)
	event.canvas.draw()

def on_click(event):
	# https://stackoverflow.com/questions/41824662/how-to-plot-a-dot-each-time-at-the-point-the-mouse-is-clicked-in-matplotlib
	if event.inaxes == ax1:
		print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
		ax1.plot(event.xdata, event.ydata, ',')
		event.canvas.draw()
	elif event.inaxes == ax2:
		ax2.plot(event.xdata, event.ydata, ',')
		event.canvas.draw()



if __name__ == '__main__':
	path = "../../boundary-first-flattening/build/"
	Originalsamples, Originalfaces, Flatsamples, Flatfaces = readObjFile(path, "test1_out.obj")


	# exit(1)
	#Originalfaces = list(Originalfaces)
	min_radius = .25

	# First subplot
	triang = Triangulation(Originalsamples[:, 0], Originalsamples[:, 1], triangles=Originalfaces)
	ax1 = plt.subplot(121, aspect='equal') # Create first subplot.
	plt.triplot(triang, 'b-')

	triang.set_mask(np.hypot(Originalsamples[:, 0][triang.triangles].mean(axis=1), Originalsamples[:, 1][triang.triangles].mean(axis=1)) < min_radius)
	trifinder = triang.get_trifinder()

	polygon1 = Polygon([[0, 0], [0, 0]], facecolor='y')  # dummy data for xs,ys
	update_polygon(-1, polygon1)

	plt.gca().add_patch(polygon1)
	plt.gcf().canvas.mpl_connect('motion_notify_event', motion_notify)
	# plt.gcf().canvas.mpl_connect('button_press_event', motion_notify1) # https://matplotlib.org/3.1.1/users/event_handling.html
	plt.gcf().canvas.mpl_connect('button_press_event',
	                             on_click)  # https://matplotlib.org/3.1.1/users/event_handling.html

	# Second subplot
	print(Flatfaces)
	triang2 = Triangulation(Flatsamples[:, 0], Flatsamples[:, 1], triangles=Flatfaces)
	ax2 = plt.subplot(122, aspect='equal')  # Create first subplot.
	plt.triplot(triang2, 'b-')

	triang2.set_mask(np.hypot(Flatsamples[:, 0][triang2.triangles].mean(axis=1), Flatsamples[:, 1][triang2.triangles].mean(axis=1)) < min_radius)
	trifinder2 = triang2.get_trifinder()


	polygon2 = Polygon([[0, 0], [0, 0]], facecolor='y')  # dummy data for xs,ys
	update_polygon2(-1, polygon2)

	plt.gca().add_patch(polygon2)
	# plt.gcf().canvas.mpl_connect('motion_notify_event', motion_notify2)
	# plt.gcf().canvas.mpl_connect('button_press_event', on_click) # https://matplotlib.org/3.1.1/users/event_handling.html

	plt.show()

