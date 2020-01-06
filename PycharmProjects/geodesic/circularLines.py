import math
import matplotlib.pyplot as plt # For displaying array as image
import numpy as np


def generateCircularPoints(angle, dimension, spacing, segments, axis):
	angle = angle % 360
	# Generate all the points around the boundary.
	# Find x values along the y = 0 axis.
	print ('starting angle: ', angle)
	xaxis = False
	if angle == 180:
		angle = angle - 180
		xaxis = True
	elif angle == 270:
		angle = angle - 180
		xaxis = False
	elif angle == 90:
		xaxis = False
	elif angle == 0:
		xaxis = True
	elif  angle > 45 and angle <= 135:
		# section b and c.
		xaxis = False
	elif (angle > 180 and angle <= 225) or (angle > 315 and angle <= 360) :
		# section e and h
		angle = angle - 180
		xaxis = True
	elif (angle > 225 and angle <= 315):
		# section f and g
		angle = angle - 180
		xaxis = False
	else:
		# section a and d.
		xaxis = True

	print('Actual angle: ', angle)
	print('XAxis: ', xaxis)

	xcenter = dimension / 2
	ycenter = dimension / 2

	radius = math.ceil(dimension / 2)
	print('Radius: ', radius)

	ringCount = math.floor(radius / spacing)
	spacing = math.floor(radius / ringCount) - 1
	circleValues=[]

	for circleRadius in range(0, radius, spacing):
		theta = np.linspace(0, 2 * np.pi, segments)

		# r = np.sqrt(1.0)

		x = circleRadius * np.cos(theta)
		x = x + xcenter

		# print(x)
		y = circleRadius * np.sin(theta)
		y = y + ycenter
		# print(x)
		# fig, ax = plt.subplots(1)
		circle = np.vstack((x,y)).T
		# print(circle)
		# axis.plot(circle[:, 0], circle[:, 1])
		circleValues.append(circle)
		# axis.set_aspect(1)

	# if xaxis:
	# 	xAxisValues = generateXAxisPoints(angle, dimension, spacing)
	# 	yAxisValues = []
	# else:
	# 	xAxisValues = []
	# 	yAxisValues = generateYAxisPoints(angle, dimension, spacing)

	return circleValues





if __name__ == '__main__':
	angle = 4
	spacing = 20
	gridsize = (3, 2)
	dimension = 100
	segments = 100
	lim = 200
	fig = plt.figure(figsize=(12, 8))

	ax1 = plt.subplot2grid(gridsize, (1, 0), rowspan=2 )
	# ax1.set_xticks(np.arange(0, dimension, spacing))
	# ax1.set_yticks(np.arange(0, dimension, spacing))
	ax1.grid()
	ax2 = plt.subplot2grid(gridsize, (1, 1), rowspan=2 )
	# ax1.set_xlim([-lim, lim])
	# ax1.set_ylim([-lim, lim])

	circleValues= generateCircularPoints(angle, dimension, spacing, segments, ax1)

	# for circle in circleValues:
		# print(line)
		# ax1.plot(circle[:,0], circle[:,1], marker='.')


	# ax1.plot((45,45), marker='o')

	plt.grid()
	plt.show()