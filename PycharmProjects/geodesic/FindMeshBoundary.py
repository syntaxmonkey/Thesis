import math
import matplotlib.pyplot as plt # For displaying array as image
import numpy as np

def euclidean_distance(a, b):
	dx = a[0] - b[0]
	dy = a[1] - b[1]
	return math.sqrt(dx * dx + dy * dy)


def findTopY(x, maxDimension, trifinder, incrementX, incrementY):
	actualY = -1
	actualX = -1
	currentX = x

	yValues = []
	i = 0
	while i < maxDimension:
		yValues.append(i)
		i += incrementY

	print(yValues)
	for y in yValues:
		tri = trifinder(currentX, y) # If the return value is -1, then no triangle was found.
		if tri != -1:
			actualY = y
			actualX = currentX
			break
		currentX += incrementX
	return actualX, actualY


def findBottomY(x, maxDimension, trifinder, incrementX, incrementY):
	actualY = -1
	actualX = -1
	currentX = x

	yValues = []
	i = 0
	while i < maxDimension:
		yValues.append(i)
		i += incrementY

	for y in yValues:
		tri = trifinder(x, y) # If the return value is -1, then no triangle was found.
		if tri != -1:
			actualY = y
			actualX = currentX
			break
		currentX += incrementX
	return actualX, actualY


def findBottom(startPoint, endPoint, incrementX, incrementY, distance, trifinder):
	actualY = -1
	actualX = -1

	points = []
	currentPoint = startPoint

	while euclidean_distance(currentPoint, startPoint) <= distance:
		tri = trifinder(currentPoint[0], currentPoint[1]) # If the return value is -1, then no triangle was found.
		if tri != -1:
			actualY = currentPoint[1]
			actualX = currentPoint[0]
			break
		currentPoint = (currentPoint[0] + incrementX, currentPoint[1] + incrementY)
	return actualX, actualY


def findTop(startPoint, endPoint, incrementX, incrementY, distance, trifinder):
	actualY = -1
	actualX = -1

	points = []
	currentPoint = startPoint

	while euclidean_distance(currentPoint, startPoint) <= distance:
		tri = trifinder(currentPoint[0], currentPoint[1]) # If the return value is -1, then no triangle was found.
		if tri != -1:
			actualY = currentPoint[1]
			actualX = currentPoint[0]
			break
		currentPoint = (currentPoint[0] + incrementX, currentPoint[1] + incrementY)
	return actualX, actualY



def findTopBottom(startPoint, endPoint, incrementX, incrementY, distance, trifinder):

	# print("findTopBottom - Increments: ", incrementX, incrementY)
	topX, topY = findTop(startPoint, endPoint, incrementX, incrementY, distance, trifinder)
	bottomX, bottomY = findBottom(endPoint, startPoint, -incrementX, -incrementY, distance, trifinder)
	return [[topX,topY], [bottomX,bottomY]]




def findLeftX(y, maxDimension, trifinder):
	actualX = -1
	for x in range(maxDimension):
		tri = trifinder(x, y) # If the return value is -1, then no triangle was found.
		if tri != -1:
			actualX = x
			break
	return actualX


def findRightX(y, maxDimension, trifinder):
	actualX = -1
	for x in range(maxDimension, 0, -1):
		tri = trifinder(x, y) # If the return value is -1, then no triangle was found.
		if tri != -1:
			actualX = x
			break
	return actualX

def findLeftRight(y, maxDimension, trifinder, spacing):
	leftX = findLeftX(y, maxDimension, trifinder)
	rightX = findRightX(y, maxDimension, trifinder)
	return [(leftX, y), (rightX, y)]


def generateBoundaryPoints(angle, dimension, spacing):
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

	if xaxis:
		xAxisValues = generateXAxisPoints(angle, dimension, spacing)
		yAxisValues = []
	else:
		xAxisValues = []
		yAxisValues = generateYAxisPoints(angle, dimension, spacing)

	return xAxisValues, yAxisValues

def generateYAxisPoints(angle, dimension, spacing):
	AxisValues = []
	extension = int(dimension * 0.8) # At most we need extension for 45 degrees: sin(45) = 0.707
	print('Extension: ', extension)
	for i in range(-extension, dimension + extension, spacing):
		singleLine = []
		startPoint = [0, i]
		singleLine.append(startPoint)
		singleLine.append(generateYAxisLimitPoint(angle, dimension, spacing, startPoint))
		AxisValues.append(singleLine)
	# print(AxisValues)
	return np.array(AxisValues)



def generateXAxisPoints(angle, dimension, spacing):
	AxisValues = []
	# extension =  int(math.sin(math.radians(angle)) * dimension *1.5 ) # int(dimension * 0.75) # At most we need extension for 45 degrees: sin(45) = 0.707
	extension = int(dimension * 0.8) # At most we need extension for 45 degrees: sin(45) = 0.707
	print('Extension: ', extension)
	for i in range(-extension, dimension + extension, spacing):
		singleLine = []
		startPoint = [i,0]
		singleLine.append(startPoint)
		singleLine.append(generateXAxisLimitPoint(angle, dimension, spacing, startPoint))
		AxisValues.append(singleLine)
	# print(AxisValues)
	return np.array(AxisValues)


def generateXAxisLimitPoint(angle, dimension, spacing, startPoint):
	# section a, d, and angle 0.
	if angle == 0:
		incrementX = 0
		incrementY = dimension
	elif angle > 0 and angle <= 45:
		incrementX = math.sin(math.radians(angle))
		incrementY = math.cos(math.radians(angle))
		adjustToDimension = dimension / abs(incrementY)
		incrementX *= adjustToDimension
		incrementY *= adjustToDimension
	elif angle > 135 and angle < 180:
		incrementX = -math.sin(math.radians(angle)) # Flip the direction to stay in positive grid.
		incrementY = -math.cos(math.radians(angle)) # Flip the direction to stay in positive grid.
		adjustToDimension = dimension / abs(incrementY)
		incrementX *= adjustToDimension
		incrementY *= adjustToDimension

	return [startPoint[0]+incrementX, startPoint[1]+incrementY]

def generateYAxisLimitPoint(angle, dimension, spacing, startPoint):
	# section a, d, and angle 0.
	if angle == 90:
		incrementX = dimension
		incrementY = 0
	elif angle > 45 and angle < 90:
		incrementX = math.sin(math.radians(angle))
		incrementY = math.cos(math.radians(angle))
		adjustToDimension = dimension / abs(incrementX)
		incrementX *= adjustToDimension
		incrementY *= adjustToDimension
	elif angle > 90 and angle <= 135:
		incrementX = math.sin(math.radians(angle))
		incrementY = -math.cos(math.radians(angle)) # Flip the direction to stay in positive grid.
		adjustToDimension = dimension / abs(incrementX)
		incrementX *= adjustToDimension
		incrementY *= adjustToDimension

	return [startPoint[0]+incrementX, startPoint[1]+incrementY]




if __name__ == '__main__':
	angle = 4
	spacing = 20
	gridsize = (3, 2)
	dimension = 100
	lim = 200
	fig = plt.figure(figsize=(12, 8))

	ax1 = plt.subplot2grid(gridsize, (1, 0), rowspan=2 )
	# ax1.set_xticks(np.arange(0, dimension, spacing))
	# ax1.set_yticks(np.arange(0, dimension, spacing))
	ax1.grid()
	ax2 = plt.subplot2grid(gridsize, (1, 1), rowspan=2 )
	# ax1.set_xlim([-lim, lim])
	# ax1.set_ylim([-lim, lim])

	xAxisValues, yAxisValues = generateBoundaryPoints(angle, dimension, spacing)

	for line in xAxisValues:
		# print(line)
		ax1.plot(line[:,0], line[:,1], marker='o')

	for line in yAxisValues:
		ax1.plot(line[:, 0], line[:, 1], marker='o')


	# ax1.plot((45,45), marker='o')

	plt.grid()
	plt.show()