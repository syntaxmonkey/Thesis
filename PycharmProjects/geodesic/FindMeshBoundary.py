

def findTopY(x, maxDimension, trifinder):
	actualY = -1
	for y in range(maxDimension):
		tri = trifinder(x, y) # If the return value is -1, then no triangle was found.
		if tri != -1:
			actualY = y
			break
	return actualY


def findBottomY(x, maxDimension, trifinder):
	actualY = -1
	for y in range(maxDimension, 0, -1):
		tri = trifinder(x, y) # If the return value is -1, then no triangle was found.
		if tri != -1:
			actualY = y
			break
	return actualY


def findTopBottom(x, maxDimension, trifinder, angle):
	print('findTopBottom: ', angle)
	topY = findTopY(x, maxDimension, trifinder)
	bottomY = findBottomY(x, maxDimension, trifinder)
	return [(x,topY), (x,bottomY)]





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

def findLeftRight(y, maxDimension, trifinder):
	leftX = findLeftX(y, maxDimension, trifinder)
	rightX = findRightX(y, maxDimension, trifinder)
	return [(leftX, y), (rightX, y)]