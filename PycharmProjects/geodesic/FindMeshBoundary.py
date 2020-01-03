

def findTopY(x, maxDimension, trifinder):
	actualY = -1
	for y in range(maxDimension):
		tri = trifinder(x, y) # If the return value is -1, then no triangle was found.
		if tri != -1:
			break
	actualY = y
	return actualY


def findBottomY(x, maxDimension, trifinder):
	actualY = -1
	for y in range(maxDimension, 0, -1):
		tri = trifinder(x, y) # If the return value is -1, then no triangle was found.
		if tri != -1:
			break
	actualY = y
	return actualY


def findTopBottom(x, maxDimension, trifinder):
	topY = findTopY(x, maxDimension, trifinder)
	bottomY = findBottomY(x, maxDimension, trifinder)
	return [(x,topY), (x,bottomY)]