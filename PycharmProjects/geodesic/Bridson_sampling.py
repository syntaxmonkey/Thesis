# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import math
import Bridson_Common

def calculateParameters(xrange, yrange, radius=0, pointCount=0):
    perimeter = xrange * 2.0 + yrange * 2.0

    # Determine the number of points on the perimeter and the radius.
    if pointCount == 0 and radius == 0:
        Exception("Define etiher pointCount or radius")
    elif pointCount == 0:
        pointCount = math.ceil(perimeter / radius)
        pointDistance = perimeter / pointCount
    else:
        pointDistance = perimeter / pointCount

    return pointDistance, pointCount


def Bridson_sampling(width=1.0, height=1.0, radius=0.025, k=30, existingPoints=[], mask=[]):
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007
    def squared_distance(p0, p1):
        return (p0[0]-p1[0])**2 + (p0[1]-p1[1])**2

    def random_point_around(p, k=1):
        # WARNING: This is not uniform around p but we can live with it
        R = np.random.uniform(radius, 2*radius, k)
        T = np.random.uniform(0, 2*np.pi, k)
        P = np.empty((k, 2))
        P[:, 0] = p[0]+R*np.sin(T)
        P[:, 1] = p[1]+R*np.cos(T)
        return P

    def in_limits(p):
        return 0 <= p[0] < width and 0 <= p[1] < height

    def neighborhood(shape, index, n=2):
        row, col = index
        row0, row1 = max(row-n, 0), min(row+n+1, shape[0])
        col0, col1 = max(col-n, 0), min(col+n+1, shape[1])
        I = np.dstack(np.mgrid[row0:row1, col0:col1])
        I = I.reshape(I.size//2, 2).tolist()
        I.remove([row, col])
        return I

    def in_neighborhood(p):
        i, j = int(p[0]/cellsize), int(p[1]/cellsize)
        if M[i, j]:
            return True
        for (i, j) in N[(i, j)]:
            if M[i, j] and squared_distance(p, P[i, j]) < squared_radius:
                return True
        return False

    def add_point(p):
        points.append(p)
        i, j = int(p[0]/cellsize), int(p[1]/cellsize)
        P[i, j], M[i, j] = p, True

    def in_mask(p, mask):
        if len(mask) > 0:
            # Bridson_Common.logDebug(__name__,  "Length of Mask: ", p)
            # Will return true if the point references a pixel that has value 255.
            if mask[int(p[0]), int(p[1])] == 255:
                return True
        return False

    # Here `2` corresponds to the number of dimension
    cellsize = radius/np.sqrt(2)
    rows = int(np.ceil(width/cellsize))
    cols = int(np.ceil(height/cellsize))

    # Squared radius because we'll compare squared distance
    squared_radius = radius*radius

    # Positions cells
    P = np.zeros((rows, cols, 2), dtype=np.float32)
    M = np.zeros((rows, cols), dtype=bool)

    # Cache generation for neighborhood
    N = {}
    for i in range(rows):
        for j in range(cols):
            N[(i, j)] = neighborhood(M.shape, (i, j), 2)

    points = []

    # Add existing points to the list.
    for point in existingPoints:
        add_point(point)

    # add_point((np.random.uniform(width), np.random.uniform(height)))

    while len(points):
        i = np.random.randint(len(points))
        p = points[i]
        del points[i]
        Q = random_point_around(p, k)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q) and not in_mask(q, mask):
            # if in_limits(q) and not in_neighborhood(q):
                add_point(q)
    return P[M]


def displayPoints(points, xrange, yrange):
    plt.figure()
    plt.subplot(1, 1, 1, aspect=1)
    plt.title('Display Points')
    # dradius = math.sqrt(2)
    # xrange, yrange = 10, 10

    X = [x for (x, y) in points]
    Y = [y for (x, y) in points]
    plt.scatter(X, Y, s=10)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.xlim(-1, xrange+1)
    plt.ylim(-1, yrange+1)
    Bridson_Common.logDebug(__name__, points)
    # plt.show()


def genSquarePerimeterPoints(xrange, yrange, pointCount=0, radius=0):

    perimeter = xrange*2.0 + yrange*2.0

    pointDistance, pointCount = calculateParameters(xrange, yrange, radius=radius, pointCount=pointCount)

    Bridson_Common.logDebug(__name__, "Perimeter", perimeter)
    Bridson_Common.logDebug(__name__, "PointDistance", pointDistance)
    deltax = pointDistance
    deltay = 0
    fudge = 0.001
    currentx = fudge
    currenty = 0
    points=[]

    # Start at 0,0.
    points.append([currentx, currenty])

    # Add points along the first X axis.
    while currentx < xrange:
        points.append([currentx, currenty])
        currentx += pointDistance



    # Transition to the first Y axis.
    Bridson_Common.logDebug(__name__, "Transition to first y axis")
    Bridson_Common.logDebug(__name__, "currentx", currentx, "currenty", currenty)
    currenty = abs(xrange - currentx)
    Bridson_Common.logDebug(__name__, "newCurrentY", currenty)
    currentx = xrange - 0.01
    while currenty < yrange:
        points.append([currentx, currenty])
        currenty += pointDistance



    # Transition to the top X axis
    currentx = xrange - abs(yrange - currenty)
    currenty = yrange - fudge
    while currentx >= 0:
        points.append([currentx, currenty])
        currentx -= pointDistance




    # Transition to the final Y axis
    currenty = yrange - abs(0 - currentx)
    currentx = fudge
    while currenty > 0:
        points.append([currentx, currenty])
        currenty -= pointDistance

    # display the generated points.
    # displayPoints(points, xrange, yrange)
    return np.array(points)


if __name__ == '__main__':

    dradius = 2
    xrange, yrange = 10, 10

    points = genSquarePerimeterPoints(xrange, yrange, radius=dradius)
    Bridson_Common.logDebug(__name__, np.shape(points))
    points = Bridson_sampling(width=xrange, height=yrange, radius=dradius, existingPoints=points)
    Bridson_Common.logDebug(__name__, np.shape(points))
    displayPoints(points, xrange, yrange)
    plt.show()
