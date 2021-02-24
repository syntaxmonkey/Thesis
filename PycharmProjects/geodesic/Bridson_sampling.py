# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import math
import Bridson_Common
import datetime
from numba import prange, njit, jit, jit_module

@jit
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



def Bridson_sampling(width=1.0, height=1.0, radius=0.025, k=20, existingPoints=[], mask=[]):
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
        points.append(p.tolist())
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
    add_point( np.array([np.random.uniform(width), np.random.uniform(height)]) ) # Initial starting point seeding point.  For some reason it is required.
    add_point(np.array([np.random.uniform(width), np.random.uniform(height)]))  # Initial starting point seeding point.  For some reason it is required.

    # Add existing points to the list.
    for point in existingPoints:
        add_point(point)

    # add_point((np.random.uniform(width), np.random.uniform(height)))

    while len(points):
        i = np.random.randint(len(points))
        p = points[i]
        # print("Bridson Disc point:", p)
        del points[i]
        Q = random_point_around(p, k)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q) and not in_mask(q, mask):
            # if in_limits(q) and not in_neighborhood(q):
                add_point(q)
    return P[M]

@jit
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
    # Bridson_Common.logDebug(__name__, points)
    # plt.show()

@jit
def genSquarePerimeterPoints(xrange, yrange, pointCount=0, radius=0):

    perimeter = xrange*2.0 + yrange*2.0

    pointDistance, pointCount = calculateParameters(xrange, yrange, radius=radius, pointCount=pointCount)

    # Bridson_Common.logDebug(__name__, "Perimeter", perimeter)
    # Bridson_Common.logDebug(__name__, "PointDistance", pointDistance)
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
    # Bridson_Common.logDebug(__name__, "Transition to first y axis")
    # Bridson_Common.logDebug(__name__, "currentx", currentx, "currenty", currenty)
    currenty = abs(xrange - currentx)
    # Bridson_Common.logDebug(__name__, "newCurrentY", currenty)
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


def compareK(xrange, yrange):
    # Test quality of different k values.
    kMax = 100
    for kvalue in range(1,kMax):
        subtotal = 0
        totalTime = 0
        for i in range(5):
            points = np.array([])
            a = datetime.datetime.now()
            points = Bridson_sampling(width=xrange, height=yrange, radius=Bridson_Common.dradius, existingPoints=points, k=kvalue)
            b = datetime.datetime.now()
            print('dradius:', Bridson_Common.dradius, 'k:', kvalue, " -> point count:", len(points) , ' - ', datetime.datetime.now(), ' elapsed:', (b-a).microseconds, 'microseconds')
            totalTime += (b-a).microseconds
            subtotal += len(points)
        print('====== dradius:', Bridson_Common.dradius, 'k:', kvalue, " -> point count average:", subtotal/5, ' - ', datetime.datetime.now(), ' average elapsed:', totalTime / 5, 'microseconds')



if __name__ == '__main__':

    # print("=======================================================================================================")
    # xrange, yrange = 50, 50
    # print('x, y:', xrange, ',', yrange)
    # compareK(xrange, yrange)
    #
    #
    # print("=======================================================================================================")
    # dradius = 2
    # xrange, yrange = 200, 200
    # print('x, y:', xrange, ',', yrange)
    # compareK(xrange, yrange)

    dradius = 2
    xrange, yrange = 50, 50

    points = np.array([])

    a = datetime.datetime.now()
    points = genSquarePerimeterPoints(xrange, yrange, radius=dradius)
    # Bridson_Common.logDebug(__name__, np.shape(points))

    b = datetime.datetime.now()
    Bridson_Common.determineRadius(xrange, yrange)
    points = Bridson_sampling(width=xrange, height=yrange, radius=Bridson_Common.dradius, existingPoints=points, k=20)
    c = datetime.datetime.now()

    # Bridson_Common.logDebug(__name__, np.shape(points))
    displayPoints(points, xrange, yrange)
    d = datetime.datetime.now()
    print("Gen Square:", (b-a).microseconds)
    print("Disc Sampling:", (c-b).microseconds)
    print("Display Points:", (d-c).microseconds)
    plt.show()


'''
Results from performance runs:
Divisor: 3.0
=======================================================================================================
x, y: 50 , 50
dradius: 1.5 k: 1  -> point count: 8  -  2021-02-24 00:29:17.960158  elapsed: 110446 microseconds
dradius: 1.5 k: 1  -> point count: 9  -  2021-02-24 00:29:18.032559  elapsed: 72333 microseconds
dradius: 1.5 k: 1  -> point count: 8  -  2021-02-24 00:29:18.131882  elapsed: 99268 microseconds
dradius: 1.5 k: 1  -> point count: 12  -  2021-02-24 00:29:18.204269  elapsed: 72332 microseconds
dradius: 1.5 k: 1  -> point count: 5  -  2021-02-24 00:29:18.301532  elapsed: 97203 microseconds
====== dradius: 1.5 k: 1  -> point count average: 8.4  -  2021-02-24 00:29:18.301576  average elapsed: 90316.4 microseconds
dradius: 1.5 k: 2  -> point count: 82  -  2021-02-24 00:29:18.408129  elapsed: 106505 microseconds
dradius: 1.5 k: 2  -> point count: 61  -  2021-02-24 00:29:18.485095  elapsed: 76905 microseconds
dradius: 1.5 k: 2  -> point count: 121  -  2021-02-24 00:29:18.592434  elapsed: 107285 microseconds
dradius: 1.5 k: 2  -> point count: 125  -  2021-02-24 00:29:18.670189  elapsed: 77701 microseconds
dradius: 1.5 k: 2  -> point count: 179  -  2021-02-24 00:29:18.779965  elapsed: 109722 microseconds
====== dradius: 1.5 k: 2  -> point count average: 113.6  -  2021-02-24 00:29:18.780022  average elapsed: 95623.6 microseconds
dradius: 1.5 k: 3  -> point count: 385  -  2021-02-24 00:29:18.900166  elapsed: 120094 microseconds
dradius: 1.5 k: 3  -> point count: 489  -  2021-02-24 00:29:18.997586  elapsed: 97363 microseconds
dradius: 1.5 k: 3  -> point count: 478  -  2021-02-24 00:29:19.125302  elapsed: 127649 microseconds
dradius: 1.5 k: 3  -> point count: 472  -  2021-02-24 00:29:19.219494  elapsed: 94103 microseconds
dradius: 1.5 k: 3  -> point count: 442  -  2021-02-24 00:29:19.338022  elapsed: 118471 microseconds
====== dradius: 1.5 k: 3  -> point count average: 453.2  -  2021-02-24 00:29:19.338076  average elapsed: 111536.0 microseconds
dradius: 1.5 k: 4  -> point count: 526  -  2021-02-24 00:29:19.473525  elapsed: 135403 microseconds
dradius: 1.5 k: 4  -> point count: 541  -  2021-02-24 00:29:19.575649  elapsed: 102064 microseconds
dradius: 1.5 k: 4  -> point count: 535  -  2021-02-24 00:29:19.706779  elapsed: 131071 microseconds
dradius: 1.5 k: 4  -> point count: 544  -  2021-02-24 00:29:19.809371  elapsed: 102523 microseconds
dradius: 1.5 k: 4  -> point count: 539  -  2021-02-24 00:29:19.938335  elapsed: 128909 microseconds
====== dradius: 1.5 k: 4  -> point count average: 537.0  -  2021-02-24 00:29:19.938382  average elapsed: 119994.0 microseconds
dradius: 1.5 k: 5  -> point count: 590  -  2021-02-24 00:29:20.081564  elapsed: 143137 microseconds
dradius: 1.5 k: 5  -> point count: 597  -  2021-02-24 00:29:20.191645  elapsed: 109998 microseconds
dradius: 1.5 k: 5  -> point count: 594  -  2021-02-24 00:29:20.331062  elapsed: 139360 microseconds
dradius: 1.5 k: 5  -> point count: 588  -  2021-02-24 00:29:20.438525  elapsed: 107389 microseconds
dradius: 1.5 k: 5  -> point count: 589  -  2021-02-24 00:29:20.580604  elapsed: 142015 microseconds
====== dradius: 1.5 k: 5  -> point count average: 591.6  -  2021-02-24 00:29:20.580665  average elapsed: 128379.8 microseconds
dradius: 1.5 k: 6  -> point count: 609  -  2021-02-24 00:29:20.727365  elapsed: 146646 microseconds
dradius: 1.5 k: 6  -> point count: 605  -  2021-02-24 00:29:20.843565  elapsed: 116143 microseconds
dradius: 1.5 k: 6  -> point count: 618  -  2021-02-24 00:29:20.989461  elapsed: 145842 microseconds
dradius: 1.5 k: 6  -> point count: 606  -  2021-02-24 00:29:21.106105  elapsed: 116586 microseconds
dradius: 1.5 k: 6  -> point count: 593  -  2021-02-24 00:29:21.247908  elapsed: 141735 microseconds
====== dradius: 1.5 k: 6  -> point count average: 606.2  -  2021-02-24 00:29:21.248006  average elapsed: 133390.4 microseconds
dradius: 1.5 k: 7  -> point count: 623  -  2021-02-24 00:29:21.371469  elapsed: 123382 microseconds
dradius: 1.5 k: 7  -> point count: 616  -  2021-02-24 00:29:21.520587  elapsed: 149064 microseconds
dradius: 1.5 k: 7  -> point count: 608  -  2021-02-24 00:29:21.672588  elapsed: 151941 microseconds
dradius: 1.5 k: 7  -> point count: 624  -  2021-02-24 00:29:21.793200  elapsed: 120558 microseconds
dradius: 1.5 k: 7  -> point count: 631  -  2021-02-24 00:29:21.945230  elapsed: 151972 microseconds
====== dradius: 1.5 k: 7  -> point count average: 620.4  -  2021-02-24 00:29:21.945274  average elapsed: 139383.4 microseconds
dradius: 1.5 k: 8  -> point count: 620  -  2021-02-24 00:29:22.072757  elapsed: 127442 microseconds
dradius: 1.5 k: 8  -> point count: 623  -  2021-02-24 00:29:22.226320  elapsed: 153498 microseconds
dradius: 1.5 k: 8  -> point count: 643  -  2021-02-24 00:29:22.379511  elapsed: 153132 microseconds
dradius: 1.5 k: 8  -> point count: 646  -  2021-02-24 00:29:22.510051  elapsed: 130484 microseconds
dradius: 1.5 k: 8  -> point count: 621  -  2021-02-24 00:29:22.661722  elapsed: 151616 microseconds
====== dradius: 1.5 k: 8  -> point count average: 630.6  -  2021-02-24 00:29:22.661770  average elapsed: 143234.4 microseconds
dradius: 1.5 k: 9  -> point count: 657  -  2021-02-24 00:29:22.796685  elapsed: 134873 microseconds
dradius: 1.5 k: 9  -> point count: 630  -  2021-02-24 00:29:22.956073  elapsed: 159333 microseconds
dradius: 1.5 k: 9  -> point count: 635  -  2021-02-24 00:29:23.124894  elapsed: 168755 microseconds
dradius: 1.5 k: 9  -> point count: 650  -  2021-02-24 00:29:23.257630  elapsed: 132681 microseconds
dradius: 1.5 k: 9  -> point count: 648  -  2021-02-24 00:29:23.420619  elapsed: 162933 microseconds
====== dradius: 1.5 k: 9  -> point count average: 644.0  -  2021-02-24 00:29:23.420660  average elapsed: 151715.0 microseconds
dradius: 1.5 k: 10  -> point count: 651  -  2021-02-24 00:29:23.558600  elapsed: 137901 microseconds
dradius: 1.5 k: 10  -> point count: 652  -  2021-02-24 00:29:23.728368  elapsed: 169709 microseconds
dradius: 1.5 k: 10  -> point count: 653  -  2021-02-24 00:29:23.897327  elapsed: 168903 microseconds
dradius: 1.5 k: 10  -> point count: 660  -  2021-02-24 00:29:24.036941  elapsed: 139559 microseconds
dradius: 1.5 k: 10  -> point count: 653  -  2021-02-24 00:29:24.204164  elapsed: 167169 microseconds
====== dradius: 1.5 k: 10  -> point count average: 653.8  -  2021-02-24 00:29:24.204208  average elapsed: 156648.2 microseconds
dradius: 1.5 k: 11  -> point count: 655  -  2021-02-24 00:29:24.347817  elapsed: 143569 microseconds
dradius: 1.5 k: 11  -> point count: 646  -  2021-02-24 00:29:24.518809  elapsed: 170936 microseconds
dradius: 1.5 k: 11  -> point count: 658  -  2021-02-24 00:29:24.692371  elapsed: 173503 microseconds
dradius: 1.5 k: 11  -> point count: 653  -  2021-02-24 00:29:24.835275  elapsed: 142847 microseconds
dradius: 1.5 k: 11  -> point count: 663  -  2021-02-24 00:29:25.008990  elapsed: 173645 microseconds
====== dradius: 1.5 k: 11  -> point count average: 655.0  -  2021-02-24 00:29:25.009037  average elapsed: 160900.0 microseconds
dradius: 1.5 k: 12  -> point count: 666  -  2021-02-24 00:29:25.158852  elapsed: 149772 microseconds
dradius: 1.5 k: 12  -> point count: 656  -  2021-02-24 00:29:25.337121  elapsed: 178208 microseconds
dradius: 1.5 k: 12  -> point count: 667  -  2021-02-24 00:29:25.483404  elapsed: 146222 microseconds
dradius: 1.5 k: 12  -> point count: 668  -  2021-02-24 00:29:25.663495  elapsed: 180037 microseconds
dradius: 1.5 k: 12  -> point count: 674  -  2021-02-24 00:29:25.842714  elapsed: 179161 microseconds
====== dradius: 1.5 k: 12  -> point count average: 666.2  -  2021-02-24 00:29:25.842759  average elapsed: 166680.0 microseconds
dradius: 1.5 k: 13  -> point count: 668  -  2021-02-24 00:29:25.997778  elapsed: 154970 microseconds
dradius: 1.5 k: 13  -> point count: 654  -  2021-02-24 00:29:26.181031  elapsed: 183197 microseconds
dradius: 1.5 k: 13  -> point count: 666  -  2021-02-24 00:29:26.336737  elapsed: 155648 microseconds
dradius: 1.5 k: 13  -> point count: 659  -  2021-02-24 00:29:26.518609  elapsed: 181805 microseconds
dradius: 1.5 k: 13  -> point count: 680  -  2021-02-24 00:29:26.703069  elapsed: 184399 microseconds
====== dradius: 1.5 k: 13  -> point count average: 665.4  -  2021-02-24 00:29:26.703128  average elapsed: 172003.8 microseconds
dradius: 1.5 k: 14  -> point count: 677  -  2021-02-24 00:29:26.861769  elapsed: 158594 microseconds
dradius: 1.5 k: 14  -> point count: 665  -  2021-02-24 00:29:27.052600  elapsed: 190770 microseconds
dradius: 1.5 k: 14  -> point count: 673  -  2021-02-24 00:29:27.214876  elapsed: 162201 microseconds
dradius: 1.5 k: 14  -> point count: 667  -  2021-02-24 00:29:27.403761  elapsed: 188830 microseconds
dradius: 1.5 k: 14  -> point count: 678  -  2021-02-24 00:29:27.599385  elapsed: 195561 microseconds
====== dradius: 1.5 k: 14  -> point count average: 672.0  -  2021-02-24 00:29:27.599431  average elapsed: 179191.2 microseconds
dradius: 1.5 k: 15  -> point count: 668  -  2021-02-24 00:29:27.769289  elapsed: 169817 microseconds
dradius: 1.5 k: 15  -> point count: 671  -  2021-02-24 00:29:27.969409  elapsed: 200054 microseconds
dradius: 1.5 k: 15  -> point count: 671  -  2021-02-24 00:29:28.142835  elapsed: 173339 microseconds
dradius: 1.5 k: 15  -> point count: 675  -  2021-02-24 00:29:28.342498  elapsed: 199607 microseconds
dradius: 1.5 k: 15  -> point count: 672  -  2021-02-24 00:29:28.548026  elapsed: 205460 microseconds
====== dradius: 1.5 k: 15  -> point count average: 671.4  -  2021-02-24 00:29:28.548077  average elapsed: 189655.4 microseconds
dradius: 1.5 k: 16  -> point count: 679  -  2021-02-24 00:29:28.728287  elapsed: 180162 microseconds
dradius: 1.5 k: 16  -> point count: 685  -  2021-02-24 00:29:28.945032  elapsed: 216684 microseconds
dradius: 1.5 k: 16  -> point count: 675  -  2021-02-24 00:29:29.129059  elapsed: 183967 microseconds
dradius: 1.5 k: 16  -> point count: 674  -  2021-02-24 00:29:29.342435  elapsed: 213316 microseconds
dradius: 1.5 k: 16  -> point count: 683  -  2021-02-24 00:29:29.561437  elapsed: 218942 microseconds
====== dradius: 1.5 k: 16  -> point count average: 679.2  -  2021-02-24 00:29:29.561485  average elapsed: 202614.2 microseconds
dradius: 1.5 k: 17  -> point count: 687  -  2021-02-24 00:29:29.751972  elapsed: 190441 microseconds
dradius: 1.5 k: 17  -> point count: 693  -  2021-02-24 00:29:29.975446  elapsed: 223411 microseconds
dradius: 1.5 k: 17  -> point count: 687  -  2021-02-24 00:29:30.168857  elapsed: 193349 microseconds
dradius: 1.5 k: 17  -> point count: 682  -  2021-02-24 00:29:30.395614  elapsed: 226696 microseconds
dradius: 1.5 k: 17  -> point count: 694  -  2021-02-24 00:29:30.592623  elapsed: 196944 microseconds
====== dradius: 1.5 k: 17  -> point count average: 688.6  -  2021-02-24 00:29:30.592683  average elapsed: 206168.2 microseconds
dradius: 1.5 k: 18  -> point count: 700  -  2021-02-24 00:29:30.825685  elapsed: 232956 microseconds
dradius: 1.5 k: 18  -> point count: 678  -  2021-02-24 00:29:31.060907  elapsed: 235160 microseconds
dradius: 1.5 k: 18  -> point count: 679  -  2021-02-24 00:29:31.258754  elapsed: 197783 microseconds
dradius: 1.5 k: 18  -> point count: 695  -  2021-02-24 00:29:31.485936  elapsed: 227095 microseconds
dradius: 1.5 k: 18  -> point count: 688  -  2021-02-24 00:29:31.687454  elapsed: 201404 microseconds
====== dradius: 1.5 k: 18  -> point count average: 688.0  -  2021-02-24 00:29:31.687500  average elapsed: 218879.6 microseconds
dradius: 1.5 k: 19  -> point count: 703  -  2021-02-24 00:29:31.921731  elapsed: 234186 microseconds
dradius: 1.5 k: 19  -> point count: 691  -  2021-02-24 00:29:32.166819  elapsed: 245018 microseconds
dradius: 1.5 k: 19  -> point count: 695  -  2021-02-24 00:29:32.377927  elapsed: 211044 microseconds
dradius: 1.5 k: 19  -> point count: 694  -  2021-02-24 00:29:32.622562  elapsed: 244569 microseconds
dradius: 1.5 k: 19  -> point count: 693  -  2021-02-24 00:29:32.834655  elapsed: 212021 microseconds
====== dradius: 1.5 k: 19  -> point count average: 695.2  -  2021-02-24 00:29:32.834706  average elapsed: 229367.6 microseconds
dradius: 1.5 k: 20  -> point count: 690  -  2021-02-24 00:29:33.092022  elapsed: 257255 microseconds
dradius: 1.5 k: 20  -> point count: 707  -  2021-02-24 00:29:33.343802  elapsed: 251703 microseconds
dradius: 1.5 k: 20  -> point count: 689  -  2021-02-24 00:29:33.559543  elapsed: 215675 microseconds
dradius: 1.5 k: 20  -> point count: 687  -  2021-02-24 00:29:33.807299  elapsed: 247691 microseconds
dradius: 1.5 k: 20  -> point count: 688  -  2021-02-24 00:29:34.025616  elapsed: 218243 microseconds
====== dradius: 1.5 k: 20  -> point count average: 692.2  -  2021-02-24 00:29:34.025678  average elapsed: 238113.4 microseconds
dradius: 1.5 k: 21  -> point count: 702  -  2021-02-24 00:29:34.283133  elapsed: 257404 microseconds
dradius: 1.5 k: 21  -> point count: 702  -  2021-02-24 00:29:34.543451  elapsed: 260257 microseconds
dradius: 1.5 k: 21  -> point count: 692  -  2021-02-24 00:29:34.774308  elapsed: 230784 microseconds
dradius: 1.5 k: 21  -> point count: 691  -  2021-02-24 00:29:35.038219  elapsed: 263820 microseconds
dradius: 1.5 k: 21  -> point count: 693  -  2021-02-24 00:29:35.271911  elapsed: 233622 microseconds
====== dradius: 1.5 k: 21  -> point count average: 696.0  -  2021-02-24 00:29:35.271961  average elapsed: 249177.4 microseconds
dradius: 1.5 k: 22  -> point count: 692  -  2021-02-24 00:29:35.542365  elapsed: 270358 microseconds
dradius: 1.5 k: 22  -> point count: 680  -  2021-02-24 00:29:35.813098  elapsed: 270665 microseconds
dradius: 1.5 k: 22  -> point count: 693  -  2021-02-24 00:29:36.048685  elapsed: 235522 microseconds
dradius: 1.5 k: 22  -> point count: 683  -  2021-02-24 00:29:36.316746  elapsed: 267995 microseconds
dradius: 1.5 k: 22  -> point count: 715  -  2021-02-24 00:29:36.560593  elapsed: 243778 microseconds
====== dradius: 1.5 k: 22  -> point count average: 692.6  -  2021-02-24 00:29:36.560652  average elapsed: 257663.6 microseconds
dradius: 1.5 k: 23  -> point count: 707  -  2021-02-24 00:29:36.836844  elapsed: 276137 microseconds
dradius: 1.5 k: 23  -> point count: 703  -  2021-02-24 00:29:37.119006  elapsed: 282086 microseconds
dradius: 1.5 k: 23  -> point count: 701  -  2021-02-24 00:29:37.362468  elapsed: 243399 microseconds
dradius: 1.5 k: 23  -> point count: 690  -  2021-02-24 00:29:37.640121  elapsed: 277589 microseconds
dradius: 1.5 k: 23  -> point count: 704  -  2021-02-24 00:29:37.888039  elapsed: 247854 microseconds
====== dradius: 1.5 k: 23  -> point count average: 701.0  -  2021-02-24 00:29:37.888094  average elapsed: 265413.0 microseconds
dradius: 1.5 k: 24  -> point count: 709  -  2021-02-24 00:29:38.173716  elapsed: 285572 microseconds
dradius: 1.5 k: 24  -> point count: 693  -  2021-02-24 00:29:38.423352  elapsed: 249569 microseconds
dradius: 1.5 k: 24  -> point count: 702  -  2021-02-24 00:29:38.709190  elapsed: 285739 microseconds
dradius: 1.5 k: 24  -> point count: 702  -  2021-02-24 00:29:38.993694  elapsed: 284439 microseconds
dradius: 1.5 k: 24  -> point count: 705  -  2021-02-24 00:29:39.250938  elapsed: 257179 microseconds
====== dradius: 1.5 k: 24  -> point count average: 702.2  -  2021-02-24 00:29:39.250989  average elapsed: 272499.6 microseconds
dradius: 1.5 k: 25  -> point count: 699  -  2021-02-24 00:29:39.544389  elapsed: 293351 microseconds
dradius: 1.5 k: 25  -> point count: 686  -  2021-02-24 00:29:39.804046  elapsed: 259587 microseconds
dradius: 1.5 k: 25  -> point count: 701  -  2021-02-24 00:29:40.089461  elapsed: 285351 microseconds
dradius: 1.5 k: 25  -> point count: 690  -  2021-02-24 00:29:40.374758  elapsed: 285230 microseconds
dradius: 1.5 k: 25  -> point count: 708  -  2021-02-24 00:29:40.634780  elapsed: 259957 microseconds
====== dradius: 1.5 k: 25  -> point count average: 696.8  -  2021-02-24 00:29:40.634839  average elapsed: 276695.2 microseconds
dradius: 1.5 k: 26  -> point count: 705  -  2021-02-24 00:29:40.926333  elapsed: 291441 microseconds
dradius: 1.5 k: 26  -> point count: 720  -  2021-02-24 00:29:41.196648  elapsed: 270250 microseconds
dradius: 1.5 k: 26  -> point count: 704  -  2021-02-24 00:29:41.488173  elapsed: 291460 microseconds
dradius: 1.5 k: 26  -> point count: 709  -  2021-02-24 00:29:41.787445  elapsed: 299206 microseconds
dradius: 1.5 k: 26  -> point count: 700  -  2021-02-24 00:29:42.052099  elapsed: 264589 microseconds
====== dradius: 1.5 k: 26  -> point count average: 707.6  -  2021-02-24 00:29:42.052150  average elapsed: 283389.2 microseconds
dradius: 1.5 k: 27  -> point count: 717  -  2021-02-24 00:29:42.357909  elapsed: 305710 microseconds
dradius: 1.5 k: 27  -> point count: 694  -  2021-02-24 00:29:42.623837  elapsed: 265865 microseconds
dradius: 1.5 k: 27  -> point count: 704  -  2021-02-24 00:29:42.922166  elapsed: 298264 microseconds
dradius: 1.5 k: 27  -> point count: 700  -  2021-02-24 00:29:43.226307  elapsed: 304074 microseconds
dradius: 1.5 k: 27  -> point count: 708  -  2021-02-24 00:29:43.494415  elapsed: 268044 microseconds
====== dradius: 1.5 k: 27  -> point count average: 704.6  -  2021-02-24 00:29:43.494470  average elapsed: 288391.4 microseconds
dradius: 1.5 k: 28  -> point count: 710  -  2021-02-24 00:29:43.805541  elapsed: 311021 microseconds
dradius: 1.5 k: 28  -> point count: 722  -  2021-02-24 00:29:44.083760  elapsed: 278154 microseconds
dradius: 1.5 k: 28  -> point count: 692  -  2021-02-24 00:29:44.385847  elapsed: 302016 microseconds
dradius: 1.5 k: 28  -> point count: 702  -  2021-02-24 00:29:44.696733  elapsed: 310816 microseconds
dradius: 1.5 k: 28  -> point count: 695  -  2021-02-24 00:29:44.964460  elapsed: 267666 microseconds
====== dradius: 1.5 k: 28  -> point count average: 704.2  -  2021-02-24 00:29:44.964511  average elapsed: 293934.6 microseconds
dradius: 1.5 k: 29  -> point count: 718  -  2021-02-24 00:29:45.283295  elapsed: 318736 microseconds
dradius: 1.5 k: 29  -> point count: 705  -  2021-02-24 00:29:45.561746  elapsed: 278381 microseconds
dradius: 1.5 k: 29  -> point count: 707  -  2021-02-24 00:29:45.874559  elapsed: 312748 microseconds
dradius: 1.5 k: 29  -> point count: 712  -  2021-02-24 00:29:46.195755  elapsed: 321128 microseconds
dradius: 1.5 k: 29  -> point count: 721  -  2021-02-24 00:29:46.476982  elapsed: 281161 microseconds
====== dradius: 1.5 k: 29  -> point count average: 712.6  -  2021-02-24 00:29:46.477031  average elapsed: 302430.8 microseconds
dradius: 1.5 k: 30  -> point count: 717  -  2021-02-24 00:29:46.801346  elapsed: 324266 microseconds
dradius: 1.5 k: 30  -> point count: 712  -  2021-02-24 00:29:47.093339  elapsed: 291927 microseconds
dradius: 1.5 k: 30  -> point count: 707  -  2021-02-24 00:29:47.408251  elapsed: 314847 microseconds
dradius: 1.5 k: 30  -> point count: 709  -  2021-02-24 00:29:47.696340  elapsed: 288021 microseconds
dradius: 1.5 k: 30  -> point count: 710  -  2021-02-24 00:29:48.012659  elapsed: 316251 microseconds
====== dradius: 1.5 k: 30  -> point count average: 711.0  -  2021-02-24 00:29:48.012712  average elapsed: 307062.4 microseconds
dradius: 1.5 k: 31  -> point count: 707  -  2021-02-24 00:29:48.348548  elapsed: 335790 microseconds
dradius: 1.5 k: 31  -> point count: 718  -  2021-02-24 00:29:48.653984  elapsed: 305370 microseconds
dradius: 1.5 k: 31  -> point count: 702  -  2021-02-24 00:29:48.981642  elapsed: 327578 microseconds
dradius: 1.5 k: 31  -> point count: 719  -  2021-02-24 00:29:49.283858  elapsed: 302152 microseconds
dradius: 1.5 k: 31  -> point count: 702  -  2021-02-24 00:29:49.602423  elapsed: 318498 microseconds
====== dradius: 1.5 k: 31  -> point count average: 709.6  -  2021-02-24 00:29:49.602479  average elapsed: 317877.6 microseconds
dradius: 1.5 k: 32  -> point count: 721  -  2021-02-24 00:29:49.945098  elapsed: 342569 microseconds
dradius: 1.5 k: 32  -> point count: 707  -  2021-02-24 00:29:50.248134  elapsed: 302974 microseconds
dradius: 1.5 k: 32  -> point count: 715  -  2021-02-24 00:29:50.582530  elapsed: 334333 microseconds
dradius: 1.5 k: 32  -> point count: 703  -  2021-02-24 00:29:50.882164  elapsed: 299567 microseconds
dradius: 1.5 k: 32  -> point count: 706  -  2021-02-24 00:29:51.208682  elapsed: 326452 microseconds
====== dradius: 1.5 k: 32  -> point count average: 710.4  -  2021-02-24 00:29:51.208731  average elapsed: 321179.0 microseconds
dradius: 1.5 k: 33  -> point count: 716  -  2021-02-24 00:29:51.547582  elapsed: 338803 microseconds
dradius: 1.5 k: 33  -> point count: 714  -  2021-02-24 00:29:51.858109  elapsed: 310449 microseconds
dradius: 1.5 k: 33  -> point count: 713  -  2021-02-24 00:29:52.197468  elapsed: 339271 microseconds
dradius: 1.5 k: 33  -> point count: 717  -  2021-02-24 00:29:52.505867  elapsed: 308337 microseconds
dradius: 1.5 k: 33  -> point count: 720  -  2021-02-24 00:29:52.844553  elapsed: 338619 microseconds
====== dradius: 1.5 k: 33  -> point count average: 716.0  -  2021-02-24 00:29:52.844606  average elapsed: 327095.8 microseconds
dradius: 1.5 k: 34  -> point count: 710  -  2021-02-24 00:29:53.192293  elapsed: 347638 microseconds
dradius: 1.5 k: 34  -> point count: 717  -  2021-02-24 00:29:53.504208  elapsed: 311838 microseconds
dradius: 1.5 k: 34  -> point count: 720  -  2021-02-24 00:29:53.853698  elapsed: 349426 microseconds
dradius: 1.5 k: 34  -> point count: 716  -  2021-02-24 00:29:54.167668  elapsed: 313906 microseconds
dradius: 1.5 k: 34  -> point count: 703  -  2021-02-24 00:29:54.514838  elapsed: 347082 microseconds
====== dradius: 1.5 k: 34  -> point count average: 713.2  -  2021-02-24 00:29:54.514892  average elapsed: 333978.0 microseconds
dradius: 1.5 k: 35  -> point count: 703  -  2021-02-24 00:29:54.874762  elapsed: 359823 microseconds
dradius: 1.5 k: 35  -> point count: 718  -  2021-02-24 00:29:55.200377  elapsed: 325550 microseconds
dradius: 1.5 k: 35  -> point count: 712  -  2021-02-24 00:29:55.558183  elapsed: 357743 microseconds
dradius: 1.5 k: 35  -> point count: 718  -  2021-02-24 00:29:55.882622  elapsed: 324372 microseconds
dradius: 1.5 k: 35  -> point count: 709  -  2021-02-24 00:29:56.231020  elapsed: 348292 microseconds
====== dradius: 1.5 k: 35  -> point count average: 712.0  -  2021-02-24 00:29:56.231073  average elapsed: 343156.0 microseconds
dradius: 1.5 k: 36  -> point count: 725  -  2021-02-24 00:29:56.574519  elapsed: 343399 microseconds
dradius: 1.5 k: 36  -> point count: 713  -  2021-02-24 00:29:56.940307  elapsed: 365722 microseconds
dradius: 1.5 k: 36  -> point count: 713  -  2021-02-24 00:29:57.310752  elapsed: 370376 microseconds
dradius: 1.5 k: 36  -> point count: 714  -  2021-02-24 00:29:57.648027  elapsed: 337196 microseconds
dradius: 1.5 k: 36  -> point count: 718  -  2021-02-24 00:29:58.022018  elapsed: 373927 microseconds
====== dradius: 1.5 k: 36  -> point count average: 716.6  -  2021-02-24 00:29:58.022069  average elapsed: 358124.0 microseconds
dradius: 1.5 k: 37  -> point count: 720  -  2021-02-24 00:29:58.366688  elapsed: 344569 microseconds
dradius: 1.5 k: 37  -> point count: 720  -  2021-02-24 00:29:58.742589  elapsed: 375825 microseconds
dradius: 1.5 k: 37  -> point count: 726  -  2021-02-24 00:29:59.124249  elapsed: 381588 microseconds
dradius: 1.5 k: 37  -> point count: 723  -  2021-02-24 00:29:59.470400  elapsed: 346075 microseconds
dradius: 1.5 k: 37  -> point count: 725  -  2021-02-24 00:29:59.851233  elapsed: 380769 microseconds
====== dradius: 1.5 k: 37  -> point count average: 722.8  -  2021-02-24 00:29:59.851286  average elapsed: 365765.2 microseconds
dradius: 1.5 k: 38  -> point count: 709  -  2021-02-24 00:30:00.205947  elapsed: 354612 microseconds
dradius: 1.5 k: 38  -> point count: 730  -  2021-02-24 00:30:00.589373  elapsed: 383363 microseconds
dradius: 1.5 k: 38  -> point count: 711  -  2021-02-24 00:30:00.972788  elapsed: 383344 microseconds
dradius: 1.5 k: 38  -> point count: 726  -  2021-02-24 00:30:01.326401  elapsed: 353542 microseconds
dradius: 1.5 k: 38  -> point count: 732  -  2021-02-24 00:30:01.752286  elapsed: 425819 microseconds
====== dradius: 1.5 k: 38  -> point count average: 721.6  -  2021-02-24 00:30:01.752340  average elapsed: 380136.0 microseconds
dradius: 1.5 k: 39  -> point count: 716  -  2021-02-24 00:30:02.116895  elapsed: 364505 microseconds
dradius: 1.5 k: 39  -> point count: 730  -  2021-02-24 00:30:02.525364  elapsed: 408393 microseconds
dradius: 1.5 k: 39  -> point count: 708  -  2021-02-24 00:30:02.919924  elapsed: 394495 microseconds
dradius: 1.5 k: 39  -> point count: 728  -  2021-02-24 00:30:03.280089  elapsed: 360097 microseconds
dradius: 1.5 k: 39  -> point count: 726  -  2021-02-24 00:30:03.670952  elapsed: 390787 microseconds
====== dradius: 1.5 k: 39  -> point count average: 721.6  -  2021-02-24 00:30:03.671024  average elapsed: 383655.4 microseconds
dradius: 1.5 k: 40  -> point count: 729  -  2021-02-24 00:30:04.039848  elapsed: 368769 microseconds
dradius: 1.5 k: 40  -> point count: 718  -  2021-02-24 00:30:04.430183  elapsed: 390270 microseconds
dradius: 1.5 k: 40  -> point count: 721  -  2021-02-24 00:30:04.831427  elapsed: 401175 microseconds
dradius: 1.5 k: 40  -> point count: 720  -  2021-02-24 00:30:05.219712  elapsed: 388221 microseconds
dradius: 1.5 k: 40  -> point count: 719  -  2021-02-24 00:30:05.613314  elapsed: 393539 microseconds
====== dradius: 1.5 k: 40  -> point count average: 721.4  -  2021-02-24 00:30:05.613367  average elapsed: 388394.8 microseconds
dradius: 1.5 k: 41  -> point count: 731  -  2021-02-24 00:30:05.990804  elapsed: 377388 microseconds
dradius: 1.5 k: 41  -> point count: 722  -  2021-02-24 00:30:06.394124  elapsed: 403232 microseconds
dradius: 1.5 k: 41  -> point count: 723  -  2021-02-24 00:30:06.796373  elapsed: 402172 microseconds
dradius: 1.5 k: 41  -> point count: 719  -  2021-02-24 00:30:07.164439  elapsed: 368003 microseconds
dradius: 1.5 k: 41  -> point count: 723  -  2021-02-24 00:30:07.568953  elapsed: 404446 microseconds
====== dradius: 1.5 k: 41  -> point count average: 723.6  -  2021-02-24 00:30:07.569019  average elapsed: 391048.2 microseconds
dradius: 1.5 k: 42  -> point count: 730  -  2021-02-24 00:30:07.952233  elapsed: 383160 microseconds
dradius: 1.5 k: 42  -> point count: 733  -  2021-02-24 00:30:08.367735  elapsed: 415416 microseconds
dradius: 1.5 k: 42  -> point count: 724  -  2021-02-24 00:30:08.785692  elapsed: 417890 microseconds
dradius: 1.5 k: 42  -> point count: 715  -  2021-02-24 00:30:09.160575  elapsed: 374815 microseconds
dradius: 1.5 k: 42  -> point count: 720  -  2021-02-24 00:30:09.574128  elapsed: 413488 microseconds
====== dradius: 1.5 k: 42  -> point count average: 724.4  -  2021-02-24 00:30:09.574188  average elapsed: 400953.8 microseconds
dradius: 1.5 k: 43  -> point count: 724  -  2021-02-24 00:30:09.969036  elapsed: 394797 microseconds
dradius: 1.5 k: 43  -> point count: 730  -  2021-02-24 00:30:10.385888  elapsed: 416788 microseconds
dradius: 1.5 k: 43  -> point count: 714  -  2021-02-24 00:30:10.758559  elapsed: 372604 microseconds
dradius: 1.5 k: 43  -> point count: 724  -  2021-02-24 00:30:11.174545  elapsed: 415912 microseconds
dradius: 1.5 k: 43  -> point count: 723  -  2021-02-24 00:30:11.588639  elapsed: 414020 microseconds
====== dradius: 1.5 k: 43  -> point count average: 723.0  -  2021-02-24 00:30:11.588700  average elapsed: 402824.2 microseconds
dradius: 1.5 k: 44  -> point count: 724  -  2021-02-24 00:30:11.980014  elapsed: 391262 microseconds
dradius: 1.5 k: 44  -> point count: 725  -  2021-02-24 00:30:12.405039  elapsed: 424945 microseconds
dradius: 1.5 k: 44  -> point count: 723  -  2021-02-24 00:30:12.795462  elapsed: 390354 microseconds
dradius: 1.5 k: 44  -> point count: 725  -  2021-02-24 00:30:13.210561  elapsed: 415029 microseconds
dradius: 1.5 k: 44  -> point count: 725  -  2021-02-24 00:30:13.628911  elapsed: 418228 microseconds
====== dradius: 1.5 k: 44  -> point count average: 724.4  -  2021-02-24 00:30:13.628965  average elapsed: 407963.6 microseconds
dradius: 1.5 k: 45  -> point count: 733  -  2021-02-24 00:30:14.035078  elapsed: 406062 microseconds
dradius: 1.5 k: 45  -> point count: 716  -  2021-02-24 00:30:14.454650  elapsed: 419507 microseconds
dradius: 1.5 k: 45  -> point count: 723  -  2021-02-24 00:30:14.853808  elapsed: 399079 microseconds
dradius: 1.5 k: 45  -> point count: 724  -  2021-02-24 00:30:15.282936  elapsed: 429062 microseconds
dradius: 1.5 k: 45  -> point count: 724  -  2021-02-24 00:30:15.710329  elapsed: 427325 microseconds
====== dradius: 1.5 k: 45  -> point count average: 724.0  -  2021-02-24 00:30:15.710420  average elapsed: 416207.0 microseconds
dradius: 1.5 k: 46  -> point count: 720  -  2021-02-24 00:30:16.117147  elapsed: 406666 microseconds
dradius: 1.5 k: 46  -> point count: 725  -  2021-02-24 00:30:16.532222  elapsed: 415009 microseconds
dradius: 1.5 k: 46  -> point count: 723  -  2021-02-24 00:30:16.921625  elapsed: 389339 microseconds
dradius: 1.5 k: 46  -> point count: 738  -  2021-02-24 00:30:17.351635  elapsed: 429945 microseconds
dradius: 1.5 k: 46  -> point count: 727  -  2021-02-24 00:30:17.779505  elapsed: 427804 microseconds
====== dradius: 1.5 k: 46  -> point count average: 726.6  -  2021-02-24 00:30:17.779561  average elapsed: 413752.6 microseconds
dradius: 1.5 k: 47  -> point count: 717  -  2021-02-24 00:30:18.178538  elapsed: 398925 microseconds
dradius: 1.5 k: 47  -> point count: 730  -  2021-02-24 00:30:18.600508  elapsed: 421904 microseconds
dradius: 1.5 k: 47  -> point count: 734  -  2021-02-24 00:30:18.998722  elapsed: 398139 microseconds
dradius: 1.5 k: 47  -> point count: 725  -  2021-02-24 00:30:19.418792  elapsed: 420003 microseconds
dradius: 1.5 k: 47  -> point count: 720  -  2021-02-24 00:30:19.840872  elapsed: 422016 microseconds
====== dradius: 1.5 k: 47  -> point count average: 725.2  -  2021-02-24 00:30:19.840925  average elapsed: 412197.4 microseconds
dradius: 1.5 k: 48  -> point count: 728  -  2021-02-24 00:30:20.242780  elapsed: 401799 microseconds
dradius: 1.5 k: 48  -> point count: 716  -  2021-02-24 00:30:20.676170  elapsed: 433324 microseconds
dradius: 1.5 k: 48  -> point count: 732  -  2021-02-24 00:30:21.075812  elapsed: 399578 microseconds
dradius: 1.5 k: 48  -> point count: 721  -  2021-02-24 00:30:21.525901  elapsed: 450024 microseconds
dradius: 1.5 k: 48  -> point count: 731  -  2021-02-24 00:30:21.933044  elapsed: 407074 microseconds
====== dradius: 1.5 k: 48  -> point count average: 725.6  -  2021-02-24 00:30:21.933129  average elapsed: 418359.8 microseconds
dradius: 1.5 k: 49  -> point count: 724  -  2021-02-24 00:30:22.367717  elapsed: 434511 microseconds
dradius: 1.5 k: 49  -> point count: 723  -  2021-02-24 00:30:22.808289  elapsed: 440495 microseconds
dradius: 1.5 k: 49  -> point count: 734  -  2021-02-24 00:30:23.221248  elapsed: 412893 microseconds
dradius: 1.5 k: 49  -> point count: 738  -  2021-02-24 00:30:23.658617  elapsed: 437289 microseconds
dradius: 1.5 k: 49  -> point count: 735  -  2021-02-24 00:30:24.074882  elapsed: 416190 microseconds
====== dradius: 1.5 k: 49  -> point count average: 730.8  -  2021-02-24 00:30:24.074934  average elapsed: 428275.6 microseconds
dradius: 1.5 k: 50  -> point count: 740  -  2021-02-24 00:30:24.525042  elapsed: 450060 microseconds
dradius: 1.5 k: 50  -> point count: 721  -  2021-02-24 00:30:24.971741  elapsed: 446634 microseconds
dradius: 1.5 k: 50  -> point count: 746  -  2021-02-24 00:30:25.400680  elapsed: 428874 microseconds
dradius: 1.5 k: 50  -> point count: 733  -  2021-02-24 00:30:25.849052  elapsed: 448304 microseconds
dradius: 1.5 k: 50  -> point count: 722  -  2021-02-24 00:30:26.264528  elapsed: 415412 microseconds
====== dradius: 1.5 k: 50  -> point count average: 732.4  -  2021-02-24 00:30:26.264578  average elapsed: 437856.8 microseconds
dradius: 1.5 k: 51  -> point count: 738  -  2021-02-24 00:30:26.727000  elapsed: 462376 microseconds
dradius: 1.5 k: 51  -> point count: 732  -  2021-02-24 00:30:27.186762  elapsed: 459692 microseconds
dradius: 1.5 k: 51  -> point count: 721  -  2021-02-24 00:30:27.606184  elapsed: 419322 microseconds
dradius: 1.5 k: 51  -> point count: 731  -  2021-02-24 00:30:28.061676  elapsed: 455428 microseconds
dradius: 1.5 k: 51  -> point count: 734  -  2021-02-24 00:30:28.486632  elapsed: 424891 microseconds
====== dradius: 1.5 k: 51  -> point count average: 731.2  -  2021-02-24 00:30:28.486685  average elapsed: 444341.8 microseconds
dradius: 1.5 k: 52  -> point count: 741  -  2021-02-24 00:30:28.945607  elapsed: 458874 microseconds
dradius: 1.5 k: 52  -> point count: 741  -  2021-02-24 00:30:29.419175  elapsed: 473502 microseconds
dradius: 1.5 k: 52  -> point count: 729  -  2021-02-24 00:30:29.846813  elapsed: 427572 microseconds
dradius: 1.5 k: 52  -> point count: 729  -  2021-02-24 00:30:30.306461  elapsed: 459584 microseconds
dradius: 1.5 k: 52  -> point count: 718  -  2021-02-24 00:30:30.731394  elapsed: 424855 microseconds
====== dradius: 1.5 k: 52  -> point count average: 731.6  -  2021-02-24 00:30:30.731443  average elapsed: 448877.4 microseconds
dradius: 1.5 k: 53  -> point count: 721  -  2021-02-24 00:30:31.191041  elapsed: 459550 microseconds
dradius: 1.5 k: 53  -> point count: 742  -  2021-02-24 00:30:31.669399  elapsed: 478283 microseconds
dradius: 1.5 k: 53  -> point count: 739  -  2021-02-24 00:30:32.109436  elapsed: 439970 microseconds
dradius: 1.5 k: 53  -> point count: 740  -  2021-02-24 00:30:32.575175  elapsed: 465672 microseconds
dradius: 1.5 k: 53  -> point count: 723  -  2021-02-24 00:30:33.003331  elapsed: 428091 microseconds
====== dradius: 1.5 k: 53  -> point count average: 733.0  -  2021-02-24 00:30:33.003384  average elapsed: 454313.2 microseconds
dradius: 1.5 k: 54  -> point count: 724  -  2021-02-24 00:30:33.468064  elapsed: 464632 microseconds
dradius: 1.5 k: 54  -> point count: 733  -  2021-02-24 00:30:33.939688  elapsed: 471560 microseconds
dradius: 1.5 k: 54  -> point count: 715  -  2021-02-24 00:30:34.367339  elapsed: 427588 microseconds
dradius: 1.5 k: 54  -> point count: 743  -  2021-02-24 00:30:34.841118  elapsed: 473712 microseconds
dradius: 1.5 k: 54  -> point count: 724  -  2021-02-24 00:30:35.275686  elapsed: 434500 microseconds
====== dradius: 1.5 k: 54  -> point count average: 727.8  -  2021-02-24 00:30:35.275752  average elapsed: 454398.4 microseconds
dradius: 1.5 k: 55  -> point count: 726  -  2021-02-24 00:30:35.745171  elapsed: 469362 microseconds
dradius: 1.5 k: 55  -> point count: 741  -  2021-02-24 00:30:36.233026  elapsed: 487790 microseconds
dradius: 1.5 k: 55  -> point count: 731  -  2021-02-24 00:30:36.684171  elapsed: 451076 microseconds
dradius: 1.5 k: 55  -> point count: 728  -  2021-02-24 00:30:37.173732  elapsed: 489498 microseconds
dradius: 1.5 k: 55  -> point count: 726  -  2021-02-24 00:30:37.622635  elapsed: 448839 microseconds
====== dradius: 1.5 k: 55  -> point count average: 730.4  -  2021-02-24 00:30:37.622687  average elapsed: 469313.0 microseconds
dradius: 1.5 k: 56  -> point count: 740  -  2021-02-24 00:30:38.109757  elapsed: 487022 microseconds
dradius: 1.5 k: 56  -> point count: 736  -  2021-02-24 00:30:38.556798  elapsed: 446974 microseconds
dradius: 1.5 k: 56  -> point count: 734  -  2021-02-24 00:30:39.029335  elapsed: 472472 microseconds
dradius: 1.5 k: 56  -> point count: 731  -  2021-02-24 00:30:39.512874  elapsed: 483464 microseconds
dradius: 1.5 k: 56  -> point count: 736  -  2021-02-24 00:30:39.972618  elapsed: 459673 microseconds
====== dradius: 1.5 k: 56  -> point count average: 735.4  -  2021-02-24 00:30:39.972671  average elapsed: 469921.0 microseconds
dradius: 1.5 k: 57  -> point count: 736  -  2021-02-24 00:30:40.461870  elapsed: 489152 microseconds
dradius: 1.5 k: 57  -> point count: 736  -  2021-02-24 00:30:40.919311  elapsed: 457376 microseconds
dradius: 1.5 k: 57  -> point count: 735  -  2021-02-24 00:30:41.400870  elapsed: 481498 microseconds
dradius: 1.5 k: 57  -> point count: 736  -  2021-02-24 00:30:41.895320  elapsed: 494384 microseconds
dradius: 1.5 k: 57  -> point count: 738  -  2021-02-24 00:30:42.346591  elapsed: 451207 microseconds
====== dradius: 1.5 k: 57  -> point count average: 736.2  -  2021-02-24 00:30:42.346647  average elapsed: 474723.4 microseconds
dradius: 1.5 k: 58  -> point count: 727  -  2021-02-24 00:30:42.829853  elapsed: 483156 microseconds
dradius: 1.5 k: 58  -> point count: 734  -  2021-02-24 00:30:43.281215  elapsed: 451303 microseconds
dradius: 1.5 k: 58  -> point count: 731  -  2021-02-24 00:30:43.754208  elapsed: 472919 microseconds
dradius: 1.5 k: 58  -> point count: 729  -  2021-02-24 00:30:44.232353  elapsed: 478081 microseconds
dradius: 1.5 k: 58  -> point count: 749  -  2021-02-24 00:30:44.697419  elapsed: 464994 microseconds
====== dradius: 1.5 k: 58  -> point count average: 734.0  -  2021-02-24 00:30:44.697468  average elapsed: 470090.6 microseconds
dradius: 1.5 k: 59  -> point count: 737  -  2021-02-24 00:30:45.177789  elapsed: 480273 microseconds
dradius: 1.5 k: 59  -> point count: 721  -  2021-02-24 00:30:45.627578  elapsed: 449722 microseconds
dradius: 1.5 k: 59  -> point count: 744  -  2021-02-24 00:30:46.117894  elapsed: 490248 microseconds
dradius: 1.5 k: 59  -> point count: 730  -  2021-02-24 00:30:46.608573  elapsed: 490609 microseconds
dradius: 1.5 k: 59  -> point count: 724  -  2021-02-24 00:30:47.073888  elapsed: 465250 microseconds
====== dradius: 1.5 k: 59  -> point count average: 731.2  -  2021-02-24 00:30:47.073943  average elapsed: 475220.4 microseconds
dradius: 1.5 k: 60  -> point count: 740  -  2021-02-24 00:30:47.586395  elapsed: 512401 microseconds
dradius: 1.5 k: 60  -> point count: 735  -  2021-02-24 00:30:48.076588  elapsed: 490127 microseconds
dradius: 1.5 k: 60  -> point count: 736  -  2021-02-24 00:30:48.604642  elapsed: 527988 microseconds
dradius: 1.5 k: 60  -> point count: 726  -  2021-02-24 00:30:49.129457  elapsed: 524729 microseconds
dradius: 1.5 k: 60  -> point count: 725  -  2021-02-24 00:30:49.635272  elapsed: 505747 microseconds
====== dradius: 1.5 k: 60  -> point count average: 732.4  -  2021-02-24 00:30:49.635328  average elapsed: 512198.4 microseconds
dradius: 1.5 k: 61  -> point count: 735  -  2021-02-24 00:30:50.184610  elapsed: 549230 microseconds
dradius: 1.5 k: 61  -> point count: 746  -  2021-02-24 00:30:50.701255  elapsed: 516577 microseconds
dradius: 1.5 k: 61  -> point count: 732  -  2021-02-24 00:30:51.244490  elapsed: 543167 microseconds
dradius: 1.5 k: 61  -> point count: 737  -  2021-02-24 00:30:51.759390  elapsed: 514830 microseconds
dradius: 1.5 k: 61  -> point count: 740  -  2021-02-24 00:30:52.306289  elapsed: 546833 microseconds
====== dradius: 1.5 k: 61  -> point count average: 738.0  -  2021-02-24 00:30:52.306348  average elapsed: 534127.4 microseconds
dradius: 1.5 k: 62  -> point count: 733  -  2021-02-24 00:30:52.859184  elapsed: 552782 microseconds
dradius: 1.5 k: 62  -> point count: 748  -  2021-02-24 00:30:53.376723  elapsed: 517465 microseconds
dradius: 1.5 k: 62  -> point count: 740  -  2021-02-24 00:30:53.924048  elapsed: 547252 microseconds
dradius: 1.5 k: 62  -> point count: 751  -  2021-02-24 00:30:54.437162  elapsed: 513016 microseconds
dradius: 1.5 k: 62  -> point count: 737  -  2021-02-24 00:30:54.966311  elapsed: 529080 microseconds
====== dradius: 1.5 k: 62  -> point count average: 741.8  -  2021-02-24 00:30:54.966363  average elapsed: 531919.0 microseconds
dradius: 1.5 k: 63  -> point count: 739  -  2021-02-24 00:30:55.514920  elapsed: 548506 microseconds
dradius: 1.5 k: 63  -> point count: 740  -  2021-02-24 00:30:56.019558  elapsed: 504572 microseconds
dradius: 1.5 k: 63  -> point count: 741  -  2021-02-24 00:30:56.568458  elapsed: 548833 microseconds
dradius: 1.5 k: 63  -> point count: 741  -  2021-02-24 00:30:57.071252  elapsed: 502729 microseconds
dradius: 1.5 k: 63  -> point count: 741  -  2021-02-24 00:30:57.599029  elapsed: 527698 microseconds
====== dradius: 1.5 k: 63  -> point count average: 740.4  -  2021-02-24 00:30:57.599088  average elapsed: 526467.6 microseconds
dradius: 1.5 k: 64  -> point count: 740  -  2021-02-24 00:30:58.139770  elapsed: 540631 microseconds
dradius: 1.5 k: 64  -> point count: 727  -  2021-02-24 00:30:58.634387  elapsed: 494547 microseconds
dradius: 1.5 k: 64  -> point count: 742  -  2021-02-24 00:30:59.166792  elapsed: 532336 microseconds
dradius: 1.5 k: 64  -> point count: 746  -  2021-02-24 00:30:59.671746  elapsed: 504848 microseconds
dradius: 1.5 k: 64  -> point count: 746  -  2021-02-24 00:31:00.211009  elapsed: 539198 microseconds
====== dradius: 1.5 k: 64  -> point count average: 740.2  -  2021-02-24 00:31:00.211059  average elapsed: 522312.0 microseconds
dradius: 1.5 k: 65  -> point count: 746  -  2021-02-24 00:31:00.758098  elapsed: 546992 microseconds
dradius: 1.5 k: 65  -> point count: 746  -  2021-02-24 00:31:01.273795  elapsed: 515635 microseconds
dradius: 1.5 k: 65  -> point count: 751  -  2021-02-24 00:31:01.805499  elapsed: 531638 microseconds
dradius: 1.5 k: 65  -> point count: 732  -  2021-02-24 00:31:02.291533  elapsed: 485963 microseconds
dradius: 1.5 k: 65  -> point count: 728  -  2021-02-24 00:31:02.803163  elapsed: 511567 microseconds
====== dradius: 1.5 k: 65  -> point count average: 740.6  -  2021-02-24 00:31:02.803214  average elapsed: 518359.0 microseconds
dradius: 1.5 k: 66  -> point count: 745  -  2021-02-24 00:31:03.336313  elapsed: 533052 microseconds
dradius: 1.5 k: 66  -> point count: 748  -  2021-02-24 00:31:03.844396  elapsed: 508021 microseconds
dradius: 1.5 k: 66  -> point count: 742  -  2021-02-24 00:31:04.382164  elapsed: 537696 microseconds
dradius: 1.5 k: 66  -> point count: 745  -  2021-02-24 00:31:04.887738  elapsed: 505512 microseconds
dradius: 1.5 k: 66  -> point count: 740  -  2021-02-24 00:31:05.415950  elapsed: 528136 microseconds
====== dradius: 1.5 k: 66  -> point count average: 744.0  -  2021-02-24 00:31:05.416018  average elapsed: 522483.4 microseconds
dradius: 1.5 k: 67  -> point count: 729  -  2021-02-24 00:31:05.941769  elapsed: 525692 microseconds
dradius: 1.5 k: 67  -> point count: 746  -  2021-02-24 00:31:06.448895  elapsed: 507058 microseconds
dradius: 1.5 k: 67  -> point count: 735  -  2021-02-24 00:31:06.978348  elapsed: 529380 microseconds
dradius: 1.5 k: 67  -> point count: 731  -  2021-02-24 00:31:07.482300  elapsed: 503887 microseconds
dradius: 1.5 k: 67  -> point count: 739  -  2021-02-24 00:31:08.026882  elapsed: 544517 microseconds
====== dradius: 1.5 k: 67  -> point count average: 736.0  -  2021-02-24 00:31:08.026932  average elapsed: 522106.8 microseconds
dradius: 1.5 k: 68  -> point count: 744  -  2021-02-24 00:31:08.549721  elapsed: 522742 microseconds
dradius: 1.5 k: 68  -> point count: 744  -  2021-02-24 00:31:09.099545  elapsed: 549761 microseconds
dradius: 1.5 k: 68  -> point count: 737  -  2021-02-24 00:31:09.642995  elapsed: 543386 microseconds
dradius: 1.5 k: 68  -> point count: 727  -  2021-02-24 00:31:10.170697  elapsed: 527639 microseconds
dradius: 1.5 k: 68  -> point count: 743  -  2021-02-24 00:31:10.713710  elapsed: 542949 microseconds
====== dradius: 1.5 k: 68  -> point count average: 739.0  -  2021-02-24 00:31:10.713760  average elapsed: 537295.4 microseconds
dradius: 1.5 k: 69  -> point count: 744  -  2021-02-24 00:31:11.247948  elapsed: 534140 microseconds
dradius: 1.5 k: 69  -> point count: 737  -  2021-02-24 00:31:11.791133  elapsed: 543117 microseconds
dradius: 1.5 k: 69  -> point count: 739  -  2021-02-24 00:31:12.348674  elapsed: 557473 microseconds
dradius: 1.5 k: 69  -> point count: 750  -  2021-02-24 00:31:12.875012  elapsed: 526274 microseconds
dradius: 1.5 k: 69  -> point count: 738  -  2021-02-24 00:31:13.427928  elapsed: 552820 microseconds
====== dradius: 1.5 k: 69  -> point count average: 741.6  -  2021-02-24 00:31:13.427984  average elapsed: 542764.8 microseconds
dradius: 1.5 k: 70  -> point count: 741  -  2021-02-24 00:31:13.950891  elapsed: 522855 microseconds
dradius: 1.5 k: 70  -> point count: 739  -  2021-02-24 00:31:14.503632  elapsed: 552678 microseconds
dradius: 1.5 k: 70  -> point count: 745  -  2021-02-24 00:31:15.056797  elapsed: 553103 microseconds
dradius: 1.5 k: 70  -> point count: 733  -  2021-02-24 00:31:15.571245  elapsed: 514385 microseconds
dradius: 1.5 k: 70  -> point count: 748  -  2021-02-24 00:31:16.133406  elapsed: 562097 microseconds
====== dradius: 1.5 k: 70  -> point count average: 741.2  -  2021-02-24 00:31:16.133512  average elapsed: 541023.6 microseconds
dradius: 1.5 k: 71  -> point count: 739  -  2021-02-24 00:31:16.672137  elapsed: 538568 microseconds
dradius: 1.5 k: 71  -> point count: 738  -  2021-02-24 00:31:17.237890  elapsed: 565686 microseconds
dradius: 1.5 k: 71  -> point count: 740  -  2021-02-24 00:31:17.812295  elapsed: 574333 microseconds
dradius: 1.5 k: 71  -> point count: 737  -  2021-02-24 00:31:18.361870  elapsed: 549505 microseconds
dradius: 1.5 k: 71  -> point count: 754  -  2021-02-24 00:31:18.950567  elapsed: 588633 microseconds
====== dradius: 1.5 k: 71  -> point count average: 741.6  -  2021-02-24 00:31:18.950619  average elapsed: 563345.0 microseconds
dradius: 1.5 k: 72  -> point count: 731  -  2021-02-24 00:31:19.499630  elapsed: 548962 microseconds
dradius: 1.5 k: 72  -> point count: 747  -  2021-02-24 00:31:20.097175  elapsed: 597395 microseconds
dradius: 1.5 k: 72  -> point count: 758  -  2021-02-24 00:31:20.690357  elapsed: 592980 microseconds
dradius: 1.5 k: 72  -> point count: 748  -  2021-02-24 00:31:21.246388  elapsed: 555963 microseconds
dradius: 1.5 k: 72  -> point count: 750  -  2021-02-24 00:31:21.839009  elapsed: 592553 microseconds
====== dradius: 1.5 k: 72  -> point count average: 746.8  -  2021-02-24 00:31:21.839108  average elapsed: 577570.6 microseconds
dradius: 1.5 k: 73  -> point count: 743  -  2021-02-24 00:31:22.396991  elapsed: 557820 microseconds
dradius: 1.5 k: 73  -> point count: 739  -  2021-02-24 00:31:22.972669  elapsed: 575613 microseconds
dradius: 1.5 k: 73  -> point count: 747  -  2021-02-24 00:31:23.519302  elapsed: 546570 microseconds
dradius: 1.5 k: 73  -> point count: 736  -  2021-02-24 00:31:24.088077  elapsed: 568709 microseconds
dradius: 1.5 k: 73  -> point count: 758  -  2021-02-24 00:31:24.671585  elapsed: 583435 microseconds
====== dradius: 1.5 k: 73  -> point count average: 744.6  -  2021-02-24 00:31:24.671636  average elapsed: 566429.4 microseconds
dradius: 1.5 k: 74  -> point count: 758  -  2021-02-24 00:31:25.253181  elapsed: 581495 microseconds
dradius: 1.5 k: 74  -> point count: 740  -  2021-02-24 00:31:25.850492  elapsed: 597245 microseconds
dradius: 1.5 k: 74  -> point count: 742  -  2021-02-24 00:31:26.401159  elapsed: 550605 microseconds
dradius: 1.5 k: 74  -> point count: 742  -  2021-02-24 00:31:26.984244  elapsed: 583015 microseconds
dradius: 1.5 k: 74  -> point count: 737  -  2021-02-24 00:31:27.564348  elapsed: 580042 microseconds
====== dradius: 1.5 k: 74  -> point count average: 743.8  -  2021-02-24 00:31:27.564401  average elapsed: 578480.4 microseconds
dradius: 1.5 k: 75  -> point count: 756  -  2021-02-24 00:31:28.134045  elapsed: 569587 microseconds
dradius: 1.5 k: 75  -> point count: 738  -  2021-02-24 00:31:28.713378  elapsed: 579271 microseconds
dradius: 1.5 k: 75  -> point count: 738  -  2021-02-24 00:31:29.273916  elapsed: 560469 microseconds
dradius: 1.5 k: 75  -> point count: 749  -  2021-02-24 00:31:29.878543  elapsed: 604554 microseconds
dradius: 1.5 k: 75  -> point count: 748  -  2021-02-24 00:31:30.465928  elapsed: 587305 microseconds
====== dradius: 1.5 k: 75  -> point count average: 745.8  -  2021-02-24 00:31:30.465977  average elapsed: 580237.2 microseconds
dradius: 1.5 k: 76  -> point count: 740  -  2021-02-24 00:31:31.028404  elapsed: 562380 microseconds
dradius: 1.5 k: 76  -> point count: 746  -  2021-02-24 00:31:31.625494  elapsed: 597023 microseconds
dradius: 1.5 k: 76  -> point count: 742  -  2021-02-24 00:31:32.181586  elapsed: 556030 microseconds
dradius: 1.5 k: 76  -> point count: 736  -  2021-02-24 00:31:32.759606  elapsed: 577953 microseconds
dradius: 1.5 k: 76  -> point count: 728  -  2021-02-24 00:31:33.343738  elapsed: 584063 microseconds
====== dradius: 1.5 k: 76  -> point count average: 738.4  -  2021-02-24 00:31:33.343794  average elapsed: 575489.8 microseconds
dradius: 1.5 k: 77  -> point count: 739  -  2021-02-24 00:31:33.906415  elapsed: 562570 microseconds
dradius: 1.5 k: 77  -> point count: 735  -  2021-02-24 00:31:34.492745  elapsed: 586265 microseconds
dradius: 1.5 k: 77  -> point count: 749  -  2021-02-24 00:31:35.050493  elapsed: 557673 microseconds
dradius: 1.5 k: 77  -> point count: 736  -  2021-02-24 00:31:35.636666  elapsed: 586098 microseconds
dradius: 1.5 k: 77  -> point count: 750  -  2021-02-24 00:31:36.246873  elapsed: 610145 microseconds
====== dradius: 1.5 k: 77  -> point count average: 741.8  -  2021-02-24 00:31:36.246926  average elapsed: 580550.2 microseconds
dradius: 1.5 k: 78  -> point count: 743  -  2021-02-24 00:31:36.830400  elapsed: 583427 microseconds
dradius: 1.5 k: 78  -> point count: 744  -  2021-02-24 00:31:37.442246  elapsed: 611775 microseconds
dradius: 1.5 k: 78  -> point count: 759  -  2021-02-24 00:31:38.033213  elapsed: 590902 microseconds
dradius: 1.5 k: 78  -> point count: 740  -  2021-02-24 00:31:38.630323  elapsed: 597047 microseconds
dradius: 1.5 k: 78  -> point count: 745  -  2021-02-24 00:31:39.235526  elapsed: 605140 microseconds
====== dradius: 1.5 k: 78  -> point count average: 746.2  -  2021-02-24 00:31:39.235583  average elapsed: 597658.2 microseconds
dradius: 1.5 k: 79  -> point count: 737  -  2021-02-24 00:31:39.819995  elapsed: 584358 microseconds
dradius: 1.5 k: 79  -> point count: 739  -  2021-02-24 00:31:40.427153  elapsed: 607094 microseconds
dradius: 1.5 k: 79  -> point count: 739  -  2021-02-24 00:31:41.013231  elapsed: 586016 microseconds
dradius: 1.5 k: 79  -> point count: 744  -  2021-02-24 00:31:41.659816  elapsed: 646520 microseconds
dradius: 1.5 k: 79  -> point count: 756  -  2021-02-24 00:31:42.312431  elapsed: 652551 microseconds
====== dradius: 1.5 k: 79  -> point count average: 743.0  -  2021-02-24 00:31:42.312481  average elapsed: 615307.8 microseconds
dradius: 1.5 k: 80  -> point count: 742  -  2021-02-24 00:31:42.912802  elapsed: 600273 microseconds
dradius: 1.5 k: 80  -> point count: 748  -  2021-02-24 00:31:43.547764  elapsed: 634897 microseconds
dradius: 1.5 k: 80  -> point count: 753  -  2021-02-24 00:31:44.168305  elapsed: 620477 microseconds
dradius: 1.5 k: 80  -> point count: 757  -  2021-02-24 00:31:44.810996  elapsed: 642599 microseconds
dradius: 1.5 k: 80  -> point count: 754  -  2021-02-24 00:31:45.424852  elapsed: 613792 microseconds
====== dradius: 1.5 k: 80  -> point count average: 750.8  -  2021-02-24 00:31:45.424905  average elapsed: 622407.6 microseconds
dradius: 1.5 k: 81  -> point count: 747  -  2021-02-24 00:31:46.072485  elapsed: 647530 microseconds
dradius: 1.5 k: 81  -> point count: 747  -  2021-02-24 00:31:46.713500  elapsed: 640952 microseconds
dradius: 1.5 k: 81  -> point count: 749  -  2021-02-24 00:31:47.329958  elapsed: 616395 microseconds
dradius: 1.5 k: 81  -> point count: 739  -  2021-02-24 00:31:47.980609  elapsed: 650585 microseconds
dradius: 1.5 k: 81  -> point count: 738  -  2021-02-24 00:31:48.593480  elapsed: 612803 microseconds
====== dradius: 1.5 k: 81  -> point count average: 744.0  -  2021-02-24 00:31:48.593531  average elapsed: 633653.0 microseconds
dradius: 1.5 k: 82  -> point count: 742  -  2021-02-24 00:31:49.251267  elapsed: 657690 microseconds
dradius: 1.5 k: 82  -> point count: 749  -  2021-02-24 00:31:49.933571  elapsed: 682239 microseconds
dradius: 1.5 k: 82  -> point count: 745  -  2021-02-24 00:31:50.581007  elapsed: 647369 microseconds
dradius: 1.5 k: 82  -> point count: 741  -  2021-02-24 00:31:51.257458  elapsed: 676383 microseconds
dradius: 1.5 k: 82  -> point count: 728  -  2021-02-24 00:31:51.911982  elapsed: 654446 microseconds
====== dradius: 1.5 k: 82  -> point count average: 741.0  -  2021-02-24 00:31:51.912036  average elapsed: 663625.4 microseconds
dradius: 1.5 k: 83  -> point count: 742  -  2021-02-24 00:31:52.600976  elapsed: 688891 microseconds
dradius: 1.5 k: 83  -> point count: 754  -  2021-02-24 00:31:53.311684  elapsed: 710644 microseconds
dradius: 1.5 k: 83  -> point count: 744  -  2021-02-24 00:31:53.978321  elapsed: 666565 microseconds
dradius: 1.5 k: 83  -> point count: 740  -  2021-02-24 00:31:54.668415  elapsed: 690027 microseconds
dradius: 1.5 k: 83  -> point count: 750  -  2021-02-24 00:31:55.331488  elapsed: 663005 microseconds
====== dradius: 1.5 k: 83  -> point count average: 746.0  -  2021-02-24 00:31:55.331544  average elapsed: 683826.4 microseconds
dradius: 1.5 k: 84  -> point count: 754  -  2021-02-24 00:31:56.018985  elapsed: 687391 microseconds
dradius: 1.5 k: 84  -> point count: 756  -  2021-02-24 00:31:56.711384  elapsed: 692310 microseconds
dradius: 1.5 k: 84  -> point count: 745  -  2021-02-24 00:31:57.378627  elapsed: 667156 microseconds
dradius: 1.5 k: 84  -> point count: 762  -  2021-02-24 00:31:58.083239  elapsed: 704447 microseconds
dradius: 1.5 k: 84  -> point count: 741  -  2021-02-24 00:31:58.721494  elapsed: 638181 microseconds
====== dradius: 1.5 k: 84  -> point count average: 751.6  -  2021-02-24 00:31:58.721546  average elapsed: 677897.0 microseconds
dradius: 1.5 k: 85  -> point count: 752  -  2021-02-24 00:31:59.422351  elapsed: 700758 microseconds
dradius: 1.5 k: 85  -> point count: 750  -  2021-02-24 00:32:00.105042  elapsed: 682621 microseconds
dradius: 1.5 k: 85  -> point count: 743  -  2021-02-24 00:32:00.748561  elapsed: 643435 microseconds
dradius: 1.5 k: 85  -> point count: 755  -  2021-02-24 00:32:01.427940  elapsed: 679309 microseconds
dradius: 1.5 k: 85  -> point count: 755  -  2021-02-24 00:32:02.071628  elapsed: 643619 microseconds
====== dradius: 1.5 k: 85  -> point count average: 751.0  -  2021-02-24 00:32:02.071679  average elapsed: 669948.4 microseconds
dradius: 1.5 k: 86  -> point count: 743  -  2021-02-24 00:32:02.743061  elapsed: 671334 microseconds
dradius: 1.5 k: 86  -> point count: 745  -  2021-02-24 00:32:03.392259  elapsed: 649129 microseconds
dradius: 1.5 k: 86  -> point count: 754  -  2021-02-24 00:32:04.076664  elapsed: 684342 microseconds
dradius: 1.5 k: 86  -> point count: 746  -  2021-02-24 00:32:04.747046  elapsed: 670317 microseconds
dradius: 1.5 k: 86  -> point count: 749  -  2021-02-24 00:32:05.392081  elapsed: 644964 microseconds
====== dradius: 1.5 k: 86  -> point count average: 747.4  -  2021-02-24 00:32:05.392131  average elapsed: 664017.2 microseconds
dradius: 1.5 k: 87  -> point count: 742  -  2021-02-24 00:32:06.073259  elapsed: 681082 microseconds
dradius: 1.5 k: 87  -> point count: 746  -  2021-02-24 00:32:06.726129  elapsed: 652805 microseconds
dradius: 1.5 k: 87  -> point count: 751  -  2021-02-24 00:32:07.410238  elapsed: 684044 microseconds
dradius: 1.5 k: 87  -> point count: 740  -  2021-02-24 00:32:08.082773  elapsed: 672465 microseconds
dradius: 1.5 k: 87  -> point count: 753  -  2021-02-24 00:32:08.731533  elapsed: 648698 microseconds
====== dradius: 1.5 k: 87  -> point count average: 746.4  -  2021-02-24 00:32:08.731582  average elapsed: 667818.8 microseconds
dradius: 1.5 k: 88  -> point count: 748  -  2021-02-24 00:32:09.422188  elapsed: 690559 microseconds
dradius: 1.5 k: 88  -> point count: 743  -  2021-02-24 00:32:10.104028  elapsed: 681769 microseconds
dradius: 1.5 k: 88  -> point count: 753  -  2021-02-24 00:32:10.802321  elapsed: 698205 microseconds
dradius: 1.5 k: 88  -> point count: 759  -  2021-02-24 00:32:11.501727  elapsed: 699329 microseconds
dradius: 1.5 k: 88  -> point count: 749  -  2021-02-24 00:32:12.160320  elapsed: 658527 microseconds
====== dradius: 1.5 k: 88  -> point count average: 750.4  -  2021-02-24 00:32:12.160371  average elapsed: 685677.8 microseconds
dradius: 1.5 k: 89  -> point count: 760  -  2021-02-24 00:32:12.849883  elapsed: 689466 microseconds
dradius: 1.5 k: 89  -> point count: 751  -  2021-02-24 00:32:13.525580  elapsed: 675631 microseconds
dradius: 1.5 k: 89  -> point count: 755  -  2021-02-24 00:32:14.215606  elapsed: 689946 microseconds
dradius: 1.5 k: 89  -> point count: 758  -  2021-02-24 00:32:14.914541  elapsed: 698866 microseconds
dradius: 1.5 k: 89  -> point count: 739  -  2021-02-24 00:32:15.575665  elapsed: 661059 microseconds
====== dradius: 1.5 k: 89  -> point count average: 752.6  -  2021-02-24 00:32:15.575717  average elapsed: 682993.6 microseconds
dradius: 1.5 k: 90  -> point count: 747  -  2021-02-24 00:32:16.267088  elapsed: 691321 microseconds
dradius: 1.5 k: 90  -> point count: 749  -  2021-02-24 00:32:16.950323  elapsed: 683171 microseconds
dradius: 1.5 k: 90  -> point count: 756  -  2021-02-24 00:32:17.653115  elapsed: 702718 microseconds
dradius: 1.5 k: 90  -> point count: 747  -  2021-02-24 00:32:18.363895  elapsed: 710717 microseconds
dradius: 1.5 k: 90  -> point count: 740  -  2021-02-24 00:32:19.022388  elapsed: 658427 microseconds
====== dradius: 1.5 k: 90  -> point count average: 747.8  -  2021-02-24 00:32:19.022440  average elapsed: 689270.8 microseconds
dradius: 1.5 k: 91  -> point count: 749  -  2021-02-24 00:32:19.727155  elapsed: 704667 microseconds
dradius: 1.5 k: 91  -> point count: 746  -  2021-02-24 00:32:20.412303  elapsed: 685082 microseconds
dradius: 1.5 k: 91  -> point count: 743  -  2021-02-24 00:32:21.119359  elapsed: 706992 microseconds
dradius: 1.5 k: 91  -> point count: 750  -  2021-02-24 00:32:21.825094  elapsed: 705669 microseconds
dradius: 1.5 k: 91  -> point count: 752  -  2021-02-24 00:32:22.498858  elapsed: 673697 microseconds
====== dradius: 1.5 k: 91  -> point count average: 748.0  -  2021-02-24 00:32:22.498909  average elapsed: 695221.4 microseconds
dradius: 1.5 k: 92  -> point count: 753  -  2021-02-24 00:32:23.222551  elapsed: 723595 microseconds
dradius: 1.5 k: 92  -> point count: 747  -  2021-02-24 00:32:23.885576  elapsed: 662948 microseconds
dradius: 1.5 k: 92  -> point count: 763  -  2021-02-24 00:32:24.595130  elapsed: 709486 microseconds
dradius: 1.5 k: 92  -> point count: 750  -  2021-02-24 00:32:25.265483  elapsed: 670284 microseconds
dradius: 1.5 k: 92  -> point count: 745  -  2021-02-24 00:32:25.949215  elapsed: 683669 microseconds
====== dradius: 1.5 k: 92  -> point count average: 751.6  -  2021-02-24 00:32:25.949266  average elapsed: 689996.4 microseconds
dradius: 1.5 k: 93  -> point count: 768  -  2021-02-24 00:32:26.661491  elapsed: 712179 microseconds
dradius: 1.5 k: 93  -> point count: 742  -  2021-02-24 00:32:27.329459  elapsed: 667907 microseconds
dradius: 1.5 k: 93  -> point count: 759  -  2021-02-24 00:32:28.041558  elapsed: 712036 microseconds
dradius: 1.5 k: 93  -> point count: 758  -  2021-02-24 00:32:28.714625  elapsed: 673003 microseconds
dradius: 1.5 k: 93  -> point count: 754  -  2021-02-24 00:32:29.416309  elapsed: 701621 microseconds
====== dradius: 1.5 k: 93  -> point count average: 756.2  -  2021-02-24 00:32:29.416357  average elapsed: 693349.2 microseconds
dradius: 1.5 k: 94  -> point count: 741  -  2021-02-24 00:32:30.128964  elapsed: 712563 microseconds
dradius: 1.5 k: 94  -> point count: 758  -  2021-02-24 00:32:30.809324  elapsed: 680292 microseconds
dradius: 1.5 k: 94  -> point count: 754  -  2021-02-24 00:32:31.510553  elapsed: 701166 microseconds
dradius: 1.5 k: 94  -> point count: 767  -  2021-02-24 00:32:32.189365  elapsed: 678750 microseconds
dradius: 1.5 k: 94  -> point count: 757  -  2021-02-24 00:32:32.877685  elapsed: 688254 microseconds
====== dradius: 1.5 k: 94  -> point count average: 755.4  -  2021-02-24 00:32:32.877737  average elapsed: 692205.0 microseconds
dradius: 1.5 k: 95  -> point count: 742  -  2021-02-24 00:32:33.577548  elapsed: 699762 microseconds
dradius: 1.5 k: 95  -> point count: 752  -  2021-02-24 00:32:34.272169  elapsed: 694549 microseconds
dradius: 1.5 k: 95  -> point count: 753  -  2021-02-24 00:32:34.972829  elapsed: 700593 microseconds
dradius: 1.5 k: 95  -> point count: 755  -  2021-02-24 00:32:35.660532  elapsed: 687631 microseconds
dradius: 1.5 k: 95  -> point count: 751  -  2021-02-24 00:32:36.390079  elapsed: 729448 microseconds
====== dradius: 1.5 k: 95  -> point count average: 750.6  -  2021-02-24 00:32:36.390246  average elapsed: 702396.6 microseconds
dradius: 1.5 k: 96  -> point count: 751  -  2021-02-24 00:32:37.149907  elapsed: 759600 microseconds
dradius: 1.5 k: 96  -> point count: 749  -  2021-02-24 00:32:37.877122  elapsed: 727143 microseconds
dradius: 1.5 k: 96  -> point count: 757  -  2021-02-24 00:32:38.642327  elapsed: 765136 microseconds
dradius: 1.5 k: 96  -> point count: 753  -  2021-02-24 00:32:39.366524  elapsed: 724128 microseconds
dradius: 1.5 k: 96  -> point count: 754  -  2021-02-24 00:32:40.138711  elapsed: 772118 microseconds
====== dradius: 1.5 k: 96  -> point count average: 752.8  -  2021-02-24 00:32:40.138761  average elapsed: 749625.0 microseconds
dradius: 1.5 k: 97  -> point count: 755  -  2021-02-24 00:32:40.906445  elapsed: 767635 microseconds
dradius: 1.5 k: 97  -> point count: 751  -  2021-02-24 00:32:41.646468  elapsed: 739957 microseconds
dradius: 1.5 k: 97  -> point count: 742  -  2021-02-24 00:32:42.410158  elapsed: 763627 microseconds
dradius: 1.5 k: 97  -> point count: 752  -  2021-02-24 00:32:43.126427  elapsed: 716194 microseconds
dradius: 1.5 k: 97  -> point count: 751  -  2021-02-24 00:32:43.897829  elapsed: 771337 microseconds
====== dradius: 1.5 k: 97  -> point count average: 750.2  -  2021-02-24 00:32:43.897878  average elapsed: 751750.0 microseconds
dradius: 1.5 k: 98  -> point count: 752  -  2021-02-24 00:32:44.658979  elapsed: 761054 microseconds
dradius: 1.5 k: 98  -> point count: 759  -  2021-02-24 00:32:45.405416  elapsed: 746373 microseconds
dradius: 1.5 k: 98  -> point count: 753  -  2021-02-24 00:32:46.167301  elapsed: 761810 microseconds
dradius: 1.5 k: 98  -> point count: 757  -  2021-02-24 00:32:46.885854  elapsed: 718487 microseconds
dradius: 1.5 k: 98  -> point count: 753  -  2021-02-24 00:32:47.642724  elapsed: 756793 microseconds
====== dradius: 1.5 k: 98  -> point count average: 754.8  -  2021-02-24 00:32:47.642777  average elapsed: 748903.4 microseconds
dradius: 1.5 k: 99  -> point count: 739  -  2021-02-24 00:32:48.362526  elapsed: 719701 microseconds
dradius: 1.5 k: 99  -> point count: 759  -  2021-02-24 00:32:49.100981  elapsed: 738390 microseconds
dradius: 1.5 k: 99  -> point count: 759  -  2021-02-24 00:32:49.853315  elapsed: 752273 microseconds
dradius: 1.5 k: 99  -> point count: 760  -  2021-02-24 00:32:50.565909  elapsed: 712533 microseconds
dradius: 1.5 k: 99  -> point count: 750  -  2021-02-24 00:32:51.298254  elapsed: 732282 microseconds
====== dradius: 1.5 k: 99  -> point count average: 753.4  -  2021-02-24 00:32:51.298302  average elapsed: 731035.8 microseconds





x, y: 200 , 200
dradius: 1.5 k: 1  -> point count: 15  -  2021-02-24 00:32:53.044358  elapsed: 745993 microseconds
dradius: 1.5 k: 1  -> point count: 9  -  2021-02-24 00:32:54.763765  elapsed: 719336 microseconds
dradius: 1.5 k: 1  -> point count: 7  -  2021-02-24 00:32:56.443569  elapsed: 679730 microseconds
dradius: 1.5 k: 1  -> point count: 9  -  2021-02-24 00:32:58.074342  elapsed: 630697 microseconds
dradius: 1.5 k: 1  -> point count: 18  -  2021-02-24 00:32:59.738101  elapsed: 663687 microseconds
====== dradius: 1.5 k: 1  -> point count average: 11.6  -  2021-02-24 00:32:59.738159  average elapsed: 687888.6 microseconds
dradius: 1.5 k: 2  -> point count: 358  -  2021-02-24 00:33:01.435456  elapsed: 697247 microseconds
dradius: 1.5 k: 2  -> point count: 311  -  2021-02-24 00:33:03.193251  elapsed: 757723 microseconds
dradius: 1.5 k: 2  -> point count: 196  -  2021-02-24 00:33:04.863288  elapsed: 669954 microseconds
dradius: 1.5 k: 2  -> point count: 2247  -  2021-02-24 00:33:06.681843  elapsed: 818482 microseconds
dradius: 1.5 k: 2  -> point count: 2806  -  2021-02-24 00:33:08.467278  elapsed: 785359 microseconds
====== dradius: 1.5 k: 2  -> point count average: 1183.6  -  2021-02-24 00:33:08.467333  average elapsed: 745753.0 microseconds
dradius: 1.5 k: 3  -> point count: 7813  -  2021-02-24 00:33:10.622162  elapsed: 154776 microseconds
dradius: 1.5 k: 3  -> point count: 7809  -  2021-02-24 00:33:12.782287  elapsed: 160008 microseconds
dradius: 1.5 k: 3  -> point count: 7823  -  2021-02-24 00:33:14.878564  elapsed: 96177 microseconds
dradius: 1.5 k: 3  -> point count: 7830  -  2021-02-24 00:33:17.050724  elapsed: 172049 microseconds
dradius: 1.5 k: 3  -> point count: 7932  -  2021-02-24 00:33:19.215296  elapsed: 164465 microseconds
====== dradius: 1.5 k: 3  -> point count average: 7841.4  -  2021-02-24 00:33:19.215354  average elapsed: 149495.0 microseconds
dradius: 1.5 k: 4  -> point count: 8830  -  2021-02-24 00:33:21.457161  elapsed: 241719 microseconds
dradius: 1.5 k: 4  -> point count: 8771  -  2021-02-24 00:33:23.755629  elapsed: 298335 microseconds
dradius: 1.5 k: 4  -> point count: 8799  -  2021-02-24 00:33:26.057067  elapsed: 301332 microseconds
dradius: 1.5 k: 4  -> point count: 8748  -  2021-02-24 00:33:28.294928  elapsed: 237763 microseconds
dradius: 1.5 k: 4  -> point count: 8746  -  2021-02-24 00:33:30.686364  elapsed: 391329 microseconds
====== dradius: 1.5 k: 4  -> point count average: 8778.8  -  2021-02-24 00:33:30.686419  average elapsed: 294095.6 microseconds
dradius: 1.5 k: 5  -> point count: 9300  -  2021-02-24 00:33:33.268242  elapsed: 581736 microseconds
dradius: 1.5 k: 5  -> point count: 9311  -  2021-02-24 00:33:35.811698  elapsed: 543347 microseconds
dradius: 1.5 k: 5  -> point count: 9388  -  2021-02-24 00:33:38.414112  elapsed: 602308 microseconds
dradius: 1.5 k: 5  -> point count: 9286  -  2021-02-24 00:33:40.895762  elapsed: 481539 microseconds
dradius: 1.5 k: 5  -> point count: 9282  -  2021-02-24 00:33:43.432693  elapsed: 536819 microseconds
====== dradius: 1.5 k: 5  -> point count average: 9313.4  -  2021-02-24 00:33:43.432757  average elapsed: 549149.8 microseconds
dradius: 1.5 k: 6  -> point count: 9604  -  2021-02-24 00:33:45.996953  elapsed: 564093 microseconds
dradius: 1.5 k: 6  -> point count: 9595  -  2021-02-24 00:33:48.618734  elapsed: 621676 microseconds
dradius: 1.5 k: 6  -> point count: 9632  -  2021-02-24 00:33:51.214225  elapsed: 595379 microseconds
dradius: 1.5 k: 6  -> point count: 9639  -  2021-02-24 00:33:53.850586  elapsed: 636253 microseconds
dradius: 1.5 k: 6  -> point count: 9570  -  2021-02-24 00:33:56.382717  elapsed: 532030 microseconds
====== dradius: 1.5 k: 6  -> point count average: 9608.0  -  2021-02-24 00:33:56.382772  average elapsed: 589886.2 microseconds
dradius: 1.5 k: 7  -> point count: 9787  -  2021-02-24 00:33:59.044643  elapsed: 661786 microseconds
dradius: 1.5 k: 7  -> point count: 9830  -  2021-02-24 00:34:01.660858  elapsed: 616040 microseconds
dradius: 1.5 k: 7  -> point count: 9834  -  2021-02-24 00:34:04.439364  elapsed: 778400 microseconds
dradius: 1.5 k: 7  -> point count: 9885  -  2021-02-24 00:34:07.380987  elapsed: 941490 microseconds
dradius: 1.5 k: 7  -> point count: 9815  -  2021-02-24 00:34:10.168121  elapsed: 787014 microseconds
====== dradius: 1.5 k: 7  -> point count average: 9830.2  -  2021-02-24 00:34:10.168182  average elapsed: 756946.0 microseconds
dradius: 1.5 k: 8  -> point count: 9970  -  2021-02-24 00:34:12.937105  elapsed: 768835 microseconds
dradius: 1.5 k: 8  -> point count: 10019  -  2021-02-24 00:34:15.699131  elapsed: 761916 microseconds
dradius: 1.5 k: 8  -> point count: 10022  -  2021-02-24 00:34:18.359064  elapsed: 659834 microseconds
dradius: 1.5 k: 8  -> point count: 9965  -  2021-02-24 00:34:21.071289  elapsed: 712113 microseconds
dradius: 1.5 k: 8  -> point count: 9977  -  2021-02-24 00:34:23.691737  elapsed: 620351 microseconds
====== dradius: 1.5 k: 8  -> point count average: 9990.6  -  2021-02-24 00:34:23.691793  average elapsed: 704609.8 microseconds
dradius: 1.5 k: 9  -> point count: 10183  -  2021-02-24 00:34:26.493975  elapsed: 802105 microseconds
dradius: 1.5 k: 9  -> point count: 10153  -  2021-02-24 00:34:29.225887  elapsed: 731816 microseconds
dradius: 1.5 k: 9  -> point count: 10161  -  2021-02-24 00:34:32.017724  elapsed: 791720 microseconds
dradius: 1.5 k: 9  -> point count: 10205  -  2021-02-24 00:34:34.762817  elapsed: 744994 microseconds
dradius: 1.5 k: 9  -> point count: 10117  -  2021-02-24 00:34:37.614489  elapsed: 851572 microseconds
====== dradius: 1.5 k: 9  -> point count average: 10163.8  -  2021-02-24 00:34:37.614548  average elapsed: 784441.4 microseconds
dradius: 1.5 k: 10  -> point count: 10250  -  2021-02-24 00:34:40.616194  elapsed: 1566 microseconds
dradius: 1.5 k: 10  -> point count: 10225  -  2021-02-24 00:34:43.656358  elapsed: 40055 microseconds
dradius: 1.5 k: 10  -> point count: 10271  -  2021-02-24 00:34:46.871544  elapsed: 215064 microseconds
dradius: 1.5 k: 10  -> point count: 10201  -  2021-02-24 00:34:49.978298  elapsed: 106644 microseconds
dradius: 1.5 k: 10  -> point count: 10212  -  2021-02-24 00:34:52.950973  elapsed: 972570 microseconds
====== dradius: 1.5 k: 10  -> point count average: 10231.8  -  2021-02-24 00:34:52.951029  average elapsed: 267179.8 microseconds
dradius: 1.5 k: 11  -> point count: 10331  -  2021-02-24 00:34:55.966109  elapsed: 15001 microseconds
dradius: 1.5 k: 11  -> point count: 10344  -  2021-02-24 00:34:59.400073  elapsed: 433861 microseconds
dradius: 1.5 k: 11  -> point count: 10312  -  2021-02-24 00:35:02.777854  elapsed: 377678 microseconds
dradius: 1.5 k: 11  -> point count: 10316  -  2021-02-24 00:35:05.897999  elapsed: 120025 microseconds
dradius: 1.5 k: 11  -> point count: 10354  -  2021-02-24 00:35:08.946664  elapsed: 48541 microseconds
====== dradius: 1.5 k: 11  -> point count average: 10331.4  -  2021-02-24 00:35:08.946722  average elapsed: 199021.2 microseconds
dradius: 1.5 k: 12  -> point count: 10370  -  2021-02-24 00:35:12.071312  elapsed: 124514 microseconds
dradius: 1.5 k: 12  -> point count: 10427  -  2021-02-24 00:35:15.231691  elapsed: 160281 microseconds
dradius: 1.5 k: 12  -> point count: 10438  -  2021-02-24 00:35:18.398741  elapsed: 166951 microseconds
dradius: 1.5 k: 12  -> point count: 10411  -  2021-02-24 00:35:21.597416  elapsed: 198559 microseconds
dradius: 1.5 k: 12  -> point count: 10461  -  2021-02-24 00:35:24.758656  elapsed: 161143 microseconds
====== dradius: 1.5 k: 12  -> point count average: 10421.4  -  2021-02-24 00:35:24.758712  average elapsed: 162289.6 microseconds
dradius: 1.5 k: 13  -> point count: 10487  -  2021-02-24 00:35:28.148852  elapsed: 390062 microseconds
dradius: 1.5 k: 13  -> point count: 10507  -  2021-02-24 00:35:31.521966  elapsed: 373017 microseconds
dradius: 1.5 k: 13  -> point count: 10494  -  2021-02-24 00:35:34.950324  elapsed: 428256 microseconds
dradius: 1.5 k: 13  -> point count: 10524  -  2021-02-24 00:35:38.303617  elapsed: 353187 microseconds
dradius: 1.5 k: 13  -> point count: 10476  -  2021-02-24 00:35:41.737771  elapsed: 434020 microseconds
====== dradius: 1.5 k: 13  -> point count average: 10497.6  -  2021-02-24 00:35:41.737835  average elapsed: 395708.4 microseconds
dradius: 1.5 k: 14  -> point count: 10524  -  2021-02-24 00:35:45.202152  elapsed: 464231 microseconds
dradius: 1.5 k: 14  -> point count: 10559  -  2021-02-24 00:35:48.717947  elapsed: 515695 microseconds
dradius: 1.5 k: 14  -> point count: 10579  -  2021-02-24 00:35:52.082482  elapsed: 364430 microseconds
dradius: 1.5 k: 14  -> point count: 10614  -  2021-02-24 00:35:55.500418  elapsed: 417828 microseconds
dradius: 1.5 k: 14  -> point count: 10512  -  2021-02-24 00:35:58.917907  elapsed: 417359 microseconds
====== dradius: 1.5 k: 14  -> point count average: 10557.6  -  2021-02-24 00:35:58.917966  average elapsed: 435908.6 microseconds
dradius: 1.5 k: 15  -> point count: 10660  -  2021-02-24 00:36:02.421541  elapsed: 503495 microseconds
dradius: 1.5 k: 15  -> point count: 10645  -  2021-02-24 00:36:05.954433  elapsed: 532712 microseconds
dradius: 1.5 k: 15  -> point count: 10601  -  2021-02-24 00:36:09.419047  elapsed: 464497 microseconds
dradius: 1.5 k: 15  -> point count: 10584  -  2021-02-24 00:36:12.961956  elapsed: 542803 microseconds
dradius: 1.5 k: 15  -> point count: 10621  -  2021-02-24 00:36:16.345175  elapsed: 383113 microseconds
====== dradius: 1.5 k: 15  -> point count average: 10622.2  -  2021-02-24 00:36:16.345230  average elapsed: 485324.0 microseconds
dradius: 1.5 k: 16  -> point count: 10667  -  2021-02-24 00:36:19.872170  elapsed: 526855 microseconds
dradius: 1.5 k: 16  -> point count: 10716  -  2021-02-24 00:36:23.367104  elapsed: 494837 microseconds
dradius: 1.5 k: 16  -> point count: 10650  -  2021-02-24 00:36:26.979070  elapsed: 611857 microseconds
dradius: 1.5 k: 16  -> point count: 10631  -  2021-02-24 00:36:30.558500  elapsed: 579312 microseconds
dradius: 1.5 k: 16  -> point count: 10676  -  2021-02-24 00:36:34.160272  elapsed: 601587 microseconds
====== dradius: 1.5 k: 16  -> point count average: 10668.0  -  2021-02-24 00:36:34.160336  average elapsed: 562889.6 microseconds
dradius: 1.5 k: 17  -> point count: 10747  -  2021-02-24 00:36:37.859670  elapsed: 699244 microseconds
dradius: 1.5 k: 17  -> point count: 10710  -  2021-02-24 00:36:41.444664  elapsed: 584851 microseconds
dradius: 1.5 k: 17  -> point count: 10711  -  2021-02-24 00:36:45.076746  elapsed: 631948 microseconds
dradius: 1.5 k: 17  -> point count: 10720  -  2021-02-24 00:36:48.943825  elapsed: 866970 microseconds
dradius: 1.5 k: 17  -> point count: 10685  -  2021-02-24 00:36:52.799256  elapsed: 855285 microseconds
====== dradius: 1.5 k: 17  -> point count average: 10714.6  -  2021-02-24 00:36:52.799313  average elapsed: 727659.6 microseconds
dradius: 1.5 k: 18  -> point count: 10831  -  2021-02-24 00:36:56.591112  elapsed: 791722 microseconds
dradius: 1.5 k: 18  -> point count: 10766  -  2021-02-24 00:37:00.346015  elapsed: 754798 microseconds
dradius: 1.5 k: 18  -> point count: 10786  -  2021-02-24 00:37:04.046604  elapsed: 700482 microseconds
dradius: 1.5 k: 18  -> point count: 10772  -  2021-02-24 00:37:07.744087  elapsed: 697347 microseconds
dradius: 1.5 k: 18  -> point count: 10776  -  2021-02-24 00:37:11.485726  elapsed: 741541 microseconds
====== dradius: 1.5 k: 18  -> point count average: 10786.2  -  2021-02-24 00:37:11.485786  average elapsed: 737178.0 microseconds
dradius: 1.5 k: 19  -> point count: 10766  -  2021-02-24 00:37:15.238141  elapsed: 752277 microseconds
dradius: 1.5 k: 19  -> point count: 10828  -  2021-02-24 00:37:19.088647  elapsed: 850407 microseconds
dradius: 1.5 k: 19  -> point count: 10838  -  2021-02-24 00:37:23.047502  elapsed: 958759 microseconds
dradius: 1.5 k: 19  -> point count: 10806  -  2021-02-24 00:37:27.083622  elapsed: 36026 microseconds
dradius: 1.5 k: 19  -> point count: 10816  -  2021-02-24 00:37:31.069228  elapsed: 985505 microseconds
====== dradius: 1.5 k: 19  -> point count average: 10810.8  -  2021-02-24 00:37:31.069286  average elapsed: 716594.8 microseconds
dradius: 1.5 k: 20  -> point count: 10883  -  2021-02-24 00:37:35.147996  elapsed: 78628 microseconds
dradius: 1.5 k: 20  -> point count: 10890  -  2021-02-24 00:37:39.142707  elapsed: 994612 microseconds
dradius: 1.5 k: 20  -> point count: 10890  -  2021-02-24 00:37:43.208239  elapsed: 65430 microseconds
dradius: 1.5 k: 20  -> point count: 10802  -  2021-02-24 00:37:47.237783  elapsed: 29442 microseconds
dradius: 1.5 k: 20  -> point count: 10809  -  2021-02-24 00:37:51.240396  elapsed: 2504 microseconds
====== dradius: 1.5 k: 20  -> point count average: 10854.8  -  2021-02-24 00:37:51.240451  average elapsed: 234123.2 microseconds
dradius: 1.5 k: 21  -> point count: 10849  -  2021-02-24 00:37:55.298206  elapsed: 57680 microseconds
dradius: 1.5 k: 21  -> point count: 10870  -  2021-02-24 00:37:59.406647  elapsed: 108344 microseconds
dradius: 1.5 k: 21  -> point count: 10911  -  2021-02-24 00:38:03.649387  elapsed: 242635 microseconds
dradius: 1.5 k: 21  -> point count: 10934  -  2021-02-24 00:38:07.724516  elapsed: 74999 microseconds
dradius: 1.5 k: 21  -> point count: 10838  -  2021-02-24 00:38:11.711291  elapsed: 986679 microseconds
====== dradius: 1.5 k: 21  -> point count average: 10880.4  -  2021-02-24 00:38:11.711346  average elapsed: 294067.4 microseconds
dradius: 1.5 k: 22  -> point count: 10937  -  2021-02-24 00:38:15.753518  elapsed: 42098 microseconds
dradius: 1.5 k: 22  -> point count: 10900  -  2021-02-24 00:38:19.818731  elapsed: 65108 microseconds
dradius: 1.5 k: 22  -> point count: 10961  -  2021-02-24 00:38:24.013799  elapsed: 194972 microseconds
dradius: 1.5 k: 22  -> point count: 10892  -  2021-02-24 00:38:28.166746  elapsed: 152848 microseconds
dradius: 1.5 k: 22  -> point count: 10884  -  2021-02-24 00:38:32.263965  elapsed: 97123 microseconds
====== dradius: 1.5 k: 22  -> point count average: 10914.8  -  2021-02-24 00:38:32.264019  average elapsed: 110429.8 microseconds
dradius: 1.5 k: 23  -> point count: 10950  -  2021-02-24 00:38:36.445863  elapsed: 181767 microseconds
dradius: 1.5 k: 23  -> point count: 10989  -  2021-02-24 00:38:40.496928  elapsed: 50957 microseconds
dradius: 1.5 k: 23  -> point count: 10954  -  2021-02-24 00:38:44.753821  elapsed: 256797 microseconds
dradius: 1.5 k: 23  -> point count: 10938  -  2021-02-24 00:38:49.260517  elapsed: 506596 microseconds
dradius: 1.5 k: 23  -> point count: 10932  -  2021-02-24 00:38:53.706232  elapsed: 445610 microseconds
====== dradius: 1.5 k: 23  -> point count average: 10952.6  -  2021-02-24 00:38:53.706292  average elapsed: 288345.4 microseconds
dradius: 1.5 k: 24  -> point count: 10981  -  2021-02-24 00:38:58.012933  elapsed: 306544 microseconds
dradius: 1.5 k: 24  -> point count: 10991  -  2021-02-24 00:39:02.251644  elapsed: 238610 microseconds
dradius: 1.5 k: 24  -> point count: 10973  -  2021-02-24 00:39:06.490883  elapsed: 239135 microseconds
dradius: 1.5 k: 24  -> point count: 10996  -  2021-02-24 00:39:10.809241  elapsed: 318256 microseconds
dradius: 1.5 k: 24  -> point count: 10991  -  2021-02-24 00:39:15.124543  elapsed: 315196 microseconds
====== dradius: 1.5 k: 24  -> point count average: 10986.4  -  2021-02-24 00:39:15.124599  average elapsed: 283548.2 microseconds
dradius: 1.5 k: 25  -> point count: 10921  -  2021-02-24 00:39:19.704012  elapsed: 579336 microseconds
dradius: 1.5 k: 25  -> point count: 10947  -  2021-02-24 00:39:24.221481  elapsed: 517365 microseconds
dradius: 1.5 k: 25  -> point count: 11012  -  2021-02-24 00:39:28.816913  elapsed: 595296 microseconds
dradius: 1.5 k: 25  -> point count: 10986  -  2021-02-24 00:39:33.299992  elapsed: 482979 microseconds
dradius: 1.5 k: 25  -> point count: 10988  -  2021-02-24 00:39:37.862517  elapsed: 562424 microseconds
====== dradius: 1.5 k: 25  -> point count average: 10970.8  -  2021-02-24 00:39:37.862578  average elapsed: 547480.0 microseconds
dradius: 1.5 k: 26  -> point count: 11015  -  2021-02-24 00:39:42.541848  elapsed: 679189 microseconds
dradius: 1.5 k: 26  -> point count: 11040  -  2021-02-24 00:39:47.166240  elapsed: 624291 microseconds
dradius: 1.5 k: 26  -> point count: 11038  -  2021-02-24 00:39:51.824350  elapsed: 658008 microseconds
dradius: 1.5 k: 26  -> point count: 11019  -  2021-02-24 00:39:56.417864  elapsed: 593412 microseconds
dradius: 1.5 k: 26  -> point count: 10997  -  2021-02-24 00:40:00.993564  elapsed: 575554 microseconds
====== dradius: 1.5 k: 26  -> point count average: 11021.8  -  2021-02-24 00:40:00.993620  average elapsed: 626090.8 microseconds
dradius: 1.5 k: 27  -> point count: 11030  -  2021-02-24 00:40:05.900322  elapsed: 906625 microseconds
dradius: 1.5 k: 27  -> point count: 11038  -  2021-02-24 00:40:11.085871  elapsed: 185450 microseconds
dradius: 1.5 k: 27  -> point count: 11039  -  2021-02-24 00:40:15.884207  elapsed: 798219 microseconds
dradius: 1.5 k: 27  -> point count: 11082  -  2021-02-24 00:40:20.523163  elapsed: 638845 microseconds
dradius: 1.5 k: 27  -> point count: 11013  -  2021-02-24 00:40:25.069214  elapsed: 545953 microseconds
====== dradius: 1.5 k: 27  -> point count average: 11040.4  -  2021-02-24 00:40:25.069270  average elapsed: 615018.4 microseconds
dradius: 1.5 k: 28  -> point count: 11059  -  2021-02-24 00:40:29.832123  elapsed: 762766 microseconds
dradius: 1.5 k: 28  -> point count: 11071  -  2021-02-24 00:40:34.503520  elapsed: 671291 microseconds
dradius: 1.5 k: 28  -> point count: 11077  -  2021-02-24 00:40:39.218429  elapsed: 714767 microseconds
dradius: 1.5 k: 28  -> point count: 11050  -  2021-02-24 00:40:43.941955  elapsed: 723423 microseconds
dradius: 1.5 k: 28  -> point count: 11126  -  2021-02-24 00:40:49.018280  elapsed: 76222 microseconds
====== dradius: 1.5 k: 28  -> point count average: 11076.6  -  2021-02-24 00:40:49.018335  average elapsed: 589693.8 microseconds
dradius: 1.5 k: 29  -> point count: 11090  -  2021-02-24 00:40:53.971147  elapsed: 952734 microseconds
dradius: 1.5 k: 29  -> point count: 11064  -  2021-02-24 00:40:58.773191  elapsed: 801939 microseconds
dradius: 1.5 k: 29  -> point count: 11137  -  2021-02-24 00:41:03.711913  elapsed: 938617 microseconds
dradius: 1.5 k: 29  -> point count: 11070  -  2021-02-24 00:41:08.473790  elapsed: 761737 microseconds
dradius: 1.5 k: 29  -> point count: 11104  -  2021-02-24 00:41:13.396151  elapsed: 922229 microseconds
====== dradius: 1.5 k: 29  -> point count average: 11093.0  -  2021-02-24 00:41:13.396223  average elapsed: 875451.2 microseconds
dradius: 1.5 k: 30  -> point count: 11132  -  2021-02-24 00:41:18.452852  elapsed: 56546 microseconds
dradius: 1.5 k: 30  -> point count: 11103  -  2021-02-24 00:41:23.520222  elapsed: 67272 microseconds
dradius: 1.5 k: 30  -> point count: 11058  -  2021-02-24 00:41:28.486001  elapsed: 965679 microseconds
dradius: 1.5 k: 30  -> point count: 11078  -  2021-02-24 00:41:33.514412  elapsed: 28310 microseconds
dradius: 1.5 k: 30  -> point count: 11117  -  2021-02-24 00:41:38.592581  elapsed: 78029 microseconds
====== dradius: 1.5 k: 30  -> point count average: 11097.6  -  2021-02-24 00:41:38.592637  average elapsed: 239167.2 microseconds
dradius: 1.5 k: 31  -> point count: 11124  -  2021-02-24 00:41:43.742481  elapsed: 149764 microseconds
dradius: 1.5 k: 31  -> point count: 11144  -  2021-02-24 00:41:48.796122  elapsed: 53504 microseconds
dradius: 1.5 k: 31  -> point count: 11100  -  2021-02-24 00:41:54.003615  elapsed: 207387 microseconds
dradius: 1.5 k: 31  -> point count: 11132  -  2021-02-24 00:41:59.223372  elapsed: 219649 microseconds
dradius: 1.5 k: 31  -> point count: 11162  -  2021-02-24 00:42:04.252556  elapsed: 29077 microseconds
====== dradius: 1.5 k: 31  -> point count average: 11132.4  -  2021-02-24 00:42:04.252613  average elapsed: 131876.2 microseconds
dradius: 1.5 k: 32  -> point count: 11161  -  2021-02-24 00:42:09.389817  elapsed: 137123 microseconds
dradius: 1.5 k: 32  -> point count: 11127  -  2021-02-24 00:42:14.497359  elapsed: 107442 microseconds
dradius: 1.5 k: 32  -> point count: 11200  -  2021-02-24 00:42:19.612047  elapsed: 114588 microseconds
dradius: 1.5 k: 32  -> point count: 11155  -  2021-02-24 00:42:24.696340  elapsed: 84195 microseconds
dradius: 1.5 k: 32  -> point count: 11112  -  2021-02-24 00:42:29.638388  elapsed: 941939 microseconds
====== dradius: 1.5 k: 32  -> point count average: 11151.0  -  2021-02-24 00:42:29.638442  average elapsed: 277057.4 microseconds
dradius: 1.5 k: 33  -> point count: 11145  -  2021-02-24 00:42:34.727767  elapsed: 89241 microseconds
dradius: 1.5 k: 33  -> point count: 11170  -  2021-02-24 00:42:39.945386  elapsed: 217516 microseconds
dradius: 1.5 k: 33  -> point count: 11171  -  2021-02-24 00:42:45.045674  elapsed: 100184 microseconds
dradius: 1.5 k: 33  -> point count: 11130  -  2021-02-24 00:42:50.335792  elapsed: 290007 microseconds
dradius: 1.5 k: 33  -> point count: 11161  -  2021-02-24 00:42:55.909809  elapsed: 573896 microseconds
====== dradius: 1.5 k: 33  -> point count average: 11155.4  -  2021-02-24 00:42:55.909862  average elapsed: 254168.8 microseconds
dradius: 1.5 k: 34  -> point count: 11127  -  2021-02-24 00:43:01.197959  elapsed: 288016 microseconds
dradius: 1.5 k: 34  -> point count: 11194  -  2021-02-24 00:43:06.377738  elapsed: 179683 microseconds
dradius: 1.5 k: 34  -> point count: 11175  -  2021-02-24 00:43:11.889197  elapsed: 511360 microseconds
dradius: 1.5 k: 34  -> point count: 11189  -  2021-02-24 00:43:17.403766  elapsed: 514461 microseconds
dradius: 1.5 k: 34  -> point count: 11196  -  2021-02-24 00:43:22.776630  elapsed: 372761 microseconds
====== dradius: 1.5 k: 34  -> point count average: 11176.2  -  2021-02-24 00:43:22.776684  average elapsed: 373256.2 microseconds
dradius: 1.5 k: 35  -> point count: 11159  -  2021-02-24 00:43:28.209749  elapsed: 432986 microseconds
dradius: 1.5 k: 35  -> point count: 11207  -  2021-02-24 00:43:33.768206  elapsed: 558329 microseconds
dradius: 1.5 k: 35  -> point count: 11211  -  2021-02-24 00:43:39.336477  elapsed: 568170 microseconds
dradius: 1.5 k: 35  -> point count: 11172  -  2021-02-24 00:43:44.737219  elapsed: 400644 microseconds
dradius: 1.5 k: 35  -> point count: 11183  -  2021-02-24 00:43:50.238906  elapsed: 501589 microseconds
====== dradius: 1.5 k: 35  -> point count average: 11186.4  -  2021-02-24 00:43:50.238960  average elapsed: 492343.6 microseconds
dradius: 1.5 k: 36  -> point count: 11207  -  2021-02-24 00:43:55.845027  elapsed: 605990 microseconds
dradius: 1.5 k: 36  -> point count: 11182  -  2021-02-24 00:44:01.076271  elapsed: 231144 microseconds
dradius: 1.5 k: 36  -> point count: 11218  -  2021-02-24 00:44:06.492142  elapsed: 415777 microseconds
dradius: 1.5 k: 36  -> point count: 11251  -  2021-02-24 00:44:11.955858  elapsed: 463614 microseconds
dradius: 1.5 k: 36  -> point count: 11197  -  2021-02-24 00:44:17.389546  elapsed: 433596 microseconds
====== dradius: 1.5 k: 36  -> point count average: 11211.0  -  2021-02-24 00:44:17.389599  average elapsed: 430024.2 microseconds
dradius: 1.5 k: 37  -> point count: 11269  -  2021-02-24 00:44:23.027890  elapsed: 638216 microseconds
dradius: 1.5 k: 37  -> point count: 11222  -  2021-02-24 00:44:28.547022  elapsed: 519028 microseconds
dradius: 1.5 k: 37  -> point count: 11267  -  2021-02-24 00:44:34.225939  elapsed: 678816 microseconds
dradius: 1.5 k: 37  -> point count: 11232  -  2021-02-24 00:44:39.722496  elapsed: 496456 microseconds
dradius: 1.5 k: 37  -> point count: 11253  -  2021-02-24 00:44:45.414596  elapsed: 691988 microseconds
====== dradius: 1.5 k: 37  -> point count average: 11248.6  -  2021-02-24 00:44:45.414654  average elapsed: 604900.8 microseconds
dradius: 1.5 k: 38  -> point count: 11198  -  2021-02-24 00:44:51.523608  elapsed: 108871 microseconds
dradius: 1.5 k: 38  -> point count: 11245  -  2021-02-24 00:44:57.136008  elapsed: 612298 microseconds
dradius: 1.5 k: 38  -> point count: 11253  -  2021-02-24 00:45:02.704092  elapsed: 567990 microseconds
dradius: 1.5 k: 38  -> point count: 11255  -  2021-02-24 00:45:08.580186  elapsed: 875997 microseconds
dradius: 1.5 k: 38  -> point count: 11296  -  2021-02-24 00:45:14.415638  elapsed: 835344 microseconds
====== dradius: 1.5 k: 38  -> point count average: 11249.4  -  2021-02-24 00:45:14.415694  average elapsed: 600100.0 microseconds
dradius: 1.5 k: 39  -> point count: 11321  -  2021-02-24 00:45:20.242033  elapsed: 826264 microseconds
dradius: 1.5 k: 39  -> point count: 11264  -  2021-02-24 00:45:26.131856  elapsed: 889720 microseconds
dradius: 1.5 k: 39  -> point count: 11264  -  2021-02-24 00:45:32.093575  elapsed: 961619 microseconds
dradius: 1.5 k: 39  -> point count: 11270  -  2021-02-24 00:45:37.952220  elapsed: 858543 microseconds
dradius: 1.5 k: 39  -> point count: 11252  -  2021-02-24 00:45:43.959446  elapsed: 7127 microseconds
====== dradius: 1.5 k: 39  -> point count average: 11274.2  -  2021-02-24 00:45:43.959529  average elapsed: 708654.6 microseconds
dradius: 1.5 k: 40  -> point count: 11237  -  2021-02-24 00:45:49.949841  elapsed: 990199 microseconds
dradius: 1.5 k: 40  -> point count: 11257  -  2021-02-24 00:45:55.817045  elapsed: 867104 microseconds
dradius: 1.5 k: 40  -> point count: 11276  -  2021-02-24 00:46:01.700704  elapsed: 883540 microseconds
dradius: 1.5 k: 40  -> point count: 11264  -  2021-02-24 00:46:07.414238  elapsed: 713440 microseconds
dradius: 1.5 k: 40  -> point count: 11290  -  2021-02-24 00:46:13.158705  elapsed: 744361 microseconds
====== dradius: 1.5 k: 40  -> point count average: 11264.8  -  2021-02-24 00:46:13.158758  average elapsed: 839728.8 microseconds
dradius: 1.5 k: 41  -> point count: 11266  -  2021-02-24 00:46:19.040992  elapsed: 882154 microseconds
dradius: 1.5 k: 41  -> point count: 11328  -  2021-02-24 00:46:24.924278  elapsed: 883184 microseconds
dradius: 1.5 k: 41  -> point count: 11289  -  2021-02-24 00:46:30.843693  elapsed: 919307 microseconds
dradius: 1.5 k: 41  -> point count: 11372  -  2021-02-24 00:46:36.904911  elapsed: 61055 microseconds
dradius: 1.5 k: 41  -> point count: 11324  -  2021-02-24 00:46:42.862257  elapsed: 957238 microseconds
====== dradius: 1.5 k: 41  -> point count average: 11315.8  -  2021-02-24 00:46:42.862317  average elapsed: 740587.6 microseconds
dradius: 1.5 k: 42  -> point count: 11245  -  2021-02-24 00:46:49.049402  elapsed: 186998 microseconds
dradius: 1.5 k: 42  -> point count: 11361  -  2021-02-24 00:46:55.469455  elapsed: 419944 microseconds
dradius: 1.5 k: 42  -> point count: 11364  -  2021-02-24 00:47:01.611323  elapsed: 141769 microseconds
dradius: 1.5 k: 42  -> point count: 11279  -  2021-02-24 00:47:07.794342  elapsed: 182920 microseconds
dradius: 1.5 k: 42  -> point count: 11298  -  2021-02-24 00:47:14.136052  elapsed: 341600 microseconds
====== dradius: 1.5 k: 42  -> point count average: 11309.4  -  2021-02-24 00:47:14.136107  average elapsed: 254646.2 microseconds
dradius: 1.5 k: 43  -> point count: 11347  -  2021-02-24 00:47:20.438254  elapsed: 302069 microseconds
dradius: 1.5 k: 43  -> point count: 11299  -  2021-02-24 00:47:26.691621  elapsed: 253263 microseconds
dradius: 1.5 k: 43  -> point count: 11310  -  2021-02-24 00:47:33.173226  elapsed: 481484 microseconds
dradius: 1.5 k: 43  -> point count: 11298  -  2021-02-24 00:47:39.425495  elapsed: 252168 microseconds
dradius: 1.5 k: 43  -> point count: 11355  -  2021-02-24 00:47:45.557985  elapsed: 132393 microseconds
====== dradius: 1.5 k: 43  -> point count average: 11321.8  -  2021-02-24 00:47:45.558040  average elapsed: 284275.4 microseconds
dradius: 1.5 k: 44  -> point count: 11315  -  2021-02-24 00:47:51.582550  elapsed: 24433 microseconds
dradius: 1.5 k: 44  -> point count: 11349  -  2021-02-24 00:47:57.643255  elapsed: 60606 microseconds
dradius: 1.5 k: 44  -> point count: 11337  -  2021-02-24 00:48:03.774927  elapsed: 131577 microseconds
dradius: 1.5 k: 44  -> point count: 11324  -  2021-02-24 00:48:10.086928  elapsed: 311896 microseconds
dradius: 1.5 k: 44  -> point count: 11312  -  2021-02-24 00:48:16.292801  elapsed: 205768 microseconds
====== dradius: 1.5 k: 44  -> point count average: 11327.4  -  2021-02-24 00:48:16.292855  average elapsed: 146856.0 microseconds
dradius: 1.5 k: 45  -> point count: 11325  -  2021-02-24 00:48:22.454386  elapsed: 161450 microseconds
dradius: 1.5 k: 45  -> point count: 11334  -  2021-02-24 00:48:28.816466  elapsed: 361972 microseconds
dradius: 1.5 k: 45  -> point count: 11365  -  2021-02-24 00:48:35.271734  elapsed: 455169 microseconds
dradius: 1.5 k: 45  -> point count: 11332  -  2021-02-24 00:48:41.435548  elapsed: 163711 microseconds
dradius: 1.5 k: 45  -> point count: 11353  -  2021-02-24 00:48:48.097613  elapsed: 661970 microseconds
====== dradius: 1.5 k: 45  -> point count average: 11341.8  -  2021-02-24 00:48:48.097669  average elapsed: 360854.4 microseconds
dradius: 1.5 k: 46  -> point count: 11345  -  2021-02-24 00:48:54.593039  elapsed: 495294 microseconds
dradius: 1.5 k: 46  -> point count: 11334  -  2021-02-24 00:49:01.102277  elapsed: 509142 microseconds
dradius: 1.5 k: 46  -> point count: 11278  -  2021-02-24 00:49:07.550651  elapsed: 448270 microseconds
dradius: 1.5 k: 46  -> point count: 11389  -  2021-02-24 00:49:14.183365  elapsed: 632605 microseconds
dradius: 1.5 k: 46  -> point count: 11371  -  2021-02-24 00:49:20.799421  elapsed: 615945 microseconds
====== dradius: 1.5 k: 46  -> point count average: 11343.4  -  2021-02-24 00:49:20.799475  average elapsed: 540251.2 microseconds
dradius: 1.5 k: 47  -> point count: 11310  -  2021-02-24 00:49:27.426720  elapsed: 627152 microseconds
dradius: 1.5 k: 47  -> point count: 11381  -  2021-02-24 00:49:34.771873  elapsed: 345043 microseconds
dradius: 1.5 k: 47  -> point count: 11407  -  2021-02-24 00:49:41.324604  elapsed: 552626 microseconds
dradius: 1.5 k: 47  -> point count: 11359  -  2021-02-24 00:49:47.650951  elapsed: 326246 microseconds
dradius: 1.5 k: 47  -> point count: 11377  -  2021-02-24 00:49:53.880712  elapsed: 229664 microseconds
====== dradius: 1.5 k: 47  -> point count average: 11366.8  -  2021-02-24 00:49:53.880769  average elapsed: 416146.2 microseconds
dradius: 1.5 k: 48  -> point count: 11415  -  2021-02-24 00:50:00.359335  elapsed: 478489 microseconds
dradius: 1.5 k: 48  -> point count: 11356  -  2021-02-24 00:50:07.115887  elapsed: 756449 microseconds
dradius: 1.5 k: 48  -> point count: 11368  -  2021-02-24 00:50:14.357291  elapsed: 241248 microseconds
dradius: 1.5 k: 48  -> point count: 11357  -  2021-02-24 00:50:21.420025  elapsed: 62630 microseconds
dradius: 1.5 k: 48  -> point count: 11363  -  2021-02-24 00:50:28.928873  elapsed: 508720 microseconds
====== dradius: 1.5 k: 48  -> point count average: 11371.8  -  2021-02-24 00:50:28.928931  average elapsed: 409507.2 microseconds
dradius: 1.5 k: 49  -> point count: 11383  -  2021-02-24 00:50:37.031987  elapsed: 102974 microseconds
dradius: 1.5 k: 49  -> point count: 11407  -  2021-02-24 00:50:44.531855  elapsed: 499747 microseconds
dradius: 1.5 k: 49  -> point count: 11402  -  2021-02-24 00:50:54.514977  elapsed: 982966 microseconds
dradius: 1.5 k: 49  -> point count: 11451  -  2021-02-24 00:51:01.715639  elapsed: 200554 microseconds
dradius: 1.5 k: 49  -> point count: 11368  -  2021-02-24 00:51:10.326627  elapsed: 610887 microseconds
====== dradius: 1.5 k: 49  -> point count average: 11402.2  -  2021-02-24 00:51:10.326685  average elapsed: 479425.6 microseconds
dradius: 1.5 k: 50  -> point count: 11390  -  2021-02-24 00:51:19.159395  elapsed: 832619 microseconds
dradius: 1.5 k: 50  -> point count: 11417  -  2021-02-24 00:51:28.894463  elapsed: 734921 microseconds
dradius: 1.5 k: 50  -> point count: 11465  -  2021-02-24 00:51:36.694715  elapsed: 800135 microseconds
dradius: 1.5 k: 50  -> point count: 11388  -  2021-02-24 00:51:44.556308  elapsed: 861493 microseconds
dradius: 1.5 k: 50  -> point count: 11342  -  2021-02-24 00:51:51.847051  elapsed: 290617 microseconds
====== dradius: 1.5 k: 50  -> point count average: 11400.4  -  2021-02-24 00:51:51.847142  average elapsed: 703957.0 microseconds
dradius: 1.5 k: 51  -> point count: 11449  -  2021-02-24 00:51:59.070208  elapsed: 222950 microseconds
dradius: 1.5 k: 51  -> point count: 11469  -  2021-02-24 00:52:06.751520  elapsed: 681209 microseconds
dradius: 1.5 k: 51  -> point count: 11412  -  2021-02-24 00:52:14.524692  elapsed: 773066 microseconds
dradius: 1.5 k: 51  -> point count: 11418  -  2021-02-24 00:52:22.306141  elapsed: 781339 microseconds
dradius: 1.5 k: 51  -> point count: 11428  -  2021-02-24 00:52:29.736005  elapsed: 429753 microseconds
====== dradius: 1.5 k: 51  -> point count average: 11435.2  -  2021-02-24 00:52:29.736060  average elapsed: 577663.4 microseconds
dradius: 1.5 k: 52  -> point count: 11449  -  2021-02-24 00:52:37.387816  elapsed: 651661 microseconds
dradius: 1.5 k: 52  -> point count: 11505  -  2021-02-24 00:52:45.047101  elapsed: 659168 microseconds
dradius: 1.5 k: 52  -> point count: 11406  -  2021-02-24 00:52:52.817967  elapsed: 770730 microseconds
dradius: 1.5 k: 52  -> point count: 11434  -  2021-02-24 00:53:00.117659  elapsed: 299586 microseconds
dradius: 1.5 k: 52  -> point count: 11424  -  2021-02-24 00:53:07.431914  elapsed: 314156 microseconds
====== dradius: 1.5 k: 52  -> point count average: 11443.6  -  2021-02-24 00:53:07.431977  average elapsed: 539060.2 microseconds
dradius: 1.5 k: 53  -> point count: 11385  -  2021-02-24 00:53:14.794975  elapsed: 362910 microseconds
dradius: 1.5 k: 53  -> point count: 11424  -  2021-02-24 00:53:22.033484  elapsed: 238411 microseconds
dradius: 1.5 k: 53  -> point count: 11379  -  2021-02-24 00:53:29.126336  elapsed: 92756 microseconds
dradius: 1.5 k: 53  -> point count: 11401  -  2021-02-24 00:53:36.220915  elapsed: 94483 microseconds
dradius: 1.5 k: 53  -> point count: 11443  -  2021-02-24 00:53:43.396057  elapsed: 175043 microseconds
====== dradius: 1.5 k: 53  -> point count average: 11406.4  -  2021-02-24 00:53:43.396111  average elapsed: 192720.6 microseconds
dradius: 1.5 k: 54  -> point count: 11464  -  2021-02-24 00:53:50.875526  elapsed: 479338 microseconds
dradius: 1.5 k: 54  -> point count: 11447  -  2021-02-24 00:54:00.047756  elapsed: 172088 microseconds
dradius: 1.5 k: 54  -> point count: 11427  -  2021-02-24 00:54:08.234318  elapsed: 186436 microseconds
dradius: 1.5 k: 54  -> point count: 11458  -  2021-02-24 00:54:16.210537  elapsed: 976110 microseconds
dradius: 1.5 k: 54  -> point count: 11489  -  2021-02-24 00:54:23.869237  elapsed: 658592 microseconds
====== dradius: 1.5 k: 54  -> point count average: 11457.0  -  2021-02-24 00:54:23.869311  average elapsed: 494512.8 microseconds
dradius: 1.5 k: 55  -> point count: 11451  -  2021-02-24 00:54:31.795514  elapsed: 926106 microseconds
dradius: 1.5 k: 55  -> point count: 11430  -  2021-02-24 00:54:39.349448  elapsed: 553828 microseconds
dradius: 1.5 k: 55  -> point count: 11474  -  2021-02-24 00:54:48.553660  elapsed: 204105 microseconds
dradius: 1.5 k: 55  -> point count: 11456  -  2021-02-24 00:54:58.960744  elapsed: 406961 microseconds
dradius: 1.5 k: 55  -> point count: 11440  -  2021-02-24 00:55:09.996547  elapsed: 35686 microseconds
====== dradius: 1.5 k: 55  -> point count average: 11450.2  -  2021-02-24 00:55:09.996608  average elapsed: 425337.2 microseconds
dradius: 1.5 k: 56  -> point count: 11458  -  2021-02-24 00:55:18.838550  elapsed: 841838 microseconds
dradius: 1.5 k: 56  -> point count: 11448  -  2021-02-24 00:55:28.530959  elapsed: 692292 microseconds
dradius: 1.5 k: 56  -> point count: 11479  -  2021-02-24 00:55:39.023744  elapsed: 492667 microseconds
dradius: 1.5 k: 56  -> point count: 11468  -  2021-02-24 00:55:48.064321  elapsed: 40436 microseconds
dradius: 1.5 k: 56  -> point count: 11492  -  2021-02-24 00:55:57.902725  elapsed: 838265 microseconds
====== dradius: 1.5 k: 56  -> point count average: 11469.0  -  2021-02-24 00:55:57.902838  average elapsed: 581099.6 microseconds
dradius: 1.5 k: 57  -> point count: 11474  -  2021-02-24 00:56:08.259400  elapsed: 356398 microseconds
dradius: 1.5 k: 57  -> point count: 11479  -  2021-02-24 00:56:17.752661  elapsed: 493147 microseconds
dradius: 1.5 k: 57  -> point count: 11518  -  2021-02-24 00:56:26.437525  elapsed: 684732 microseconds
dradius: 1.5 k: 57  -> point count: 11503  -  2021-02-24 00:56:35.522941  elapsed: 85302 microseconds
dradius: 1.5 k: 57  -> point count: 11490  -  2021-02-24 00:56:43.547103  elapsed: 24049 microseconds
====== dradius: 1.5 k: 57  -> point count average: 11492.8  -  2021-02-24 00:56:43.547166  average elapsed: 328725.6 microseconds
dradius: 1.5 k: 58  -> point count: 11513  -  2021-02-24 00:56:51.905402  elapsed: 358143 microseconds
dradius: 1.5 k: 58  -> point count: 11499  -  2021-02-24 00:56:59.131402  elapsed: 225898 microseconds
dradius: 1.5 k: 58  -> point count: 11521  -  2021-02-24 00:57:06.345871  elapsed: 214364 microseconds
dradius: 1.5 k: 58  -> point count: 11461  -  2021-02-24 00:57:14.416738  elapsed: 70769 microseconds
dradius: 1.5 k: 58  -> point count: 11475  -  2021-02-24 00:57:22.686863  elapsed: 270015 microseconds
====== dradius: 1.5 k: 58  -> point count average: 11493.8  -  2021-02-24 00:57:22.686919  average elapsed: 227837.8 microseconds
dradius: 1.5 k: 59  -> point count: 11449  -  2021-02-24 00:57:30.820743  elapsed: 133740 microseconds
dradius: 1.5 k: 59  -> point count: 11458  -  2021-02-24 00:57:39.275383  elapsed: 454537 microseconds
dradius: 1.5 k: 59  -> point count: 11503  -  2021-02-24 00:57:47.994240  elapsed: 718753 microseconds
dradius: 1.5 k: 59  -> point count: 11493  -  2021-02-24 00:57:55.636181  elapsed: 641843 microseconds
dradius: 1.5 k: 59  -> point count: 11455  -  2021-02-24 00:58:04.029400  elapsed: 393123 microseconds
====== dradius: 1.5 k: 59  -> point count average: 11471.6  -  2021-02-24 00:58:04.029458  average elapsed: 468399.2 microseconds
dradius: 1.5 k: 60  -> point count: 11519  -  2021-02-24 00:58:14.231319  elapsed: 201781 microseconds
dradius: 1.5 k: 60  -> point count: 11545  -  2021-02-24 00:58:24.126432  elapsed: 895007 microseconds
dradius: 1.5 k: 60  -> point count: 11512  -  2021-02-24 00:58:32.589317  elapsed: 462756 microseconds
dradius: 1.5 k: 60  -> point count: 11476  -  2021-02-24 00:58:40.321550  elapsed: 732133 microseconds
dradius: 1.5 k: 60  -> point count: 11444  -  2021-02-24 00:58:48.045297  elapsed: 723649 microseconds
====== dradius: 1.5 k: 60  -> point count average: 11499.2  -  2021-02-24 00:58:48.045353  average elapsed: 603065.2 microseconds
dradius: 1.5 k: 61  -> point count: 11520  -  2021-02-24 00:58:56.236779  elapsed: 191322 microseconds
dradius: 1.5 k: 61  -> point count: 11508  -  2021-02-24 00:59:03.824653  elapsed: 587751 microseconds
dradius: 1.5 k: 61  -> point count: 11501  -  2021-02-24 00:59:12.367801  elapsed: 543047 microseconds
dradius: 1.5 k: 61  -> point count: 11500  -  2021-02-24 00:59:21.152267  elapsed: 784267 microseconds
dradius: 1.5 k: 61  -> point count: 11515  -  2021-02-24 00:59:29.719402  elapsed: 567034 microseconds
====== dradius: 1.5 k: 61  -> point count average: 11508.8  -  2021-02-24 00:59:29.719455  average elapsed: 534684.2 microseconds
dradius: 1.5 k: 62  -> point count: 11466  -  2021-02-24 00:59:37.533412  elapsed: 813884 microseconds
dradius: 1.5 k: 62  -> point count: 11610  -  2021-02-24 00:59:45.329970  elapsed: 796463 microseconds
dradius: 1.5 k: 62  -> point count: 11524  -  2021-02-24 00:59:53.312626  elapsed: 982562 microseconds
dradius: 1.5 k: 62  -> point count: 11532  -  2021-02-24 01:00:01.368422  elapsed: 55699 microseconds
dradius: 1.5 k: 62  -> point count: 11519  -  2021-02-24 01:00:09.545177  elapsed: 176657 microseconds
====== dradius: 1.5 k: 62  -> point count average: 11530.2  -  2021-02-24 01:00:09.545230  average elapsed: 565053.0 microseconds
dradius: 1.5 k: 63  -> point count: 11511  -  2021-02-24 01:00:17.623957  elapsed: 78648 microseconds
dradius: 1.5 k: 63  -> point count: 11469  -  2021-02-24 01:00:25.837029  elapsed: 212974 microseconds
dradius: 1.5 k: 63  -> point count: 11528  -  2021-02-24 01:00:33.852962  elapsed: 15837 microseconds
dradius: 1.5 k: 63  -> point count: 11526  -  2021-02-24 01:00:41.887234  elapsed: 34178 microseconds
dradius: 1.5 k: 63  -> point count: 11527  -  2021-02-24 01:00:51.363413  elapsed: 476035 microseconds
====== dradius: 1.5 k: 63  -> point count average: 11512.2  -  2021-02-24 01:00:51.363473  average elapsed: 163534.4 microseconds
dradius: 1.5 k: 64  -> point count: 11528  -  2021-02-24 01:01:00.656351  elapsed: 292794 microseconds
dradius: 1.5 k: 64  -> point count: 11535  -  2021-02-24 01:01:08.566181  elapsed: 909729 microseconds
dradius: 1.5 k: 64  -> point count: 11568  -  2021-02-24 01:01:16.651895  elapsed: 85620 microseconds
dradius: 1.5 k: 64  -> point count: 11549  -  2021-02-24 01:01:24.637536  elapsed: 985545 microseconds
dradius: 1.5 k: 64  -> point count: 11525  -  2021-02-24 01:01:32.697108  elapsed: 59467 microseconds
====== dradius: 1.5 k: 64  -> point count average: 11541.0  -  2021-02-24 01:01:32.697161  average elapsed: 466631.0 microseconds
dradius: 1.5 k: 65  -> point count: 11502  -  2021-02-24 01:01:41.168288  elapsed: 471051 microseconds
dradius: 1.5 k: 65  -> point count: 11528  -  2021-02-24 01:01:50.284838  elapsed: 116451 microseconds
dradius: 1.5 k: 65  -> point count: 11555  -  2021-02-24 01:02:00.248885  elapsed: 963943 microseconds
dradius: 1.5 k: 65  -> point count: 11523  -  2021-02-24 01:02:09.183199  elapsed: 934210 microseconds
dradius: 1.5 k: 65  -> point count: 11535  -  2021-02-24 01:02:17.593444  elapsed: 410138 microseconds
====== dradius: 1.5 k: 65  -> point count average: 11528.6  -  2021-02-24 01:02:17.593498  average elapsed: 579158.6 microseconds
dradius: 1.5 k: 66  -> point count: 11567  -  2021-02-24 01:02:25.924041  elapsed: 330471 microseconds
dradius: 1.5 k: 66  -> point count: 11510  -  2021-02-24 01:02:35.129136  elapsed: 205001 microseconds
dradius: 1.5 k: 66  -> point count: 11522  -  2021-02-24 01:02:44.609549  elapsed: 480311 microseconds
dradius: 1.5 k: 66  -> point count: 11559  -  2021-02-24 01:02:54.193555  elapsed: 583906 microseconds
dradius: 1.5 k: 66  -> point count: 11512  -  2021-02-24 01:03:03.723782  elapsed: 530128 microseconds
====== dradius: 1.5 k: 66  -> point count average: 11534.0  -  2021-02-24 01:03:03.723844  average elapsed: 425963.4 microseconds
dradius: 1.5 k: 67  -> point count: 11520  -  2021-02-24 01:03:14.025267  elapsed: 301333 microseconds
dradius: 1.5 k: 67  -> point count: 11582  -  2021-02-24 01:03:23.607291  elapsed: 581851 microseconds
dradius: 1.5 k: 67  -> point count: 11548  -  2021-02-24 01:03:33.371199  elapsed: 763779 microseconds
dradius: 1.5 k: 67  -> point count: 11531  -  2021-02-24 01:03:43.349068  elapsed: 977755 microseconds
dradius: 1.5 k: 67  -> point count: 11573  -  2021-02-24 01:03:53.751292  elapsed: 402115 microseconds
====== dradius: 1.5 k: 67  -> point count average: 11550.8  -  2021-02-24 01:03:53.751353  average elapsed: 605366.6 microseconds
dradius: 1.5 k: 68  -> point count: 11567  -  2021-02-24 01:04:03.865512  elapsed: 114078 microseconds
dradius: 1.5 k: 68  -> point count: 11520  -  2021-02-24 01:04:13.438574  elapsed: 572950 microseconds
dradius: 1.5 k: 68  -> point count: 11569  -  2021-02-24 01:04:22.115887  elapsed: 677210 microseconds
dradius: 1.5 k: 68  -> point count: 11564  -  2021-02-24 01:04:31.583427  elapsed: 467444 microseconds
dradius: 1.5 k: 68  -> point count: 11546  -  2021-02-24 01:04:41.072564  elapsed: 489038 microseconds
====== dradius: 1.5 k: 68  -> point count average: 11553.2  -  2021-02-24 01:04:41.072648  average elapsed: 464144.0 microseconds
dradius: 1.5 k: 69  -> point count: 11537  -  2021-02-24 01:04:51.365431  elapsed: 292659 microseconds
dradius: 1.5 k: 69  -> point count: 11569  -  2021-02-24 01:05:01.327674  elapsed: 962140 microseconds
dradius: 1.5 k: 69  -> point count: 11573  -  2021-02-24 01:05:11.621443  elapsed: 293660 microseconds
dradius: 1.5 k: 69  -> point count: 11563  -  2021-02-24 01:05:21.992402  elapsed: 370849 microseconds
dradius: 1.5 k: 69  -> point count: 11595  -  2021-02-24 01:05:31.185054  elapsed: 192543 microseconds
====== dradius: 1.5 k: 69  -> point count average: 11567.4  -  2021-02-24 01:05:31.185121  average elapsed: 422370.2 microseconds
dradius: 1.5 k: 70  -> point count: 11575  -  2021-02-24 01:05:40.262767  elapsed: 77541 microseconds
dradius: 1.5 k: 70  -> point count: 11560  -  2021-02-24 01:05:48.656023  elapsed: 393155 microseconds
dradius: 1.5 k: 70  -> point count: 11621  -  2021-02-24 01:05:57.191441  elapsed: 535317 microseconds
dradius: 1.5 k: 70  -> point count: 11550  -  2021-02-24 01:06:06.491468  elapsed: 299934 microseconds
dradius: 1.5 k: 70  -> point count: 11512  -  2021-02-24 01:06:16.749758  elapsed: 258149 microseconds
====== dradius: 1.5 k: 70  -> point count average: 11563.6  -  2021-02-24 01:06:16.749855  average elapsed: 312819.2 microseconds
dradius: 1.5 k: 71  -> point count: 11605  -  2021-02-24 01:06:26.803112  elapsed: 53165 microseconds
dradius: 1.5 k: 71  -> point count: 11549  -  2021-02-24 01:06:37.529388  elapsed: 726173 microseconds
dradius: 1.5 k: 71  -> point count: 11569  -  2021-02-24 01:06:49.771661  elapsed: 242150 microseconds
dradius: 1.5 k: 71  -> point count: 11559  -  2021-02-24 01:07:02.132801  elapsed: 361036 microseconds
dradius: 1.5 k: 71  -> point count: 11587  -  2021-02-24 01:07:12.734498  elapsed: 601591 microseconds
====== dradius: 1.5 k: 71  -> point count average: 11573.8  -  2021-02-24 01:07:12.734554  average elapsed: 396823.0 microseconds
dradius: 1.5 k: 72  -> point count: 11622  -  2021-02-24 01:07:22.647530  elapsed: 912897 microseconds
dradius: 1.5 k: 72  -> point count: 11540  -  2021-02-24 01:07:32.858325  elapsed: 210693 microseconds
dradius: 1.5 k: 72  -> point count: 11576  -  2021-02-24 01:07:43.462162  elapsed: 603732 microseconds
dradius: 1.5 k: 72  -> point count: 11566  -  2021-02-24 01:07:53.516984  elapsed: 54719 microseconds
dradius: 1.5 k: 72  -> point count: 11585  -  2021-02-24 01:08:03.146552  elapsed: 629466 microseconds
====== dradius: 1.5 k: 72  -> point count average: 11577.8  -  2021-02-24 01:08:03.146608  average elapsed: 482301.4 microseconds
dradius: 1.5 k: 73  -> point count: 11626  -  2021-02-24 01:08:13.211518  elapsed: 64833 microseconds
dradius: 1.5 k: 73  -> point count: 11558  -  2021-02-24 01:08:23.265927  elapsed: 54305 microseconds
dradius: 1.5 k: 73  -> point count: 11564  -  2021-02-24 01:08:33.108771  elapsed: 842747 microseconds
dradius: 1.5 k: 73  -> point count: 11600  -  2021-02-24 01:08:42.969811  elapsed: 860939 microseconds
dradius: 1.5 k: 73  -> point count: 11606  -  2021-02-24 01:08:53.623771  elapsed: 653856 microseconds
====== dradius: 1.5 k: 73  -> point count average: 11590.8  -  2021-02-24 01:08:53.623828  average elapsed: 495336.0 microseconds
dradius: 1.5 k: 74  -> point count: 11605  -  2021-02-24 01:09:03.745136  elapsed: 121228 microseconds
dradius: 1.5 k: 74  -> point count: 11623  -  2021-02-24 01:09:14.331711  elapsed: 586470 microseconds
dradius: 1.5 k: 74  -> point count: 11616  -  2021-02-24 01:09:26.738366  elapsed: 406542 microseconds
dradius: 1.5 k: 74  -> point count: 11619  -  2021-02-24 01:09:37.847553  elapsed: 109083 microseconds
dradius: 1.5 k: 74  -> point count: 11629  -  2021-02-24 01:09:47.910367  elapsed: 62710 microseconds
====== dradius: 1.5 k: 74  -> point count average: 11618.4  -  2021-02-24 01:09:47.910423  average elapsed: 257206.6 microseconds
dradius: 1.5 k: 75  -> point count: 11568  -  2021-02-24 01:09:57.741118  elapsed: 830617 microseconds
dradius: 1.5 k: 75  -> point count: 11612  -  2021-02-24 01:10:07.892999  elapsed: 151781 microseconds
dradius: 1.5 k: 75  -> point count: 11609  -  2021-02-24 01:10:17.830451  elapsed: 937352 microseconds
dradius: 1.5 k: 75  -> point count: 11634  -  2021-02-24 01:10:28.154107  elapsed: 323554 microseconds
dradius: 1.5 k: 75  -> point count: 11580  -  2021-02-24 01:10:40.409924  elapsed: 255707 microseconds
====== dradius: 1.5 k: 75  -> point count average: 11600.6  -  2021-02-24 01:10:40.409982  average elapsed: 499802.2 microseconds
dradius: 1.5 k: 76  -> point count: 11585  -  2021-02-24 01:10:53.248263  elapsed: 838200 microseconds
dradius: 1.5 k: 76  -> point count: 11662  -  2021-02-24 01:11:04.741951  elapsed: 493576 microseconds
dradius: 1.5 k: 76  -> point count: 11633  -  2021-02-24 01:11:16.483487  elapsed: 741420 microseconds
dradius: 1.5 k: 76  -> point count: 11619  -  2021-02-24 01:11:28.555241  elapsed: 71632 microseconds
dradius: 1.5 k: 76  -> point count: 11583  -  2021-02-24 01:11:41.208262  elapsed: 652787 microseconds
====== dradius: 1.5 k: 76  -> point count average: 11616.4  -  2021-02-24 01:11:41.208322  average elapsed: 559523.0 microseconds
dradius: 1.5 k: 77  -> point count: 11657  -  2021-02-24 01:11:53.188121  elapsed: 979709 microseconds
dradius: 1.5 k: 77  -> point count: 11624  -  2021-02-24 01:12:04.972277  elapsed: 783929 microseconds
dradius: 1.5 k: 77  -> point count: 11640  -  2021-02-24 01:12:15.926244  elapsed: 953861 microseconds
dradius: 1.5 k: 77  -> point count: 11602  -  2021-02-24 01:12:26.327371  elapsed: 401027 microseconds
dradius: 1.5 k: 77  -> point count: 11636  -  2021-02-24 01:12:37.136332  elapsed: 808848 microseconds
====== dradius: 1.5 k: 77  -> point count average: 11631.8  -  2021-02-24 01:12:37.136390  average elapsed: 785474.8 microseconds
dradius: 1.5 k: 78  -> point count: 11614  -  2021-02-24 01:12:49.146007  elapsed: 9530 microseconds
dradius: 1.5 k: 78  -> point count: 11603  -  2021-02-24 01:13:00.225845  elapsed: 79725 microseconds
dradius: 1.5 k: 78  -> point count: 11650  -  2021-02-24 01:13:10.762723  elapsed: 536770 microseconds
dradius: 1.5 k: 78  -> point count: 11638  -  2021-02-24 01:13:21.328115  elapsed: 565259 microseconds
dradius: 1.5 k: 78  -> point count: 11649  -  2021-02-24 01:13:31.653042  elapsed: 324755 microseconds
====== dradius: 1.5 k: 78  -> point count average: 11630.8  -  2021-02-24 01:13:31.653096  average elapsed: 303207.8 microseconds
dradius: 1.5 k: 79  -> point count: 11649  -  2021-02-24 01:13:41.907517  elapsed: 254338 microseconds
dradius: 1.5 k: 79  -> point count: 11640  -  2021-02-24 01:13:52.174618  elapsed: 267001 microseconds
dradius: 1.5 k: 79  -> point count: 11588  -  2021-02-24 01:14:02.542761  elapsed: 368042 microseconds
dradius: 1.5 k: 79  -> point count: 11648  -  2021-02-24 01:14:13.415892  elapsed: 873028 microseconds
dradius: 1.5 k: 79  -> point count: 11615  -  2021-02-24 01:14:25.969562  elapsed: 553558 microseconds
====== dradius: 1.5 k: 79  -> point count average: 11628.0  -  2021-02-24 01:14:25.969620  average elapsed: 463193.4 microseconds
dradius: 1.5 k: 80  -> point count: 11664  -  2021-02-24 01:14:39.570908  elapsed: 601205 microseconds
dradius: 1.5 k: 80  -> point count: 11664  -  2021-02-24 01:14:52.661699  elapsed: 90646 microseconds
dradius: 1.5 k: 80  -> point count: 11648  -  2021-02-24 01:15:03.801904  elapsed: 140092 microseconds
dradius: 1.5 k: 80  -> point count: 11673  -  2021-02-24 01:15:15.071729  elapsed: 269725 microseconds
dradius: 1.5 k: 80  -> point count: 11633  -  2021-02-24 01:15:26.137428  elapsed: 65595 microseconds
====== dradius: 1.5 k: 80  -> point count average: 11656.4  -  2021-02-24 01:15:26.137485  average elapsed: 233452.6 microseconds
dradius: 1.5 k: 81  -> point count: 11649  -  2021-02-24 01:15:36.885122  elapsed: 747559 microseconds
dradius: 1.5 k: 81  -> point count: 11656  -  2021-02-24 01:15:48.412875  elapsed: 527659 microseconds
dradius: 1.5 k: 81  -> point count: 11659  -  2021-02-24 01:15:59.549610  elapsed: 136634 microseconds
dradius: 1.5 k: 81  -> point count: 11605  -  2021-02-24 01:16:10.788515  elapsed: 238798 microseconds
dradius: 1.5 k: 81  -> point count: 11623  -  2021-02-24 01:16:21.982747  elapsed: 194124 microseconds
====== dradius: 1.5 k: 81  -> point count average: 11638.4  -  2021-02-24 01:16:21.982803  average elapsed: 368954.8 microseconds
dradius: 1.5 k: 82  -> point count: 11655  -  2021-02-24 01:16:33.257989  elapsed: 275108 microseconds
dradius: 1.5 k: 82  -> point count: 11653  -  2021-02-24 01:16:44.294794  elapsed: 36701 microseconds
dradius: 1.5 k: 82  -> point count: 11686  -  2021-02-24 01:16:55.792071  elapsed: 497173 microseconds
dradius: 1.5 k: 82  -> point count: 11675  -  2021-02-24 01:17:06.511858  elapsed: 719666 microseconds
dradius: 1.5 k: 82  -> point count: 11639  -  2021-02-24 01:17:17.409175  elapsed: 897216 microseconds
====== dradius: 1.5 k: 82  -> point count average: 11661.6  -  2021-02-24 01:17:17.409252  average elapsed: 485172.8 microseconds
dradius: 1.5 k: 83  -> point count: 11673  -  2021-02-24 01:17:28.179347  elapsed: 769976 microseconds
dradius: 1.5 k: 83  -> point count: 11643  -  2021-02-24 01:17:38.912380  elapsed: 732929 microseconds
dradius: 1.5 k: 83  -> point count: 11655  -  2021-02-24 01:17:50.288241  elapsed: 375757 microseconds
dradius: 1.5 k: 83  -> point count: 11699  -  2021-02-24 01:18:01.343338  elapsed: 54882 microseconds
dradius: 1.5 k: 83  -> point count: 11686  -  2021-02-24 01:18:12.516359  elapsed: 172921 microseconds
====== dradius: 1.5 k: 83  -> point count average: 11671.2  -  2021-02-24 01:18:12.516414  average elapsed: 421293.0 microseconds
dradius: 1.5 k: 84  -> point count: 11636  -  2021-02-24 01:18:23.699648  elapsed: 183156 microseconds
dradius: 1.5 k: 84  -> point count: 11621  -  2021-02-24 01:18:34.850626  elapsed: 150875 microseconds
dradius: 1.5 k: 84  -> point count: 11687  -  2021-02-24 01:18:46.211291  elapsed: 360568 microseconds
dradius: 1.5 k: 84  -> point count: 11652  -  2021-02-24 01:18:57.430338  elapsed: 218943 microseconds
dradius: 1.5 k: 84  -> point count: 11684  -  2021-02-24 01:19:08.087792  elapsed: 657355 microseconds
====== dradius: 1.5 k: 84  -> point count average: 11656.0  -  2021-02-24 01:19:08.087846  average elapsed: 314179.4 microseconds
dradius: 1.5 k: 85  -> point count: 11717  -  2021-02-24 01:19:19.048941  elapsed: 961019 microseconds
dradius: 1.5 k: 85  -> point count: 11683  -  2021-02-24 01:19:30.269359  elapsed: 220319 microseconds
dradius: 1.5 k: 85  -> point count: 11683  -  2021-02-24 01:19:41.510785  elapsed: 241320 microseconds
dradius: 1.5 k: 85  -> point count: 11662  -  2021-02-24 01:19:53.039417  elapsed: 528502 microseconds
dradius: 1.5 k: 85  -> point count: 11647  -  2021-02-24 01:20:04.527443  elapsed: 487881 microseconds
====== dradius: 1.5 k: 85  -> point count average: 11678.4  -  2021-02-24 01:20:04.527502  average elapsed: 487808.2 microseconds
dradius: 1.5 k: 86  -> point count: 11707  -  2021-02-24 01:20:19.592917  elapsed: 65332 microseconds
dradius: 1.5 k: 86  -> point count: 11732  -  2021-02-24 01:20:36.020860  elapsed: 427823 microseconds
dradius: 1.5 k: 86  -> point count: 11678  -  2021-02-24 01:20:52.628706  elapsed: 607729 microseconds
dradius: 1.5 k: 86  -> point count: 11701  -  2021-02-24 01:21:03.557082  elapsed: 928268 microseconds
dradius: 1.5 k: 86  -> point count: 11662  -  2021-02-24 01:21:13.825728  elapsed: 268549 microseconds
====== dradius: 1.5 k: 86  -> point count average: 11696.0  -  2021-02-24 01:21:13.825782  average elapsed: 459540.2 microseconds
dradius: 1.5 k: 87  -> point count: 11685  -  2021-02-24 01:21:24.430463  elapsed: 604604 microseconds
dradius: 1.5 k: 87  -> point count: 11698  -  2021-02-24 01:21:35.134795  elapsed: 704223 microseconds
dradius: 1.5 k: 87  -> point count: 11696  -  2021-02-24 01:21:45.961386  elapsed: 826481 microseconds
dradius: 1.5 k: 87  -> point count: 11694  -  2021-02-24 01:21:56.848150  elapsed: 886665 microseconds
dradius: 1.5 k: 87  -> point count: 11651  -  2021-02-24 01:22:07.734338  elapsed: 886088 microseconds
====== dradius: 1.5 k: 87  -> point count average: 11684.8  -  2021-02-24 01:22:07.734394  average elapsed: 781612.2 microseconds
dradius: 1.5 k: 88  -> point count: 11643  -  2021-02-24 01:22:18.581911  elapsed: 847431 microseconds
dradius: 1.5 k: 88  -> point count: 11726  -  2021-02-24 01:22:29.090058  elapsed: 508048 microseconds
dradius: 1.5 k: 88  -> point count: 11684  -  2021-02-24 01:22:39.857974  elapsed: 767779 microseconds
dradius: 1.5 k: 88  -> point count: 11641  -  2021-02-24 01:22:50.534440  elapsed: 676367 microseconds
dradius: 1.5 k: 88  -> point count: 11724  -  2021-02-24 01:23:01.027823  elapsed: 493289 microseconds
====== dradius: 1.5 k: 88  -> point count average: 11683.6  -  2021-02-24 01:23:01.027874  average elapsed: 658582.8 microseconds
dradius: 1.5 k: 89  -> point count: 11705  -  2021-02-24 01:23:11.541096  elapsed: 513151 microseconds
dradius: 1.5 k: 89  -> point count: 11693  -  2021-02-24 01:23:22.439128  elapsed: 897935 microseconds
dradius: 1.5 k: 89  -> point count: 11695  -  2021-02-24 01:23:33.209970  elapsed: 770733 microseconds
dradius: 1.5 k: 89  -> point count: 11656  -  2021-02-24 01:23:44.389778  elapsed: 179692 microseconds
dradius: 1.5 k: 89  -> point count: 11726  -  2021-02-24 01:23:55.078318  elapsed: 688443 microseconds
====== dradius: 1.5 k: 89  -> point count average: 11695.0  -  2021-02-24 01:23:55.078374  average elapsed: 609990.8 microseconds
dradius: 1.5 k: 90  -> point count: 11698  -  2021-02-24 01:24:05.730782  elapsed: 652332 microseconds
dradius: 1.5 k: 90  -> point count: 11691  -  2021-02-24 01:24:16.451968  elapsed: 721091 microseconds
dradius: 1.5 k: 90  -> point count: 11746  -  2021-02-24 01:24:26.918180  elapsed: 466115 microseconds
dradius: 1.5 k: 90  -> point count: 11677  -  2021-02-24 01:24:37.435726  elapsed: 517448 microseconds
dradius: 1.5 k: 90  -> point count: 11672  -  2021-02-24 01:24:48.473685  elapsed: 37868 microseconds
====== dradius: 1.5 k: 90  -> point count average: 11696.8  -  2021-02-24 01:24:48.473752  average elapsed: 478970.8 microseconds
dradius: 1.5 k: 91  -> point count: 11670  -  2021-02-24 01:24:59.461427  elapsed: 987598 microseconds
dradius: 1.5 k: 91  -> point count: 11709  -  2021-02-24 01:25:10.706681  elapsed: 245162 microseconds
dradius: 1.5 k: 91  -> point count: 11731  -  2021-02-24 01:25:21.768837  elapsed: 62038 microseconds
dradius: 1.5 k: 91  -> point count: 11673  -  2021-02-24 01:25:32.713471  elapsed: 944538 microseconds
dradius: 1.5 k: 91  -> point count: 11672  -  2021-02-24 01:25:43.720017  elapsed: 6452 microseconds
====== dradius: 1.5 k: 91  -> point count average: 11691.0  -  2021-02-24 01:25:43.720070  average elapsed: 449157.6 microseconds
dradius: 1.5 k: 92  -> point count: 11778  -  2021-02-24 01:25:54.600906  elapsed: 880762 microseconds
dradius: 1.5 k: 92  -> point count: 11734  -  2021-02-24 01:26:05.416804  elapsed: 815804 microseconds
dradius: 1.5 k: 92  -> point count: 11720  -  2021-02-24 01:26:16.260005  elapsed: 843108 microseconds
dradius: 1.5 k: 92  -> point count: 11693  -  2021-02-24 01:26:27.113758  elapsed: 853662 microseconds
dradius: 1.5 k: 92  -> point count: 11670  -  2021-02-24 01:26:37.832577  elapsed: 718714 microseconds
====== dradius: 1.5 k: 92  -> point count average: 11719.0  -  2021-02-24 01:26:37.832631  average elapsed: 822410.0 microseconds
dradius: 1.5 k: 93  -> point count: 11704  -  2021-02-24 01:26:51.777942  elapsed: 945239 microseconds
dradius: 1.5 k: 93  -> point count: 11731  -  2021-02-24 01:27:04.132796  elapsed: 354739 microseconds
dradius: 1.5 k: 93  -> point count: 11696  -  2021-02-24 01:27:15.800133  elapsed: 667177 microseconds
dradius: 1.5 k: 93  -> point count: 11691  -  2021-02-24 01:27:27.164505  elapsed: 364266 microseconds
dradius: 1.5 k: 93  -> point count: 11696  -  2021-02-24 01:27:38.253045  elapsed: 88443 microseconds
====== dradius: 1.5 k: 93  -> point count average: 11703.6  -  2021-02-24 01:27:38.253109  average elapsed: 483972.8 microseconds
dradius: 1.5 k: 94  -> point count: 11726  -  2021-02-24 01:27:49.656801  elapsed: 403615 microseconds
dradius: 1.5 k: 94  -> point count: 11728  -  2021-02-24 01:28:00.834015  elapsed: 177117 microseconds
dradius: 1.5 k: 94  -> point count: 11700  -  2021-02-24 01:28:12.095548  elapsed: 261437 microseconds
dradius: 1.5 k: 94  -> point count: 11742  -  2021-02-24 01:28:23.807759  elapsed: 712120 microseconds
dradius: 1.5 k: 94  -> point count: 11746  -  2021-02-24 01:28:37.028117  elapsed: 220251 microseconds
====== dradius: 1.5 k: 94  -> point count average: 11728.4  -  2021-02-24 01:28:37.028178  average elapsed: 354908.0 microseconds
dradius: 1.5 k: 95  -> point count: 11758  -  2021-02-24 01:28:51.523497  elapsed: 495237 microseconds
dradius: 1.5 k: 95  -> point count: 11732  -  2021-02-24 01:29:05.370944  elapsed: 847336 microseconds
dradius: 1.5 k: 95  -> point count: 11769  -  2021-02-24 01:29:18.078381  elapsed: 707321 microseconds
dradius: 1.5 k: 95  -> point count: 11728  -  2021-02-24 01:29:29.605772  elapsed: 527291 microseconds
dradius: 1.5 k: 95  -> point count: 11727  -  2021-02-24 01:29:40.941599  elapsed: 335733 microseconds
====== dradius: 1.5 k: 95  -> point count average: 11742.8  -  2021-02-24 01:29:40.941655  average elapsed: 582583.6 microseconds
dradius: 1.5 k: 96  -> point count: 11693  -  2021-02-24 01:29:54.216577  elapsed: 274844 microseconds
dradius: 1.5 k: 96  -> point count: 11729  -  2021-02-24 01:30:07.499725  elapsed: 283038 microseconds
dradius: 1.5 k: 96  -> point count: 11715  -  2021-02-24 01:30:19.894431  elapsed: 394606 microseconds
dradius: 1.5 k: 96  -> point count: 11761  -  2021-02-24 01:30:31.646172  elapsed: 751642 microseconds
dradius: 1.5 k: 96  -> point count: 11735  -  2021-02-24 01:30:44.511967  elapsed: 865697 microseconds
====== dradius: 1.5 k: 96  -> point count average: 11726.6  -  2021-02-24 01:30:44.512032  average elapsed: 513965.4 microseconds
dradius: 1.5 k: 97  -> point count: 11778  -  2021-02-24 01:30:58.046986  elapsed: 534868 microseconds
dradius: 1.5 k: 97  -> point count: 11739  -  2021-02-24 01:31:10.171831  elapsed: 124742 microseconds
dradius: 1.5 k: 97  -> point count: 11770  -  2021-02-24 01:31:22.894103  elapsed: 722176 microseconds
dradius: 1.5 k: 97  -> point count: 11764  -  2021-02-24 01:31:36.076095  elapsed: 181889 microseconds
dradius: 1.5 k: 97  -> point count: 11729  -  2021-02-24 01:31:49.252484  elapsed: 176290 microseconds
====== dradius: 1.5 k: 97  -> point count average: 11756.0  -  2021-02-24 01:31:49.252556  average elapsed: 347993.0 microseconds
dradius: 1.5 k: 98  -> point count: 11730  -  2021-02-24 01:32:02.225433  elapsed: 972769 microseconds
dradius: 1.5 k: 98  -> point count: 11748  -  2021-02-24 01:32:14.976849  elapsed: 751318 microseconds
dradius: 1.5 k: 98  -> point count: 11740  -  2021-02-24 01:32:28.324664  elapsed: 347710 microseconds
dradius: 1.5 k: 98  -> point count: 11727  -  2021-02-24 01:32:40.116211  elapsed: 791442 microseconds
dradius: 1.5 k: 98  -> point count: 11712  -  2021-02-24 01:32:52.471305  elapsed: 354948 microseconds
====== dradius: 1.5 k: 98  -> point count average: 11731.4  -  2021-02-24 01:32:52.471360  average elapsed: 643637.4 microseconds


'''