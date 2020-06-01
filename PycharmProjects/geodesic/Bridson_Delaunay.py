from scipy.spatial import Delaunay # For generating Delaunay - https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.html
import matplotlib.pyplot as plt
import Bridson_Common
import numpy as np
import matplotlib.tri as mtri
import math
import pylab

removalRatio = 0.05

def generateDelaunay(points, radius, mask, xrange):
    tri = Delaunay(points)  # Generate the triangles from the vertices.

    if Bridson_Common.debug:
        plt.figure()
        plt.subplot(1, 1, 1, aspect=1)
        plt.title('Display Delaunay')
        plt.triplot(points[:,1], xrange-points[:,0], tri.simplices.copy())
        # plt.triplot(points[:, 0], points[:, 1], tri.triangles)
        plt.plot(points[:, 1], xrange-points[:, 0], 'o')
        thismanager = pylab.get_current_fig_manager()
        thismanager.window.wm_geometry("+640+0")


    newMask = Bridson_Common.blurArray(mask, 3)
    triangulation = removeLongTriangles(points, tri, radius, newMask)

    # Display the newly generated Mask
    if Bridson_Common.debug:
        plt.figure()
        plt.subplot(1, 1, 1, aspect=1)
        plt.title('newMask')
        plt.imshow(newMask)
        thismanager = pylab.get_current_fig_manager()
        thismanager.window.wm_geometry("+40+560")

    return triangulation



def removeLongTriangles(points, tri, radius, mask):
    # Find Average Area.
    averageArea = Bridson_Common.findAverageArea(tri.simplices.copy(), points)
    print("Bridson_Delaunay::removeLongTriangles Average Area:", averageArea)

    triangles = tri.simplices.copy()
    newTriangles = []

    # print(">>>>>>>>>>>>><<<<<<<<<<<<<<<<< Triangles: ", triangles)
    for triangle in triangles:
        # print("Single Triangle", triangle)
        Keep = True
        # Iterate through all 3 edges.  Ensure their euclidean distance is less than or equal to radius.
        # print("Triangle:", triangle[0])
        for i in range(3):
            currentIndex = i
            nextIndex = (currentIndex + 1) % 3
            # print("Indeces:", currentIndex, nextIndex)
            distance = Bridson_Common.euclidean_distance(points[triangle[currentIndex]], points[triangle[nextIndex]])
            # print("distance:", radius, distance)
            # if distance > radius:
            ''' The fudge factor of 1.1 is to account for rounding errors.  The "isExteriorTriangle" needs Mask blur of 3. '''
            # if distance > radius*math.sqrt(2)*1.1 or isExteriorTriangle(points[triangle[currentIndex]], points[triangle[nextIndex]], mask):
            if distance > radius * 3 or isExteriorTriangle(points[triangle[currentIndex]], points[triangle[nextIndex]], mask):
            # if isExteriorTriangle(points[triangle[currentIndex]], points[triangle[nextIndex]], mask):
                Keep = False
                # print("Bridson_Delaunay::removeLongTriangles Distance is too long OR exterior triangle.")
                break
        triangleHeights = Bridson_Common.findTriangleHeight(points[triangle[0]], points[triangle[1]], points[triangle[2]])
        # print("Triangle Heights: ", triangleHeights)
        if np.min(triangleHeights) < removalRatio*radius:
            # print("***** Triangle " + str(triangle) + " has a minimum of " + str(np.min(triangleHeights)))
            Keep = False

        area = Bridson_Common.findArea(points[triangle[0]], points[triangle[1]], points[triangle[2]])
        if area < averageArea / 10.0 or area > 10.0*averageArea:
            Keep = False


        if Keep:
            newTriangles.append(triangle)
        # else:
        #     print("Bridson_Delaunay::removeLongTriangles Removing triangle Area: ", area)

    # print("New Triangle Shape:", np.shape(newTriangles))
    newTriangles = np.array(newTriangles)

    newTri = mtri.Triangulation(points[:,0], points[:,1], newTriangles)

    return newTri


def isExteriorTriangle(p1, p2, mask):
    # return False
    # Find the mid point of the lines of the triangles.  See if the midpoint of the line intersects with the mask.
    midx, midy = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
    # If the value is greater than 0, then the line intersects with the mask and should be removed.
    # if mask[midx, midy] == 255:
    if mask[midx, midy] > 10:
        return True
    return False


def displayDelaunayMesh(points, radius, mask, xrange):
    triangulation = generateDelaunay(points, radius, mask, xrange)

    # triangles = tri.simplices
    # for triangle in triangles:
    # 	print("Triangle:", triangle)

    if Bridson_Common.debug:
        print("tri", triangulation)
        plt.figure()
        plt.subplot(1, 1, 1, aspect=1)
        plt.title('Display Triangulation')
        # plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
        # Plot the lines representing the mesh.
        plt.triplot(points[:, 1], xrange-points[:, 0], triangulation.triangles)
        # Plot the points on the border.
        plt.plot(points[:, 1], xrange-points[:, 0], 'o')

        thismanager = pylab.get_current_fig_manager()
        thismanager.window.wm_geometry("+640+560")

    return triangulation
        # plt.show()

