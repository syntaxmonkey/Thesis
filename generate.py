# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


baseFolder="/Users/hengsun/Documents/Thesis/boundary-first-flattening/build/"

def runit():
    exec(open('generate.py').read())

import pymesh
import numpy as np

def ConvertToMesh(vertices, faces, filename):
    print("Writing to ", filename)
    output=open(filename, 'w')

    for vertex in vertices:
        output.write("v %f %f %f"%(vertex[0], vertex[1], vertex[2]))
        output.write("\n")
        
    for face in faces:
        # IMPORTANT * Vertex indexing starts at 1.        
        output.write("f %d %d %d"% (face[0]+1, face[1]+1, face[2]+1))
        output.write("\n")
    output.close()
    
    

print( 'Generating mesh' )

# Create sphere about origin.
mesh=pymesh.generate_icosphere(2, (0,0,0), 4)

# Create cube
box=pymesh.generate_box_mesh((-2, -2, -4), (2,2,0))

# Create sylinder 
tube=pymesh.generate_tube((0,-1,0), (0,1,0), 1.1, 1.1, 1.0, 1.0,  with_quad=True)

# Need to determine which faces reference at least one vertex that has a negative 
#pieces = pymesh.slice_mesh(mesh, (1,1,0), 2)

# One option is to create a cube and a sphere, then take the diffference.
# https://pymesh.readthedocs.io/en/latest/mesh_boolean.html
intersection=pymesh.boolean(mesh,box,'difference', 'carve')

print( 'Saving mesh to '+baseFolder+'/icosphere.obj' )
pymesh.save_mesh(baseFolder+"icosphere.obj", mesh, ascii=True)
print( 'Saving mesh to '+baseFolder+'box.obj' )
pymesh.save_mesh(baseFolder+"box.obj", box, ascii=True)
print( 'Saving mesh to '+baseFolder+'intersection.obj' )
pymesh.save_mesh(baseFolder+"intersection.obj", intersection, ascii=True)
print( 'Saving mesh to '+baseFolder+'tube.obj' )
pymesh.save_mesh(baseFolder+"tube.obj", tube, ascii=True)


# The above creates a solid object.

# Try to remove faces that have vertices with negative z values.
#print(mesh.faces)


print('Generating new mesh')
newFaces = []

for face in mesh.faces:
#    print(face)
    negative=False
    for vertex in face:
        if np.amin(mesh.vertices[vertex][2]) < 0:
#            print("Negative vertex:", mesh.vertices[vertex])
            negative=True
            break
    
    if negative==False:
        newFaces.append(list(face))
    
#ConvertToMesh(mesh.vertices, newFaces, "/Users/hengsun/Documents/Thesis/newMesh.obj")

newFaces = np.array(newFaces)
# Was able to generate new mesh of faces on the positive side of Z axis.
# However, the lost faces leave gaps.  As a result, we no longer have something that is manifold to BFF.
# How to create new faces ... or remove the backing from the intersection.
newMesh = pymesh.form_mesh(mesh.vertices, newFaces)


print('Saving '+baseFolder+'newMesh.obj')
pymesh.save_mesh(baseFolder+"newMesh.obj", newMesh, ascii=True)
#newMesh.write_to_file("/Users/hengsun/Documents/Thesis/newMesh.obj")












# Stage 1 filtering.
# We can obtain the face area.
intersection.add_attribute("face_area")
intersection.add_attribute("face_voronoi_area")

intersectFaces = []
areas = intersection.get_attribute("face_area")
#keeper = np.average(areas)
keeper = np.amax(areas)/2.0
index = 0

for faceIndex in range(len(intersection.faces)):
    if areas[faceIndex] < keeper*2:
        face = intersection.faces[faceIndex]
        intersectFaces.append(list(face))        

intersectFaces = np.array(intersectFaces)
newIntersect = pymesh.form_mesh(intersection.vertices, intersectFaces)
print('Saving '+baseFolder+'newIntersect.obj')
pymesh.save_mesh(baseFolder+"newIntersect.obj", newIntersect, ascii=True)


# Stage 2 filtering.
# Successful attempt.
# This version removes triangles that co-planar to x,y plane.  This version is accepted by BFF.

intersectFaces2 = []

for face in newIntersect.faces:
    zCount=0.0

    # Counting the z axis.
    for vertex in face:
        zCount+=abs(zCount-abs(newIntersect.vertices[vertex][2]))
  
    if zCount>0.0:
        intersectFaces2.append(list(face))
    
intersectFaces2 = np.array(intersectFaces2)
newIntersect2 = pymesh.form_mesh(intersection.vertices, intersectFaces2)
print('Saving '+baseFolder+'newIntersect2.obj')
pymesh.save_mesh(baseFolder+"newIntersect2.obj", newIntersect2, ascii=True)


trimmed=pymesh.collapse_short_edges(newIntersect2, rel_threshold=0.5, preserve_feature=True)
print('Saving '+baseFolder+'trimmed.obj')
pymesh.save_mesh(baseFolder+"trimmed.obj", trimmed[0], ascii=True)






