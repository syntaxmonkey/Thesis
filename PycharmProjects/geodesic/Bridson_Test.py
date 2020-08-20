import numpy as np

# https://stackoverflow.com/questions/38073433/determine-adjacent-regions-in-numpy-array

x = np.array([[1, 1, 1], [1, 1, 2], [2, 2, 2], [3, 3, 3]], np.int32)
region = 1   # number of region whose neighbors we want

y = x == region  # convert to Boolean

rolled = np.roll(y, 1, axis=0)          # shift down
rolled[0, :] = False             
z = np.logical_or(y, rolled)

rolled = np.roll(y, -1, axis=0)         # shift up 
rolled[-1, :] = False
z = np.logical_or(z, rolled)

rolled = np.roll(y, 1, axis=1)          # shift right
rolled[:, 0] = False
z = np.logical_or(z, rolled)

rolled = np.roll(y, -1, axis=1)         # shift left
rolled[:, -1] = False
z = np.logical_or(z, rolled)

neighbors = set(np.unique(np.extract(z, x))) - set([region])
print(neighbors)


# Find the distance from edge.
# https://stackoverflow.com/questions/40492159/find-distance-from-the-edge-of-a-numpy-array

from scipy.ndimage.morphology import binary_erosion
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import distance_transform_cdt

def dist_from_edge(img):
    I = binary_erosion(img) # Interior mask
    C = img - I             # Contour mask
    out = C.astype(int)     # Setup o/p and assign cityblock distances
    out[I] = cdist(np.argwhere(C), np.argwhere(I), 'cityblock').min(0) + 1
    return out

def distance_from_edge_a(x):
    x = np.pad(x, 1, mode='constant')
    dist = distance_transform_cdt(x, metric='chessboard')
    return dist[1:-1, 1:-1]

a = np.array([[0, 0, 0, 0, 1, 0, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 1],
       [0, 0, 1, 1, 1, 1, 1],
       [0, 0, 0, 1, 0, 0, 0]])




# a = np.array([[0, 0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 1, 1, 1, 0],
#               [0, 0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 1, 0, 0, 0]])


b = dist_from_edge(a)
print(b)

print('\n')

c = distance_from_edge_a(a)
print(c)

mc = np.ma.masked_array( c, np.invert( np.logical_and(c < 2 , c > 0 ) )  )
print(mc)