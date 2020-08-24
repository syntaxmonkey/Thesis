import numpy as np
from scipy.spatial import distance
import EdgePoint
import copy

edgePoint1 = EdgePoint.EdgePoint((1,2), [(1,2), (2,3)], 1, 0)

print("EdgePoint1:", edgePoint1.xy)

edgePoint2 = copy.deepcopy( edgePoint1 )

edgePoint1.xy = (3,3)
print("EdgePoint1:", edgePoint1.xy)
print("EdgePoint2:", edgePoint2.xy)