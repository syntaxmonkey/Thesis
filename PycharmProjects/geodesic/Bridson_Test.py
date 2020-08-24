import numpy as np
from scipy.spatial import distance

s1 = np.array([(0,0), (0,1), (1,0), (2,0)])
s2 = np.array([(1,2), (2,1)])

distances = distance.cdist(s1,s2)
print(distance.cdist(s1,s2))  # This gives the distance from each point in s1 to s2.
print()
shortestDistances = distance.cdist(s1,s2).min(axis=1)
print(distance.cdist(s1,s2).min(axis=1)) # This gives the shortest distance from s1 to s2.  However, it doesn't provide the index.
print()
print(distance.cdist(s1,s2).min())



# array([3.60555128, 3.16227766, 2.82842712, 2.23606798])


def findClosestIndex(s1, s2):
	distances = distance.cdist(s1, s2)
	# shortestDistances = distance.cdist(s1, s2).min(axis=1)
	location = np.where(distances == distances.min())
	print('Location:', location)
	return location



location = findClosestIndex(s1, s2)
print(location)

print(s1[location[0][0]], "to", s2[location[1][0]])