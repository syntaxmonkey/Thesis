import math
import Bridson_Common

def euclidean_distance(a, b):
	dx = a[0] - b[0]
	dy = a[1] - b[1]
	return math.sqrt(dx * dx + dy * dy)

def findArea(v1, v2, v3):
	# https://www.javatpoint.com/python-area-of-triangle
	a = euclidean_distance(v1, v2)
	b = euclidean_distance(v2, v3)
	c = euclidean_distance(v3, v1)
	s = (a+b+c)/2

	area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
	return area


def findAverageArea(triangles, samples):
	count = 0
	area = 0
	# for facet in triangleValues.simplices.copy():
	for facet in triangles:
		area += Bridson_Common.findArea(samples[facet[0]], samples[facet[1]], samples[facet[2]])
		count+=1
	averageArea = area / count
	return averageArea