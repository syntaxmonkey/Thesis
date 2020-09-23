# This class handles various calculations for angles.


# This site claims to have an algorithm that can sum of arbitrary angles.
# https://www.themathdoctors.org/averaging-angles/
import math
import numpy as np



def calcAverageAngle(angle1, angle2):
	'''
	Given two different angles, calculate the naive average.
	'''

	angle1, angle2 = findMinAngleDiff(angle1, angle2)

	toRad = math.pi/180
	numerator = math.sin(angle1*toRad ) + math.sin(angle2*toRad)
	denominator = math.cos(angle1*toRad) + math.cos(angle2*toRad)
	print('numerator', numerator, 'denominator', denominator)
	# value = numerator / denominator
	# print("Value:", value)
	# averageAngle = math.atan( value )
	# averageAngle = np.arctan( value ) * 180 / math.pi # Need to convert to
	averageAngle = np.arctan2( numerator, denominator ) * 180 / math.pi
	return averageAngle, angle1, angle2


def findMinAngleDiff(angle1, angle2):
	if angle1 > angle2:
		angle1, angle2 = angle2, angle1

	diff1 = findAngleDiff(angle1, angle2)
	reversedAngle2 = (angle2 + 180) % 360
	diff2 = findAngleDiff( angle1, reversedAngle2)

	angle2 = angle2 if diff1 <= diff2  else reversedAngle2

	if angle1 > angle2:
		angle1, angle2 = angle2, angle1

	return angle1, angle2


def findAngleDiff(targetA, sourceA):
	a = targetA - sourceA
	a = (a + 180) % 360 - 180
	return abs(a)

if __name__ == '__main__':


	angles = []
	angles.append([5,15])
	angles.append([1, 180])
	angles.append([1, 359])
	angles.append([0, 180])
	angles.append([0, 90])
	angles.append([90, 180])
	angles.append([45, 135])
	angles.append([0, 0])
	angles.append([350, 10])
	angles.append([270, 135])

	for anglePair in angles:
		angle1, angle2 = anglePair
		print("Angle", angle1, "and", angle2)

		if False:
			angle1, angle2 = findMinAngleDiff(angle1, angle2)
			print("Closer Angles", angle1, "and", angle2, "\n")

		if False:
			diff = findAngleDiff(angle1, angle2)
			print("Angle", angle1, "and", angle2, "diff:", diff, "\n")

		if True:
			averageAngle, angle1, angle2 = calcAverageAngle(angle1, angle2)
			print("Angle", angle1, "and", angle2, "Average:", averageAngle, "\n")






