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
	# print('numerator', numerator, 'denominator', denominator)

	averageAngle = np.arctan2( numerator, denominator ) * 180 / math.pi
	averageAngle = averageAngle % 360
	return averageAngle

def calcAverageAngleWeighted(angle1, angle2, weight1, weight2):
	'''
	Given two different angles, calculate the naive average.
	'''
	if weight1 == 0 and weight2 == 0:
		weight1 += 0.00001 # Make sure this value is not zero.
		weight2 += 0.00001 # Make sure this value is not zero.

	totalWeight = weight1 + weight2
	weight1 = weight1 / totalWeight
	weight2 = weight2 / totalWeight


	angle1, angle2 = findMinAngleDiff(angle1, angle2)

	toRad = math.pi/180
	numerator = math.sin(angle1*toRad )*weight1 + math.sin(angle2*toRad)*weight2
	denominator = math.cos(angle1*toRad)*weight1 + math.cos(angle2*toRad)*weight2
	# print('numerator', numerator, 'denominator', denominator)

	averageAngle = np.arctan2( numerator, denominator ) * 180 / math.pi
	averageAngle = averageAngle % 360
	return averageAngle


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


def repelAngles(stableAngle, movingAngle):
	'''
	Need to figure out a way to push angles away from each other.
	:param stableAngle:
	:param movingAngle:
	:return:
	'''
	pass

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
	angles.append([350, 359])
	angles.append([350, 1])
	angles.append([350, 181])

	for anglePair in angles:
		angle1, angle2 = anglePair
		print("Angle", angle1, "and", angle2)

		if True:
			angle1, angle2 = findMinAngleDiff(angle1, angle2)
			print("Closer Angles", angle1, "and", angle2, "\n")

		if False:
			diff = findAngleDiff(angle1, angle2)
			print("Angle", angle1, "and", angle2, "diff:", diff, "\n")

		if True:
			averageAngle = calcAverageAngle(angle1, angle2)
			print("Angle", angle1, "and", angle2, "Average:", averageAngle, "\n")

		if True:
			averageAngle = calcAverageAngleWeighted(angle1, angle2, 0.01, 0.0)
			print("Angle", angle1, "and", angle2, "Average Weighted:", averageAngle, "\n")






