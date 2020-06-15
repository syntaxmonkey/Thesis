# Modified the code from here: https://www.kaggle.com/mburger/freeman-chain-code-second-attempt

# This code is based on http://www.cs.unca.edu/~reiser/imaging/chaincode.html

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import cv2
# from math import sqrt
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# from itertools import chain

import math
from FilledLetter import genLetter

import Bridson_CreateMask
import Bridson_Common
import math



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# Bridson_Common.logDebug(__name__, check_output(["ls", "../input"]).decode("utf8"))
# train = pd.read_csv("../input/train.csv")
#
# train[:3]
# # Any results you write to the current directory are saved as output.
# labels = train['label']
# train_images = train.drop('label', axis=1)
# train_images.head()
# image = np.reshape(train_images[9:10].as_matrix(), (-1, 28)).astype(np.uint8)
# plt.imshow(image, cmap='Greys')
#

def sumPreviousX(array, currentIndex, x ):
	return sum(array[currentIndex-x: currentIndex])


def writeChainCodeFile(path, filename, chainDirection):
	f = open(path + filename, "w+")

	for direction in chainDirection:
		f.write("%d\r\n" % (direction))
	f.close()

def generateChainCode(img, rotate=False, angle=0):
	closeLoop = True # Ensure the chain code forms a loop.
	angleChangeThreshold = 270
	correctionValue = 1
	currentDirection = 0
	cumulativeDirection = 0 # Sum all the direction changes.

	## Discover the first point

	for i, row in enumerate(img):
		for j, value in enumerate(row):
	# for j, row in enumerate(img):  # Swapped row and column compared to the original.
	# 	for i, value in enumerate(row):
			if value == 255:
				start_point = (i, j)
				Bridson_Common.logDebug(__name__, start_point, value)
				break
		else:
			continue
		break


	directions = [ 0,  1,  2,
				   7,      3,
				   6,  5,  4]
	dir2idx = dict(zip(directions, range(len(directions))))

	change_j =   [-1,  0,  1, # x or columns
				  -1,      1,
				  -1,  0,  1]

	change_i =   [-1, -1, -1, # y or rows
				   0,      0,
				   1,  1,  1]

	directionAngleMap = []
	for i in range(8):
		directionAngleMap.append((315+i*45)%360)

	border = []
	chain = []
	chainDirection = [] # Will contain the degree change.

	originalStart = start_point # Original start point.
	Bridson_Common.logDebug(__name__, 'Original Start:', originalStart)
	curr_point = start_point

	''' Block A1 - pick the starting direction.  Should aim for direction 3.'''
	direction = 2
	b_direction = (direction + 5) % 8
	dirs_1 = range(b_direction, 8)
	dirs_2 = range(0, b_direction)
	dirs = []
	dirs.extend(dirs_1)
	dirs.extend(dirs_2)
	for direction in dirs:
		''' Block A1 end'''
	# for direction in directions:
		idx = dir2idx[direction]
		new_point = (start_point[0]+change_i[idx], start_point[1]+change_j[idx])
		if img[new_point] != 0: # if is ROI
			border.append(new_point)
			chain.append(direction)
			curr_point = new_point
			break

	count = 0
	while curr_point != originalStart:
		#figure direction to start search
		b_direction = (direction + 5) % 8
		dirs_1 = range(b_direction, 8)
		dirs_2 = range(0, b_direction)
		dirs = []
		dirs.extend(dirs_1)
		dirs.extend(dirs_2)
		for direction in dirs:
			idx = dir2idx[direction]
			new_point = (curr_point[0]+change_i[idx], curr_point[1]+change_j[idx])
			if img[new_point] != 0: # if is ROI
				border.append(new_point)
				chain.append(direction)
				directionChange = (chain[-1] - chain[-2]) # HSC
				# If directionChange is not zero
				if directionChange != 0:
					if directionChange > 4:
						# In this case, we are going CCW.
						directionChange = directionChange - 8
					elif directionChange < -4:
						directionChange = directionChange + 8
					else:
						# In this case, we are going CW.
						directionChange = directionChange

				directionChange = directionChange * 45

				cumulativeDirection += directionChange
				if abs(cumulativeDirection) > 360:
					Bridson_Common.logDebug(__name__, 'CumulativeDirection: ', cumulativeDirection)

				currentDirection += directionChange
				chainDirection.append(directionChange)

				if abs(sumPreviousX(chainDirection, len(chainDirection) , 4)) > angleChangeThreshold:    # Check if the previous X direction changes are above a threshold.
					Bridson_Common.logDebug(__name__, '*** Direction Change warning: ', chainDirection[-2:])
					# chainDirection[-1] = (chainDirection[-1] / abs(chainDirection[-1]) * -correctionValue) + chainDirection[-1]
					# chainDirection[-2] = (chainDirection[-2] / abs(chainDirection[-2]) * -correctionValue) * \
					#                      chainDirection[-2]

					chainDirection[-1] = chainDirection[-1] * correctionValue
					chainDirection[-2] = chainDirection[-2] * correctionValue


				curr_point = new_point
				break
		#if count == 10000: break
		count += 1


	# Need to add last transition to close the loop.
	# Find the direction to the original Start point.
	# HSC
	if False:
		for direction in directions:
			idx = dir2idx[direction]
			new_point = (curr_point[0]+change_i[idx], curr_point[1]+change_j[idx])
			if new_point == border[0]: # if is ROI
				border.append(new_point)
				chain.append(direction)
				directionChange = (chain[-1] - chain[-2])  # HSC
				# If directionChange is not zero
				if directionChange != 0:
					if directionChange > 4:
						# In this case, we are going CCW.
						directionChange = directionChange - 8
					elif directionChange < -4:
						directionChange = directionChange + 8
					else:
						# In this case, we are going CW.
						directionChange = directionChange

				directionChange = directionChange * 45
				cumulativeDirection += directionChange
				if abs(cumulativeDirection) > 360:
					Bridson_Common.logDebug(__name__, 'CumulativeDirection: ', cumulativeDirection)

				chainDirection.append(directionChange)

				if abs(sumPreviousX(chainDirection, len(chainDirection), 4)) > angleChangeThreshold:    # Check if the previous X direction changes are above a threshold.
					Bridson_Common.logDebug(__name__, '*** Direction Change warning: ', chainDirection[-2:])
					# chainDirection[-1] = (chainDirection[-1] / abs(chainDirection[-1]) * -correctionValue) + chainDirection[-1]
					# chainDirection[-2] = (chainDirection[-2] / abs(chainDirection[-2]) * -correctionValue) + \
					#                      chainDirection[-2]
					chainDirection[-1] = chainDirection[-1] * correctionValue
					chainDirection[-2] = chainDirection[-2] * correctionValue


				break
		count += 1

	Bridson_Common.logDebug(__name__, 'Final Cumulative Direction: ', cumulativeDirection)


	if rotate == True:
		# Perform a rotation of the chain code.
		# Bridson_Common.logDebug(__name__, "Length of Chain code", len(chain))
		# Bridson_Common.logDebug(__name__, chain)
		# Bridson_Common.logDebug(__name__, "length of chainDirection", len(chainDirection))
		# Bridson_Common.logDebug(__name__, chainDirection)
		# Bridson_Common.logDebug(__name__, "Length of border", len(border))

		shiftRatio = (angle%360)/360

		dirLength = len(chainDirection)
		quarterLength = math.floor(dirLength * shiftRatio)

		newChain = chain[quarterLength:-1]
		newChain.extend(chain[0:quarterLength])
		chain=newChain

		newChain = chainDirection[quarterLength:-1]
		newChain.extend(chainDirection[0:quarterLength])
		chainDirection = newChain

		newChain = border[quarterLength:-1]
		newChain.extend(border[0:quarterLength])
		border = newChain

	# Bridson_Common.logDebug(__name__, "chainDirection:", chainDirection)
	# chainDirection = list(reverseDirection(chainDirection))
	# Bridson_Common.logDebug(__name__, "chainDirection after reversal:", chainDirection)
	Bridson_Common.logDebug(__name__, "** Border: ", border)
	return count, chain, chainDirection, border



def reverseDirection(chainDirection):
	# This method will reverse the direction of the chaincode.
	# newDirection = []
	newDirection = np.array(chainDirection)
	newDirection = newDirection * -1

	return newDirection

def writeChainCodeFile(path, filename, chainDirection):
	f = open(path + filename, "w+")

	for direction in chainDirection:
		f.write("%d\r\n" % (direction))
	f.close()



def generateBorder(borderChainCoordinates, radius):
	# Given the border coordinates for a chain code...
	# Determine the perimeter length.
	# Determine the number of nodes that we need to insert.
	# Walk around the perimeter and insert points as necessary.
	borderSegments = len(borderChainCoordinates)
	perimeterLength = 0
	for i in range( borderSegments ):
		startIndex = i
		endIndex = (i+1)%borderSegments
		perimeterLength += Bridson_Common.euclidean_distance(borderChainCoordinates[startIndex], borderChainCoordinates[endIndex] )

	nodeCount = math.floor( perimeterLength / radius )

	perimeterNodes = []
	distanceTravelled = 0
	for i in range( borderSegments ):
		startIndex = i
		endIndex = (i+1)%borderSegments
		distanceTravelled += Bridson_Common.euclidean_distance(borderChainCoordinates[startIndex], borderChainCoordinates[endIndex] )
		if distanceTravelled > radius:
			# Keep this node.
			currentNode = borderChainCoordinates[endIndex]
			# Bridson_Common.logDebug(__name__, "CurrentNode: ", currentNode)
			currentNode = (currentNode[0] + 0.5 , currentNode[1] + 0.5)
			perimeterNodes.append(currentNode)
			# Keep track of the difference between radius and actual distance travelled.
			distanceTravelled = distanceTravelled - radius

	return perimeterNodes


if __name__ == '__main__':
	xrange, yrange = 80, 80
	radius = 4
	# The chain code requires the original mask.  DO NOT use the inverted Mask.
	mask =  Bridson_CreateMask.genLetter(xrange, yrange, character='Y')

	# img = genLetter(boxsize=int(100/4), character='C', blur=1)
	# letterDimension = 20
	# character='C'
	# img = genLetter(boxsize=letterDimension, character=character, blur=1)
	count, chain, chainDirection, border = generateChainCode(mask, rotate=False)
	Bridson_Common.logDebug(__name__, 'Count:', count)
	Bridson_Common.logDebug(__name__, 'Chain:', len(chain), chain)
	Bridson_Common.logDebug(__name__, 'ChainDirection:', len(chainDirection), chainDirection)
	Bridson_Common.logDebug(__name__, 'Border:', len(border), border)

	writeChainCodeFile('./', 'testchaincode.txt', chainDirection)

	border = generateBorder( border, radius )

	plt.imshow(mask, cmap='Greys')
	plt.plot(border[-1][1], border[-1][0], 'og') # Plot the starting point as a red dot.
	plt.plot(border[0][1], border[0][0], 'or')  # Plot the ending point as a green dot.

	plt.plot([i[1] for i in border], [i[0] for i in border], 'og') # Plot the "boundaries" points as green dots.
	plt.show()