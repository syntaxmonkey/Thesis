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

from FilledLetter import genLetter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
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



def generateChainCode(img):
	currentDirection = 0
	## Discover the first point
	for i, row in enumerate(img):
		for j, value in enumerate(row):
			if value == 255:
				start_point = (i, j)
				print(start_point, value)
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

	curr_point = start_point
	for direction in directions:
		idx = dir2idx[direction]
		new_point = (start_point[0]+change_i[idx], start_point[1]+change_j[idx])
		if img[new_point] != 0: # if is ROI
			border.append(new_point)
			chain.append(direction)
			curr_point = new_point
			break

	count = 0
	while curr_point != start_point:
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

				# directionChange = directionChange * 45
				directionChange = directionChange * 45
				# if directionChange > 90:
				# 	directionChange = 90
				# elif directionChange < -90:
				# 	directionChange = -90
				currentDirection += directionChange
				# print(chain[-2], '-->', chain[-1], 'angle:', directionChange, 'Current Direction: ', currentDirection)
				chainDirection.append(directionChange)
				curr_point = new_point
				break
		if count == 10000: break
		count += 1

	return count, chain, chainDirection, border


def writeChainCodeFile(path, filename, chainDirection):
	f = open(path + filename, "w+")

	for direction in chainDirection:
		f.write("%d\r\n" % (direction))
	f.close()


if __name__ == '__main__':
	img = genLetter(boxsize=int(100/4), character='H', blur=1)
	count, chain, chainDirection, border = generateChainCode(img)
	print('Count:', count)
	print('Chain:', len(chain), chain)
	print('ChainDirection:', len(chainDirection), chainDirection)
	print('Border:', len(border), border)

	writeChainCodeFile('./', 'testchaincode.txt', chainDirection)

	plt.imshow(img, cmap='Greys')
	plt.plot([i[1] for i in border], [i[0] for i in border])
	plt.show()