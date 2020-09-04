# Re-implement this without opencv.
# https://www.ccoderun.ca/programming/doxygen/opencv/tutorial_anisotropic_image_segmentation_by_a_gst.html

import cv2 as cv
import numpy as np
import argparse
import pandas as pd
from numpy.linalg import eig

desired_width = 820
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 100)
np.set_printoptions(linewidth=desired_width)
np.set_printoptions(edgeitems=100)
np.core.arrayprint._line_width = 800


W = 52  # window size is WxW
C_Thr = 0.3  # threshold for coherency
LowThr = 35  # threshold1 for orientation, it ranges from 0 to 180
HighThr = 57  # threshold2 for orientation, it ranges from 0 to 180



class ST:
	def __init__(self, inputIMG):
		self.inputIMG = inputIMG


	def calcGST(self, w):
		img = self.inputIMG.astype(np.float32)
		# GST components calculation (start)
		# J =  (J11 J12; J12 J22) - GST
		imgDiffX = cv.Sobel(img, cv.CV_32F, 1, 0, 3)
		imgDiffY = cv.Sobel(img, cv.CV_32F, 0, 1, 3)
		imgDiffXY = cv.multiply(imgDiffX, imgDiffY)

		imgDiffXX = cv.multiply(imgDiffX, imgDiffX)
		imgDiffYY = cv.multiply(imgDiffY, imgDiffY)
		self.J11 = cv.boxFilter(imgDiffXX, cv.CV_32F, (w, w))
		self.J22 = cv.boxFilter(imgDiffYY, cv.CV_32F, (w, w))
		self.J12 = cv.boxFilter(imgDiffXY, cv.CV_32F, (w, w))
		print("Shape of J11:", np.shape( self.J11 ))
		print("Shape of J12:", np.shape(self.J12))
		print("Shape of J22:", np.shape(self.J22))
		# GST components calculations (stop)
		# eigenvalue calculation (start)
		# lambda1 = J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2)
		# lambda2 = J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2)
		tmp1 = self.J11 + self.J22
		tmp2 = self.J11 - self.J22
		tmp2 = cv.multiply(tmp2, tmp2)
		tmp3 = cv.multiply(self.J12, self.J12)
		tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
		self.lambda1 = tmp1 + tmp4  # biggest eigenvalue
		self.lambda2 = tmp1 - tmp4  # smallest eigenvalue
		# eigenvalue calculation (stop)
		# Coherency calculation (start)
		# Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
		# Coherency is anisotropy degree (consistency of local orientation)
		imgCoherencyOut = cv.divide(self.lambda1 - self.lambda2, self.lambda1 + self.lambda2)
		# Coherency calculation (stop)
		# orientation angle calculation (start)
		# tan(2*Alpha) = 2*J12/(J22 - J11)
		# Alpha = 0.5 atan2(2*J12/(J22 - J11))
		imgOrientationOut = cv.phase(self.J22 - self.J11, 2.0 * self.J12, angleInDegrees=True)
		imgOrientationOut = 0.5 * imgOrientationOut
		# orientation angle calculation (stop)
		return imgCoherencyOut, imgOrientationOut

	def calculateEigenVector(self, x, y):
		u = x
		v = y
		# self.h = [u*( u*self.J11 + v*self.J12 ), v*( u*self.J12 + v*self.J22 )]
		self.st = np.array([[np.sum( self.J11 ), np.sum( self.J12 )], [np.sum( self.J12 ), np.sum( self.J22 ) ]])
		self.h = [u * (u * self.st[0,0] + v * self.st[0,1]), v * (u * self.st[1,0] + v * self.st[1,1])]
		print("Manual ST:", self.h / abs(np.amax(self.h)) )
		# print("ST shape:", np.shape(self.st))
		# print("X component:",self.st[0][x,y])
		# print("Y component:", self.st[1][x,y])

		# self.st = np.array([[np.sum( self.J11 ), np.sum( self.J12 )], [np.sum( self.J12 ), np.sum( self.J22 ) ]])
		self.st = np.array( [[self.J11[x,y], self.J12[x,y]], [self.J12[x,y], self.J22[x,y]]] )
		print("ST: ", self.st)
		print("\n\nH:", self.st)
		self.values, self.vectors = eig( self.st )
		print("Shape of ST:", np.shape(self.st))
		print("Eigen Values:", self.values )
		print("Eigen Vectors:", self.vectors)
		listValues = self.values.tolist()
		# The direction is the eigenvector associated with the larger eigenvalue.
		direction = self.vectors[ listValues.index( np.amax(listValues) ) ]
		print("Direction: ", self.vectors[ listValues.index( np.amax(listValues) ) ] )



test = [-1000000, -10]
print("TEst Max:", np.amax(test))

#
# parser = argparse.ArgumentParser(description='Code for Anisotropic image segmentation tutorial.')
# parser.add_argument('-i', '--input', help='Path to input image.', required=True)
# args = parser.parse_args()
# input = "alex-furgiuele-UkH7L-aag8A-unsplash_Cactus.jpg"
input = "parallelLines45.png"

# imgIn = cv.imread(args.input, cv.IMREAD_GRAYSCALE)
# imgIn = cv.imread(input, cv.IMREAD_GRAYSCALE)
# print(imgIn[100,10])
# imgIn[100:110,10:50]=128
# if imgIn is None:
# 	print('Could not open or find the image: {}'.format(input))
# 	exit(0)

files = ["parallelLines45.png", "parallelLines90.png", "parallelLines135.png", "parallelLines180.png"]
for input in files:
	imgIn = cv.imread(input, cv.IMREAD_GRAYSCALE)
	abc = ST(imgIn)
	print("FileName: ", input)
	shape = np.shape(imgIn)
	print("Image shape:", shape)
	middle = [ int(shape[0]/2), int(shape[1]/2) ]
	imgCoherency, imgOrientation = abc.calcGST( W)
	abc.calculateEigenVector(middle[0], middle[1])
	print("***************************************")


# _, imgCoherencyBin = cv.threshold(imgCoherency, C_Thr, 255, cv.THRESH_BINARY)
# _, imgOrientationBin = cv.threshold(imgOrientation, LowThr, HighThr, cv.THRESH_BINARY)
# imgBin = cv.bitwise_and(imgCoherencyBin, imgOrientationBin)
# imgCoherency = cv.normalize(imgCoherency, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
# imgOrientation = cv.normalize(imgOrientation, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

# print("Orientation:", imgOrientation)
# print("Coherency:", imgCoherency)

# cv.imshow('result.jpg', np.uint8(0.5 * (imgIn + imgBin)))
# cv.imshow('Coherency.jpg', imgCoherency)
# cv.imshow('Orientation.jpg', imgOrientation)
# print("Average Orientation", np.average(imgOrientation) * 360 )




cv.waitKey(0)


