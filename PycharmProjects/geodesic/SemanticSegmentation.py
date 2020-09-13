# Original source: https://towardsdatascience.com/image-segmentation-with-six-lines-0f-code-acb870a462e8
# ade20k source: https://morioh.com/p/b04406d57772
# mask_rcnn_coco.h5: https://dev.to/ayoolaolafenwa/image-segmentation-with-5-lines-of-code-4lbd

from pixellib.semantic import semantic_segmentation
from pixellib.instance import instance_segmentation
import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import Bridson_Common

class Mask_RCNN:
	def __init__(self):
		self.model = instance_segmentation()
		self.model.load_model('mask_rcnn_coco.h5')

	def processImage(self, filename):
		path_to_image = './' + filename
		originalImage = cv2.imread(path_to_image)

		# segmask, output = self.model.segmentImage(path_to_image, show_bboxes=False,output_image_name=path_to_output_image)
		self.segmask, self.output = self.model.segmentImage(path_to_image, show_bboxes=False)

		self.genMask()

		return self.maskImage

	def getImageMaskMerged(self):
		return self.output

	def genMask(self):
		shape = np.shape(self.segmask['masks'])
		# print("Shape:", shape)
		mask = np.zeros((shape[0], shape[1]), dtype=(int, 3))

		truthMask = self.segmask['masks']

		for x in range(shape[0]):
			for y in range(shape[1]):
				if truthMask[x, y] == True:
					mask[x, y] = (255, 0, 0)

		mask = mask.astype(np.uint8)

		uniqueColours = set(tuple(v) for m2d in mask for v in m2d)
		print("Unique colours:", uniqueColours)  # https://stackoverflow.com/questions/24780697/numpy-unique-list-of-colors-in-the-image

		if Bridson_Common.semanticInvertMaskrcnn:
			for colour in uniqueColours:
				mask = np.bitwise_xor(mask, colour)
			# self.maskImage = cv2.bitwise_not(mask)  # Second implementation from here: https://stackoverflow.com/questions/19580102/inverting-image-in-python-with-opencv
		self.maskImage = mask

class Deeplabv3:
	def __init__(self):
		self.model = semantic_segmentation()
		self.model.load_ade20k_model("deeplabv3_xception65_ade20k.h5")

	def processImage(self, filename):
		path_to_image = './' + filename
		# originalImage = cv2.imread( path_to_image )

		rawLabels, output = self.model.segmentAsAde20k(path_to_image, overlay=False)

		dimensions = output.shape
		distance = min(dimensions[0], dimensions[1])
		# Make the distance 10% of the smaller dimension of the image.
		distance = int( distance / 10 )
		if distance % 2 == 0:
			distance += 1

		output = cv2.medianBlur(output, distance)

		# output = cv2.addWeighted(originalImage, 0.5, output, 0.5, 0.0)

		return output



if __name__ == '__main__':
	filenames = []
	# filenames.append('ruben-rodriguez-GFZZmRbyPFQ-unsplash_Egg.jpg')
	filenames.append('kaitlyn-ahnert-3iQ_t2EXfsM-unsplash.jpg')
	# filenames.append('BaseLineImage1.jpeg')
	# filenames.append('david-dibert-Huza8QOO3tc-unsplash.jpg')
	# filenames.append('ruslan-keba-G5tOIWFZqFE-unsplash_RubiksCube.jpg')
	# filenames.append('valentin-lacoste-GcepdU3MyKE-unsplash.jpg')

	type = 'mask_rcnn' # Valid values: 'deeplabv3', 'mask_rcnn'

	a = datetime.datetime.now()
	mask_rcnn = Mask_RCNN()
	deeplabv3 = Deeplabv3()
	b = datetime.datetime.now()

	print('Initialize:', (b-a).microseconds)

	for filename in filenames:
		path_to_image = './' + filename
		path_to_output_image = './' + 'segmented_' + filename
		originalImage = cv2.imread(path_to_image)

		d = datetime.datetime.now()
		output = mask_rcnn.processImage( filename )
		# maskImage = mask_rcnn.getMask()

		# mask = mask * (128.0,128.0,128.0)
		# mask = mask * segmask
		# print(mask)
		# plt.figure()
		# plt.imshow(mask)
		cv2.imshow('Masked region', output)
		# outputArray = np.asarray( output )
		# print("Unique values:", np.unique(outputArray, axis=1 ) )


		# plt.figure()
		# plt.imshow( maskImage )
		# output = deeplabv3.processImage( filename )
		# output = cv2.blur(output, (20,20))
		# dimensions = output.shape
		# distance = min(dimensions[0], dimensions[1])
		# distance = int( distance / 10 )
		# if distance % 2 == 0:
		# 	distance += 1
		# distance = 50
		# print("Distance:", distance)
		# # output = cv2.medianBlur(output, distance)
		#
		# output = cv2.bilateralFilter(output, distance, 75, 75 )
		# cv2.imshow('deeplabv3' + path_to_output_image, output)
		e = datetime.datetime.now()

		print('Saved image: ', filename, 'Processing time:', (e-d).microseconds )

	# plt.show()
	cv2.waitKey()

