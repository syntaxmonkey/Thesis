# Original source: https://towardsdatascience.com/image-segmentation-with-six-lines-0f-code-acb870a462e8
# ade20k source: https://morioh.com/p/b04406d57772
# mask_rcnn_coco.h5: https://dev.to/ayoolaolafenwa/image-segmentation-with-5-lines-of-code-4lbd

from pixellib.semantic import semantic_segmentation
from pixellib.instance import instance_segmentation
import datetime
import cv2



class Mask_RCNN:
	def __init__(self):
		self.model = instance_segmentation()
		self.model.load_model('mask_rcnn_coco.h5')

	def processImage(self, filename):
		path_to_image = './' + filename

		# segmask, output = self.model.segmentImage(path_to_image, show_bboxes=False,output_image_name=path_to_output_image)
		segmask, output = self.model.segmentImage(path_to_image, show_bboxes=False)
		return output

class Deeplabv3:
	def __init__(self):
		self.model = semantic_segmentation()
		self.model.load_ade20k_model("deeplabv3_xception65_ade20k.h5")

	def processImage(self, filename):
		path_to_image = './' + filename

		rawLabels, output = self.model.segmentAsAde20k(path_to_image, overlay=True)
		return output



if __name__ == '__main__':
	filenames = []
	filenames.append('ruben-rodriguez-GFZZmRbyPFQ-unsplash_Egg.jpg')
	filenames.append('kaitlyn-ahnert-3iQ_t2EXfsM-unsplash.jpg')
	filenames.append('BaseLineImage1.jpeg')
	filenames.append('david-dibert-Huza8QOO3tc-unsplash.jpg')
	filenames.append('ruslan-keba-G5tOIWFZqFE-unsplash_RubiksCube.jpg')
	filenames.append('valentin-lacoste-GcepdU3MyKE-unsplash.jpg')

	type = 'mask_rcnn' # Valid values: 'deeplabv3', 'mask_rcnn'

	a = datetime.datetime.now()
	mask_rcnn = Mask_RCNN()
	deeplabv3 = Deeplabv3()
	# if type == 'mask_rcnn':
	# 	segment_image = instance_segmentation()
	# else:
	# 	segment_image = semantic_segmentation()

	b = datetime.datetime.now()
	# if type == 'mask_rcnn':
	# 	segment_image.load_model('mask_rcnn_coco.h5')
	# else:
	# 	segment_image.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
	c = datetime.datetime.now()

	print('Initialize:', (b-a).microseconds)
	print('Load Model:', (c-b).microseconds)

	for filename in filenames:
		path_to_image = './' + filename
		path_to_output_image = './' + 'segmented_' + filename

		# segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
		# segment_image.load_pascalvoc_model("mask_rcnn_coco.h5")
		# segment_image.load_model("mask_rcnn_coco.h5")

		d = datetime.datetime.now()
		# segment_image.segmentAsPascalvoc(path_to_image, output_image_name = path_to_output_image, overlay = True)
		# if type == 'mask_rcnn':
			# segmask, output = segment_image.segmentImage(path_to_image, show_bboxes=False,output_image_name=path_to_output_image)
		output = mask_rcnn.processImage( filename )
		# print('segmask:', segmask)
		cv2.imshow('mask_rcnn' + path_to_output_image, output)
		# else:
			# rawLabels, image_overlay = segment_image.segmentAsAde20k(path_to_image, overlay=False,output_image_name=path_to_output_image)
		output = deeplabv3.processImage( filename )
		# cv2.imwrite(path_to_output_image, image_overlay)
		cv2.imshow('deeplabv3' + path_to_output_image, output)
		e = datetime.datetime.now()

		print('Saved image: ', filename, 'Processing time:', (e-d).microseconds )

	cv2.waitKey()
# print(F"Hello")

