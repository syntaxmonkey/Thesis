# Image modification library.
from PIL import Image, ImageEnhance
from scipy import io, misc
import matplotlib.pyplot as plt


# Increase edge contrast - hopefully it will impact segmentation

def increaseContrast(image, factor):
	enhancer = ImageEnhance.Contrast( image )
	output = enhancer.enhance( factor )
	return output

def toGrey(image):
	pass

def openImage(filename):
	image = misc.imread(filename)
	return image

if __name__ == '__main__':
	filename = 'ruslan-keba-G5tOIWFZqFE-unsplash_RubiksCube_small.png'

	image = Image.open( filename )
	image = image.convert('LA')
	plt.figure()
	plt.title("Original Image")
	plt.imshow(image)

	output = increaseContrast(image, 1.1)
	output = output.convert('LA')
	plt.figure()
	plt.title("Contrasted Image")
	plt.imshow(output)

	plt.show()