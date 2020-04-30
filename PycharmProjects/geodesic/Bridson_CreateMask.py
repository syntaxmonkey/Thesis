
import numpy as np
import matplotlib.pyplot as plt
import Bridson_Common

from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np


def genLetter(xsize, ysize, character = 'Y', blur=2):
	fontsize = min(xsize, ysize)
	# fontsize = int(boxsize * 1.1)
	img = Image.new('RGB', (xsize, ysize), color=(255, 255, 255))
	# get a font
	# character = 'P'

	# font = ImageFont.truetype("/System/Library/Fonts/Keyboard.ttf", fontsize)
	font = ImageFont.truetype("/System/Library/Fonts/Geneva.dfont", fontsize)
	width, height = font.getsize(character)

	x = int((xsize - width)/2)
	y = int((ysize - height*1.3)/2) + 1 # Need to adjust for font height: https://websemantics.uk/articles/font-size-conversion/

	d = ImageDraw.Draw(img)
	d.text( (x,y) , character, fill=(0, 0, 0), font=font) # Add the text.

	# Blur the image.
	if blur > 0:
		img = img.filter(ImageFilter.BoxBlur(blur))
		img = img.filter(ImageFilter.SMOOTH_MORE)

	# Flood file for masking.
	ImageDraw.floodfill(img, xy=(0, 0), value=(255, 0, 255), thresh=200) # https://stackoverflow.com/questions/46083880/fill-in-a-hollow-shape-using-python-and-pillow-pil

	# Fill in holes.
	n = np.array(img)
	n[(n[:, :, 0:3] != [255, 0, 255]).any(2)] = [0, 0, 0]
	# Revert all artifically filled magenta pixels to white
	n[(n[:, :, 0:3] == [255,0,255]).all(2)] = [255,255,255]

	img = Image.fromarray(n)
	img = img.convert('L')
	# print(img.size)

	n = np.array(img)
	n = np.reshape(n, img.size)
	# print(np.shape(n))
	# print(n)
	n = 255 - n # Need to flip the bits.  The Freeman chain code generator requires the letter portion to have a value of 255 instead of 0.
	return(n)



def CreateCircleMask(xrange, yrange, radius):
	# Given dimensions, we will create a mask.
	# For now, create a circle.

	mask = np.zeros((xrange,yrange))
	centerx = xrange / 2.0
	centery = yrange / 2.0

	for xpoint in range(1, xrange-1):
		for ypoint in range(1, yrange-1):
			if Bridson_Common.euclidean_distance((xpoint,ypoint), (centerx,centery)) <= radius:
				mask[xpoint,ypoint] = 255

	return mask

def InvertMask(mask):
	# Given a mask, invert the regions.
	# Converts the 255 regions to 0 and converts the zero regions to 255.
	newMask = 255 - mask
	return newMask


def generatePoints(mask):
	x, y = np.shape(mask)
	points = []
	for i in range(x):
		for j in range(y):
			if mask[i][j] == 255:
				points.append((i,j))
	return np.array(points)

if __name__ == '__main__':
	x, y = 100, 100
	radius = 5
	mask = CreateCircleMask(x, y, radius)
	# print(mask)

	invertedMask = InvertMask(mask)

	invertedMask = genLetter(x, y, character = 'A', blur=2)
	invertedMask = InvertMask(invertedMask)
	plt.imshow(invertedMask)
	plt.show()



