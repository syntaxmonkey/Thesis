from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np


def genLetter(boxsize=80, character = 'M', blur=1):
	fontsize = int(boxsize * 1.0)
	img = Image.new('RGB', (boxsize, boxsize), color=(255, 255, 255))
	# get a font
	# character = 'P'

	font = ImageFont.truetype("/System/Library/Fonts/Keyboard.ttf", fontsize)
	width, height = font.getsize(character)

	x = int((boxsize - width)/2)
	y = int((boxsize - height*1.3)/2) # Need to adjust for font height: https://websemantics.uk/articles/font-size-conversion/

	d = ImageDraw.Draw(img)
	d.text( (x,y) , character, fill=(0, 0, 0), font=font) # Add the text.

	# Blur the image.
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



if __name__ == '__main__':
	n = genLetter(boxsize=80, character = 'S', blur=2)

	img = Image.fromarray(n)
	img.save('pil_text.png')
	img.show()
