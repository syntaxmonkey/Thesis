from PIL import Image, ImageFont, ImageDraw
import numpy as np


def genLetter():
	boxsize = 100
	fontsize = int(boxsize * 0.8)
	img = Image.new('RGB', (boxsize, boxsize), color=(255, 255, 255))

	# get a font
	character = 'P'

	font = ImageFont.truetype("/System/Library/Fonts/Keyboard.ttf", fontsize)
	width, height = font.getsize(character)

	x = int((boxsize - width)/2)
	y = int((boxsize - height*1.3)/2) # Need to adjust for font height: https://websemantics.uk/articles/font-size-conversion/

	d = ImageDraw.Draw(img)
	d.text( (x,y) , character, fill=(0, 0, 0), font=font)

	# Flood file for masking.
	ImageDraw.floodfill(img, xy=(0, 0), value=(255, 0, 255), thresh=200) # https://stackoverflow.com/questions/46083880/fill-in-a-hollow-shape-using-python-and-pillow-pil

	# Fill in holes.
	n = np.array(img)
	n[(n[:, :, 0:3] != [255, 0, 255]).any(2)] = [0, 0, 0]
	# Revert all artifically filled magenta pixels to white
	n[(n[:, :, 0:3] == [255,0,255]).all(2)] = [255,255,255]

	return(n)



if __name__ == '__main__':
	n = genLetter()

	img = Image.fromarray(n)
	img.save('pil_text.png')
	img.show()
