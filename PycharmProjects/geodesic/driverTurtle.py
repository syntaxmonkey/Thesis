import numpy as np
import matplotlib
matplotlib.use('tkagg') # Fix for turtle and Tk: https://github.com/matplotlib/matplotlib/issues/10239/

import matplotlib.pyplot as plt # For displaying array as image
from PIL import Image, ImageDraw
#Program to draw concentric circles in Python Turtle
import turtle # Draw with Turtle: https://www.tutorialsandyou.com/python/how-to-draw-circle-in-python-turtle-7.html

print('Generating geodesic')

xsize = 100
ysize = 100

centerx = xsize / 2
centery = ysize / 2
radius = xsize / 2 - 1


def generateRaster():
	raster = [[0 for i in range(xsize)] for j in range(ysize)]
	#raster = np.xrange((xsize,ysize))

	print(raster)


	# Display array as image: https://stackoverflow.com/questions/3886281/display-array-as-raster-image-in-python
	x, y = np.meshgrid(np.linspace(-2,2,xsize), np.linspace(-2,2,ysize))
	x, y = x - x.mean(), y - y.mean()
	z = x * np.exp(-x**2 - y**2)

	plt.imshow(z)
	plt.gray()
	plt.show()






def genTurtleCircle():
	t = turtle.Turtle()

	for i in range(2):
		t.circle(10*i)
		t.up()
		t.sety((10*i)*(-1))
		t.down()

	cv = t.getscreen().getcanvas()
	cv

#generateRaster()
#baseCircle = genCircle()
#print(baseCircle)
#drawCircle()

genTurtleCircle()