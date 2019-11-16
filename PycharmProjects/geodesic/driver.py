import numpy as np
import matplotlib
matplotlib.use('tkagg')
import time

import matplotlib.pyplot as plt # For displaying array as image
from PIL import Image, ImageDraw

import math
#from poisson_disc_bridson import poisson_disc_samples

print('Generating geodesic')

tracerLines = False

xsize = 100
ysize = 100

centerx = xsize / 2
centery = ysize / 2
center = (centerx, centery)
radius = xsize / 2 - 1
perimeterCoords = []

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




def genCircle(xsize, ysize):
	# Draw Circle Image: https://code-maven.com/create-images-with-python-pil-pillow
	image = Image.new('RGB', (xsize,ysize), color='black') # Modes defined here: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes

	draw = ImageDraw.Draw(image)
	#draw.ellipse((centerx - radius, centery-radius, centerx+radius, centery+radius), fill=(0), outline=(255,255,255)) # Drawing outline: https://www.geeksforgeeks.org/python-pil-imagedraw-draw-ellipse/

	angle=0 # Start at angle 0.
	segments = 50
	angleIncrement = 360/segments

	for i in range(segments):
		angle = i*angleIncrement
		perimeterPoint = polar_point(center, angle, radius)
		perimeterCoords.append(perimeterPoint)
		if tracerLines:
			shape=[center, perimeterPoint] # Radial lines.
			draw.line(shape, fill=(255, 0, 0), width=1)
		else:
			shape=[perimeterPoint, perimeterPoint] # Draw red dots on the perimeter.
			draw.line(shape, fill=(255,255,255), width=1)

	pix = np.array(image) # Convert image to array: https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array
	#print(pix)
	return(pix)


def genCircleCoords(xsize, ysize,  center, radius, segments=5):
	# Draw Circle Image: https://code-maven.com/create-images-with-python-pil-pillow
	angle=0 # Start at angle 0.
	angleIncrement = 360/segments


	for i in range(segments):
		angle = i*angleIncrement
		perimeterPoint = polar_point(center, angle, radius)
		perimeterCoords.append(perimeterPoint)

	#print(pix)
	return perimeterCoords



# Calculate the polar point: https://gis.stackexchange.com/questions/67478/how-to-create-a-circle-vector-layer-with-12-sectors-with-python-pyqgis
# helper function to calculate point from relative polar coordinates (degrees)
def polar_point(origin_point, angle,  distance):
    #return (origin_point[0] + math.sin(math.radians(angle)) * distance, origin_point[1] + math.cos(math.radians(angle)) * distance)
    return [origin_point[0] + math.sin(math.radians(angle)) * distance,
            origin_point[1] + math.cos(math.radians(angle)) * distance
            ]

def drawCircle():
	circle = genCircle(xsize, ysize)

	plt.imshow(circle)
	plt.gray()
	plt.show()




#generateRaster()
#baseCircle = genCircle()
#print(baseCircle)
#drawCircle()
'''
circle=genCircle(xsize, ysize)

flattenedCircle = list(circle.flatten())
print(flattenedCircle)

k = 50
#xsize = 20
#ysize = 20
startTime = int(round(time.time() * 1000))

samples = poisson_disc_samples(width=xsize, height=ysize, r=5)

endTime = int(round(time.time() * 1000))

print("Execution time: " + str(endTime - startTime))

raster = [[0 for i in range(xsize)] for j in range(ysize)]

for coords in samples:
	x, y = coords
	xint = int(x)
	yint = int(y)
	raster[xint][yint] = int(255)

print(raster)
plt.imshow(raster)
plt.gray()
plt.show()



exit(0)
'''