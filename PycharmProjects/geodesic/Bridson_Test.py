# https://stackoverflow.com/questions/36727257/calculating-rotation-degrees-based-on-delta-x-y-movement-of-touch-or-mouse
import math
import Bridson_Common

for angle in [0, 22, 45, 77, 90, 115, 135, 145, 180, 190, 225, 245, 220]:
	print("============================================")
	print("Starting Angle:", angle)
	dx, dy = Bridson_Common.calculateDirection( angle )
	print("Dx Dy:", dx, dy)
	# //direction from old to new location in radians, easy to convert to degrees
	dir = ( math.atan2(dx, dy) * 180 / math.pi ) % 360;
	print("Recovered angle:", dir)

