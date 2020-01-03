from matplotlib import pyplot as plt
import numpy as np
import sys
import math

class DottedLine:
	def __init__(self, firstpoint):
		self.points = np.array(firstpoint)

	def addPoint(self, secondpoint):
		self.points = np.vstack((self.points, secondpoint))






# Key Press event: https://matplotlib.org/examples/event_handling/keypress_demo.html
def press(event):
	global dt, startPoint, tempLine
	print('press', event.key)
	sys.stdout.flush()
	# with NonBlockingConsole() as nbc:
	if event.key == 'escape' and tempLine != None:
		# visible = xl.get_visible()
		# xl.set_visible(not visible)
		print('escape pressed')
		tempLine.remove() # Delete the line. https://stackoverflow.com/questions/4981815/how-to-remove-lines-in-a-matplotlib-plot
		dt = startPoint = tempLine = None
		event.canvas.draw()




def on_click(event):
	global dt, tempLine
	# https://stackoverflow.com/questions/41824662/how-to-plot-a-dot-each-time-at-the-point-the-mouse-is-clicked-in-matplotlib
	# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
	if event.inaxes == ax:
		handleDottedLine(event, event.inaxes)


def handleDottedLine(event, ax):
	global dt, tempLine, lines
	linePoints = []
	if dt == None:
		dt = DottedLine([event.xdata, event.ydata])
		# ax.plot(event.xdata, event.ydata, 'or')
	else:
		linePoints = createDottedLine(ax, dt.points, (event.xdata, event.ydata))
		tempLine.remove()
		# dt.addPoint([event.xdata, event.ydata])
		# print(dt.points)
		dt = None
		tempLine = None

	event.canvas.draw()
	return linePoints


def createDottedLine(ax, startPoint, endPoint):
	linePoints = generateLinePoints(startPoint, endPoint)  # Generate the points along the line.
	newLinePoints = []
	# Plot dots.
	for linePoint in linePoints:
		dot, = ax.plot(linePoint[0], linePoint[1], 'or')
		newLinePoints.append(dot)

	linePoints = newLinePoints
	return linePoints




def generateLinePoints(startPoint, endPoint, segmentLength=10):
	x1, y1 = startPoint
	x2, y2 = endPoint

	deltax = x2 - x1
	deltay = y2 - y1
	pointDistance = math.sqrt( math.pow(deltax, 2) + math.pow(deltay, 2) )

	segmentCount = math.floor(pointDistance / segmentLength)
	if (segmentCount == 0):
		segmentCount = 1
	adjustedSegmentDistance = pointDistance / segmentCount
	segmentDistance = adjustedSegmentDistance / pointDistance

	print("Segment Count %d" % (segmentCount))
	points = []
	for i in range(segmentCount):
		points.append((x1 + segmentDistance*deltax*i, y1 + segmentDistance*deltay*i))
	points.append((x2,y2))

	return points



def on_move(event):
	global dt
	# https://stackoverflow.com/questions/41824662/how-to-plot-a-dot-each-time-at-the-point-the-mouse-is-clicked-in-matplotlib
	# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
	if event.inaxes == ax and dt != None:
		handleMove(event)


def handleMove(event, ax):
	global tempLine, dt
	# ax.plot(event.xdata, event.ydata, 'or')
	# dt.addPoint([event.xdata, event.ydata])
	# dt.points=np.array(dt.points)
	# print(dt.points)
	if dt != None:
		xpoints = (dt.points[0], event.xdata) # Need to construct temporary line points.
		ypoints = (dt.points[1], event.ydata) # Need to construct temporary line points.
		if tempLine == None:
			tempLine, = ax.plot(xpoints, ypoints, '--b')
		else:
			tempLine.set_data(xpoints, ypoints)
		# print(dir(tempLine))

		# lines.append(dt)
		# dt = None
		# print(lines)
		event.canvas.draw()


dt = None
tempLine = None
startPoint = None
currentAxis = None


if __name__ == '__main__':
	lines = []
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title('click to build line segments')
	ax.set_xlim([0, 5])
	ax.set_ylim([0, 5])
	# line, = ax.plot([0], [0])  # empty line
	# linebuilder = LineBuilder(line)

	plt.gcf().canvas.mpl_connect('button_press_event', on_click)
	plt.gcf().canvas.mpl_connect('motion_notify_event', on_move)
	plt.gcf().canvas.mpl_connect('key_press_event', press)

	plt.show()

