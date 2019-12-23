from matplotlib import pyplot as plt
import numpy as np
import sys

class DottedLine:
	def __init__(self, firstpoint):
		self.points = np.array(firstpoint)

	def addPoint(self, secondpoint):
		self.points = np.vstack((self.points, secondpoint))



# Get the end points.
# Calculate the temporary points.



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
		fig.canvas.draw()


def on_click(event):
	global dt, startPoint, tempLine
	# https://stackoverflow.com/questions/41824662/how-to-plot-a-dot-each-time-at-the-point-the-mouse-is-clicked-in-matplotlib
	# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
	if event.inaxes == ax:
		# print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
		# ax.plot(event.xdata, event.ydata, 'go')
		if dt == None:
			dt = DottedLine([event.xdata, event.ydata])
			# ax.plot(event.xdata, event.ydata, 'or')
		else:
			# startPoint, = ax.plot(event.xdata, event.ydata, 'or')
			linePoints = generateLinePoints(dt.points, [event.xdata, event.ydata]) # Generate the points along the line.

			dt.addPoint([event.xdata, event.ydata])
			# dt.points=np.array(dt.points)
			print(dt.points)
			xpoints = dt.points[:, 0]
			ypoints = dt.points[:, 1]
			# ax.plot(xpoints, ypoints, '--b')
			lines.append(dt)
			dt = None
			startPoint = None
			tempLine = None
			print(lines)
		event.canvas.draw()


def generateLinePoints(startPoint, endPoint, segmentLength=5):
	return []



def on_move(event):
	global dt, tempLine
	# https://stackoverflow.com/questions/41824662/how-to-plot-a-dot-each-time-at-the-point-the-mouse-is-clicked-in-matplotlib
	# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
	if event.inaxes == ax and dt != None:
		# ax.plot(event.xdata, event.ydata, 'or')
		# dt.addPoint([event.xdata, event.ydata])
		# dt.points=np.array(dt.points)
		# print(dt.points)
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

