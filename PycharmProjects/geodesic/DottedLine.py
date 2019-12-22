from matplotlib import pyplot as plt
import numpy as np


class DottedLine:
	def __init__(self, firstpoint):
		self.points = np.array(firstpoint)

	def addPoint(self, secondpoint):
		self.points = np.vstack((self.points, secondpoint))




def on_click(event):
	global dt
	# https://stackoverflow.com/questions/41824662/how-to-plot-a-dot-each-time-at-the-point-the-mouse-is-clicked-in-matplotlib
	# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
	if event.inaxes == ax:
		print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
		# ax.plot(event.xdata, event.ydata, 'go')
		if dt == None:
			dt = DottedLine([event.xdata, event.ydata])
			ax.plot(event.xdata, event.ydata, 'or')
		else:
			ax.plot(event.xdata, event.ydata, 'or')
			dt.addPoint([event.xdata, event.ydata])
			# dt.points=np.array(dt.points)
			print(dt.points)
			xpoints = dt.points[:, 0]
			ypoints = dt.points[:, 1]
			ax.plot(xpoints, ypoints, '--b')
			lines.append(dt)
			dt = None
			print(lines)
		event.canvas.draw()

dt = None

lines = []
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('click to build line segments')
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
# line, = ax.plot([0], [0])  # empty line
# linebuilder = LineBuilder(line)

plt.gcf().canvas.mpl_connect('button_press_event', on_click)

plt.show()

