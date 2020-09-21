# https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.structure_tensor

# Tutorial on structure tensor: https://www.mathworks.com/matlabcentral/fileexchange/12362-structure-tensor-introduction-and-tutorial

# Tensor Structure testg

from skimage.feature import structure_tensor
from skimage.feature import structure_tensor_eigvals
import numpy as np
import math

import skimage
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from skimage import data, img_as_float
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from numpy.linalg import eig

from multiprocessing import Process, freeze_support, set_start_method, Pool
import time
import random

import uuid

print("Test")
if False:
	square = np.ones((5, 5))
	square[2:5, 3] = 5
	square[2:5, 2] = 3

	print("Square:", square)

	Arr, Arc, Acc = structure_tensor(square, sigma=0.1)


	print(square)
	print(Arr)
	print(Arc)
	print(Acc)

	e1 = structure_tensor_eigvals(Arr, Arc, Acc)[0]
	e2 = structure_tensor_eigvals(Arr, Arc, Acc)[1]
	print("E1", e1 )
	print("E2", e2 )
	print("Structure Tensor Eigenvalue", structure_tensor_eigvals(Arr, Arc, Acc) )

	I1 = np.linalg.eig(e1)
	I2 = np.linalg.eig(e2)

	print("I1", I1)
	print("I2", I2)


	npE = np.linalg.eigvals(square)
	print("Numpy EigenValues", npE)


if False:
	image1 = '/Users/hengsun/Downloads/larry-ferreira-9Rs5eop48KQ-unsplash.jpg'
	image = skimage.io.imread(image1)

	greyscaleImage = Image.fromarray(image).convert('L')
	plt.figure()
	plt.imshow(image)
	plt.figure()
	plt.imshow(greyscaleImage)

	image = skimage.io.imread(image1)
	# skimage = skimage.color.rgb2lab(image)
	# plt.figure()
	# plt.imshow(skimage)

	skimageGrey = skimage.color.rgb2grey(image)
	plt.figure()
	plt.imshow(skimageGrey)


	plt.show()

if False:

	matplotlib.rcParams['font.size'] = 8


	def plot_img_and_hist(image, axes, bins=256):
		"""Plot an image along with its histogram and cumulative histogram.

		"""
		image = img_as_float(image)
		ax_img, ax_hist = axes
		ax_cdf = ax_hist.twinx()

		# Display image
		ax_img.imshow(image, cmap=plt.cm.gray)
		ax_img.set_axis_off()

		# Display histogram
		ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
		ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
		ax_hist.set_xlabel('Pixel intensity')
		ax_hist.set_xlim(0, 1)
		ax_hist.set_yticks([])

		# Display cumulative distribution
		img_cdf, bins = exposure.cumulative_distribution(image, bins)
		ax_cdf.plot(bins, img_cdf, 'r')
		ax_cdf.set_yticks([])

		return ax_img, ax_hist, ax_cdf


	# Load an example image
	img = data.moon()
	# img = np.array( Image.fromarray(img).convert('L') )
	# img = np.asarray( img )

	# Contrast stretching
	p2, p98 = np.percentile(img, (2, 98))
	img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

	# Equalization
	img_eq = exposure.equalize_hist(img)

	print("Type:", type(img))
	print("Shape:", np.shape(img))
	# Adaptive Equalization
	img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
	img_adapteq2 = exposure.equalize_adapthist(img)
	plt.figure()
	plt.imshow(img)
	plt.title("Original")

	plt.figure()
	plt.imshow(img_eq)
	plt.title("Euqalize Histogram")

	img_eq_median = skimage.filters.median(img_eq)
	plt.figure()
	plt.imshow(img_eq_median)
	plt.title("Euqalize Histogram Median")

	plt.figure()
	plt.imshow(img_adapteq)
	plt.title("Euqalize Adaptive Histogram")

	plt.figure()
	plt.imshow(img_adapteq2)
	plt.title("Euqalize Adaptive Histogram 2")
	img_median = skimage.filters.median( img_adapteq2 )


	plt.figure()
	plt.imshow(img_median)
	plt.title("Euqalize Adaptive Histogram 2 median")



	print(skimage.__version__)

	# Equalization
	# selem = disk(30)
	# img_loc_eq = rank.equalize(img, selem=selem)
	# plt.figure()
	# plt.imshow(img_loc_eq)
	# plt.title("Local Equalize")


	# img_meijering = skimage.filters.meijering( img_adapteq2 )
	# plt.figure()
	# plt.imshow( img_meijering )
	# plt.title("Adaptive Histogram 2 meijering")
	# #
	# # Display results
	# fig = plt.figure(figsize=(8, 5))
	# axes = np.zeros((2, 4), dtype=np.object)
	# axes[0, 0] = fig.add_subplot(2, 4, 1)
	# for i in range(1, 4):
	# 	axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
	# for i in range(0, 4):
	# 	axes[1, i] = fig.add_subplot(2, 4, 5 + i)
	#
	# ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
	# ax_img.set_title('Low contrast image')
	#
	# y_min, y_max = ax_hist.get_ylim()
	# ax_hist.set_ylabel('Number of pixels')
	# ax_hist.set_yticks(np.linspace(0, y_max, 5))
	#
	# ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
	# ax_img.set_title('Contrast stretching')
	#
	# ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
	# ax_img.set_title('Histogram equalization')
	#
	# ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
	# ax_img.set_title('Adaptive equalization')
	#
	# ax_cdf.set_ylabel('Fraction of total intensity')
	# ax_cdf.set_yticks(np.linspace(0, 1, 5))
	#
	# # prevent overlap of y-axis labels
	# fig.tight_layout()
	plt.show()

external = 'name'

if False:
	def f():
		time.sleep( random.random()*5 + 2 )
		print('hello,', external)

	def wrapper(name, secondName):
		global external
		print(name)
		external=secondName
		f()

	with Pool(processes=4) as pool:
		names = [('bob', 'john'), ('mack', 'another')]
		pool.starmap(wrapper, names)

if False:
	unique_filename = str(uuid.uuid4().hex)
	print(unique_filename)

	# for name in ['bob', 'john', 'mack']:
	# 	# freeze_support()
	# 	# set_start_method('spawn')
	# 	external = name
	# 	p = Process(target=f)
	# 	p.start()
	# 	# p.join()

	# p = Pool(5)
	# p.map(f, ['bob', 'john', 'mack'])

# print("Coherency:", np.power((I1 - I2) / (I1+I2), 2) )
# array([[0., 0., 0., 0., 0.],
#        [0., 1., 0., 1., 0.],
#        [0., 4., 0., 4., 0.],
#        [0., 1., 0., 1., 0.],
#        [0., 0., 0., 0., 0.]])


# import math

# intensityValues = [255, 200, 128, 100, 50, 30, 10, 1,0]
# for intensity in intensityValues:
# 	print("Intensity:", intensity, "produces", math.exp(intensity/50))



# Utilize colormath to compare two different colours.


from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

rgb1 = sRGBColor(255, 255, 0, True)
cie1 = convert_color(rgb1, LabColor)
print(rgb1, "-->", cie1)

rgb2 = sRGBColor(0, 255, 0, True)
cie2 = convert_color(rgb2, LabColor)
print(rgb2, "-->", cie2)

rgb3 = sRGBColor(0, 0, 255, True)
cie3 = convert_color(rgb3, LabColor)
print(rgb3, "-->", cie3)


print("rgb1 --> rgb2", delta_e_cie2000(cie1, cie2))
print("rgb1 --> rgb3", delta_e_cie2000(cie1, cie3))
print("rgb2 --> rgb3", delta_e_cie2000(cie2, cie3))