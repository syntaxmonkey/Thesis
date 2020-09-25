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

if False:
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

from skimage.future import graph
from skimage import data, io, segmentation, color
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage import draw
import numpy as np



def show_img(img):
	width = 10.0
	height = img.shape[0] * width / img.shape[1]
	f = plt.figure(figsize=(width, height))
	plt.imshow(img)



def display_edges(image, g, threshold):
	"""Draw edges of a RAG on its image

	Returns a modified image with the edges drawn.Edges are drawn in green
	and nodes are drawn in yellow.

	Parameters
	----------
	image : ndarray
		The image to be drawn on.
	g : RAG
		The Region Adjacency Graph.
	threshold : float
		Only edges in `g` below `threshold` are drawn.

	Returns:
	out: ndarray
		Image with the edges drawn.
	"""
	image = image.copy()
	# for edge in g.edges_iter():
	for edge in g.edges:
		n1, n2 = edge

		r1, c1 = map(int, rag.nodes[n1]['centroid'])
		r2, c2 = map(int, rag.nodes[n2]['centroid'])

		line = draw.line(r1, c1, r2, c2)
		circle = draw.circle(r1, c1, 2)

		if g[n1][n2]['weight'] < threshold:
			image[line] = 0, 1, 0
		image[circle] = 1, 1, 0

	return image



if False:
	img = data.coffee()
	show_img(img)

	labels = segmentation.slic(img, compactness=30, n_segments=400)
	print(labels)
	# labels = labels + 1  # So that no labelled region is 0 and ignored by regionprops
	# regions = regionprops(labels)

	label_rgb = color.label2rgb(labels, img, kind='avg')
	show_img(label_rgb)

	label_rgb = segmentation.mark_boundaries(label_rgb, labels, (0, 0, 0))
	show_img(label_rgb)

	rag = graph.rag_mean_color(img, labels)

	# print("RAG nodes", rag.nodes)  # the nodes are the regions.
	# print("RAG edges", rag.edges)  # the edges are the pairs of adjacency regions.

	plt.show()


import matplotlib.pyplot as plt
rng = np.random.RandomState(10)  # deterministic random data
a = [1.1722387933044753, 0.6339870657719583, 0.0, 0.9095586690079516, 0.9995492123633111, 0.7971820117960954, 0.4402927786879194, 1.1290411027404785, 0.8986298019267875, 0.5300803040066302, 1.030504375723123, 0.6166941933783106, 1.512884138392961, 0.918977331294466, 1.512884138392961, 0.5428982800879159, 1.2851937620616645, 0.51586576088942, 1.8072364585094984, 1.3851307202159409, 2.160776967202136, 1.033396529180652, 0.5158657608894209, 1.8821615223518273, 0.9324541904873698, 1.0685335570859076, 0.21128694552520144, 1.4833252011881966, 2.1718660144102344, 7.9030999723524555, 2.3781084472859484, 7.751307639543284, 54.64398581720799, 1.9564378058171408, 7.345415468733228, 57.25536833923069, 0.517465743100278, 8.882542498456758, 1.0021087382687222, 1.474120959544485, 8.856730573444588, 0.4770315882812865, 1.68109492776178, 2.1734630357104416, 0.4882864598181419, 1.7411945402666744, 2.5802277059296492, 0.49985092841077916, 2.285169391867587, 2.8448271974975223, 1.244229889411558, 2.1245290552627294, 2.502097629827144, 1.7619392701818715, 2.965940659094384, 1.3851307202159409, 1.981703083837131, 2.8805638948254413, 0.5647323813779708, 0.9698006149492666, 3.4319008229557553, 0.0, 0.6321849089530971, 1.2606458196615191, 9.2500728921213, 2.874933213762586, 3.0725744590909505, 0.9889317707566184, 1.222243408240615, 2.7617561058812305, 0.4972677118333698, 0.633987065771959, 0.7950154425069685, 0.9749216543170842, 2.214427343199861, 2.5068547638777114, 60.18035796856086, 3.145335017766601, 55.03919321774138, 53.711523626127615, 1.7287906544545664, 3.0212561978635692, 2.6360222762793866, 8.435298363788206, 56.21750569794485, 0.6361106281381592, 0.5031482819982256, 2.446464562702543, 2.5891291839509663, 1.8315106787035313, 2.4771136275865584, 1.523861679429855, 1.9366687429556129, 4.345025931531784, 1.5654064787356285, 2.3539788526717516, 0.6285583746671805, 1.1645583385990845, 1.8916574370125305, 1.2606458196615191, 1.8320731260739496, 2.3584997987204694, 0.494760545200073, 1.3482371823191712, 2.224028841251591, 3.676002662581086, 1.957810832209265, 3.809625658664046, 3.3317346585809955, 2.7965405593437014, 2.8565991523542684, 55.86217666842671, 3.0277728027580384, 0.4484945116093227, 0.9088024265868323, 0.45074376340107325, 0.9214816387517053, 2.172646495889679, 0.8109371119105168, 0.8841187461102981, 0.7793625961849016, 1.267706448299642, 1.3463023854341234, 2.206117979885172, 58.915568334914305, 3.3608848242764346, 2.677347770574479, 0.9542862534076239, 2.0102150186054035, 1.7948315263205998, 3.0042148974963854, 60.47104157177224, 59.353451297881755, 3.7962822939808967, 56.39406166632926, 55.81854290801916, 54.98164681015636, 3.3905214493489786, 2.804921281374936, 53.452775880096, 1.5121214938477794, 1.3485760932340167, 1.6636781769890483, 1.6636781769890483, 1.2136766632438865, 0.4171444837654843, 41.39838951676656, 46.17853335095088, 3.522992836451153, 3.353481192007555, 50.041834507237574, 44.86575172930376, 42.96490136722811, 1.400197312854589, 3.113786739550549, 2.6178662220967692, 41.032987890583975, 0.7512035244632475, 2.4965479358908595, 0.0, 0.7667674409992801, 1.812482023825305, 59.007010952940455, 3.051145579372404, 4.388247950069298, 4.213808610737562, 2.3498716290948174, 3.167090870221429, 2.496547935890858, 38.85636776745534, 52.45180149334427, 52.8900571719785, 1.1537254128512002, 0.7512035244632406, 1.4222180355946377, 0.9990428929814407, 2.0941584684413326, 0.5104128752382455, 1.2568366278385399, 0.8520028409662576, 1.8042818828358893, 1.1421994352256672, 0.0, 0.6136345928730329, 0.6241259189194451, 1.6185384889543588, 6.557990511603852, 4.372947821944699, 0.9342358441607719, 2.1143736201136654, 0.4632373710341722, 0.4458444013616414, 1.9645796646477205, 0.42526051760285655, 0.8426576811827249, 5.404481021250635, 12.045855719817862, 5.364193819479131, 52.45364323713513, 22.34586435034257, 1.893810915918034, 1.8547100471645703, 2.002600937361884, 1.9919889970525086, 0.8779903040930324, 5.918903241009316, 1.7634266762547146, 4.903566421224699, 0.6136345928730329, 0.7925458380813071, 1.1857229461266412, 0.9996555855636009, 2.28183611811414, 1.7945534446882165, 1.229069175218651, 0.6497874234669593, 7.378366477158958, 1.8146814770598905, 1.1398316148718999, 2.950730331773435, 2.1948643206334997, 2.6772242510991564, 2.172916454207826, 2.6062989682877076, 42.01667437371807, 8.248632812511364, 6.056271716174368, 52.79369175465012, 2.4313657905355344, 40.60754384180658, 4.0172059997193434, 0.7590231124990311, 0.8093671612473243, 4.074105919159385, 0.7828282892248263, 4.11794292799811, 3.3619291963448656, 6.037009355854339, 3.4989873024972815, 1.8549423169769463, 13.99224239481686, 0.4480625677332528, 0.49517066144775373, 8.419557382227028, 1.4110855470122285, 2.813054034446769, 4.795439941575826, 6.785746987435619, 8.504647633569226, 3.382968511407355, 15.84169842219408, 8.001149046421245, 0.41735457885447896, 8.136097954707418, 7.891248400369542, 24.05038070540656, 33.92033386208048, 24.08154836484448, 31.1660540995202, 0.4825492837270243, 2.460672442721773, 5.425561114129288, 7.844813356105014, 5.102387484018969, 3.382968511407355, 15.89073111276084, 3.7443481430248124, 3.9551947658512527, 5.332788818708537, 2.094727936275106, 7.934114213434131, 0.5168283286372021, 3.257240267323783, 6.614140369939124, 14.212149867850806, 0.4899067909488913, 15.26610798607305, 2.888581287746139, 12.507161397182921, 13.719864336514847, 12.590641701778075, 15.254380082828924, 6.491105309822373, 1.0015475273958265, 6.548246042809099, 6.546823836564188, 6.819062346938806, 38.88011281391471, 39.13436600115514, 8.53727747225454, 45.41145310365529, 18.850690472724978, 13.87815861064137, 17.095389780946654, 16.962649524730192, 4.157606269375534, 21.498645759083573, 11.930654528332644, 1.2764348353270143, 7.9292599340795515, 3.83346856108445, 2.442330012550474, 2.7249840447864644, 2.8147803878871693, 2.1702231931538583, 0.2578005918050065, 11.733528445612924, 10.352044349172292, 1.5537890576395532, 3.0594445681613256, 1.4240999121323323, 14.212006076511496, 46.13878087474242, 38.19935815169687, 11.19458533922005, 3.0918510045149756, 17.625787512769193, 1.5749494876789318, 2.290638880393314, 18.742804140229197, 17.7506487795791, 14.736776821254656, 2.9370329382607996, 10.6007331675929, 11.07235153078196, 13.711582888196254, 14.615023227061814, 1.7220827821620657, 43.1389906734547, 45.7191099733481, 20.832885260420785, 4.142765543522805, 43.381243202747605, 1.6221145108218649, 6.033262101850122, 5.254040459673878, 0.7860206185549705, 3.9764109542922808, 3.171007279776923, 0.905022870019571, 2.999668202637628, 1.069643310525656, 3.9305645570805137, 14.492095073745235, 23.148886526841228, 18.80691611965448, 24.19902041314847, 2.1662597290618573, 3.2057377195613443, 3.036198398349104, 0.9705945437849719, 3.1583078209193087, 14.253051778564219, 7.835967458832868, 34.182190863619496, 8.90726962479876, 8.464555606821403, 25.14051338829862, 27.044341546138554, 1.632104883941433, 14.309946662319296, 9.832822627953192, 15.99573023750765, 2.397530692944928, 2.253337504386983, 1.6466957613628266, 2.287999538983152, 5.696908916784689, 3.99148427095329, 1.2143244972556633, 4.189355407832811, 6.329401073413906, 18.42052193992468, 3.6082066287376295, 3.25927174201821, 24.459910441002975, 42.30537549678488, 0.7237100844228975, 0.827250187696741, 16.812702053702182, 26.112048470845416, 16.033753057141535, 1.4318533043672828, 16.170180918396735, 27.81042683635276, 30.30665970702112, 11.401012872150085, 25.344092544699542, 30.309918220892314, 24.394277545594477, 26.207468907200358, 31.50158785813471, 41.5384155632896, 30.041078212752527, 19.74686952527058, 25.105961956149198, 24.561345947058534, 39.158968990595746, 18.141421885636206, 0.7244115477149307, 3.77310890155241, 6.291530757973375, 6.902515531926483, 0.4778085754522888, 0.7034306638331979, 2.9610708863124806, 31.80182234217677, 1.475762684149016, 2.652922878662328, 13.177884169707246, 10.866466120546127, 27.380413135683813, 28.131091753475115, 6.602642765227231, 5.970185972626102, 2.7570147661702795, 2.240480314704063, 23.828594967329675, 31.50107440235915, 13.924459964385894, 17.366787674130563, 19.148551960186687, 7.1041938428827756, 10.510821424669768, 10.085396426587229, 20.014886740142167, 19.54206824363306, 17.472850680124488, 3.312250279251943, 4.436786927368903, 4.643607360492023, 4.918993557084549, 18.881554943816823, 18.975833299903673, 2.294855688111894, 27.600739196888675, 27.37557319030758, 1.460852734023256, 2.1967933466114777, 5.1555411736782935, 18.551398044165204, 19.448085286840413, 25.427515012886126, 2.5800968278588634, 1.7833543877051976, 8.768626365794242, 1.7085443631344015, 27.81551998924302, 24.994107398981782, 25.400635173066934, 26.368030543088484, 26.444090647162238, 0.7015608617978326, 0.5814945146589732, 24.11808111164012, 26.92171537575274, 26.879111042587454, 3.6545015745183096, 1.4038522217991491, 7.080730468056177, 5.067934845232474, 24.380392270777204, 23.420974619106698, 27.168959353842375, 1.7135130468856237, 1.7924718815492742, 7.073501843355114, 7.13759995266279, 18.99982439676363, 18.370587168841666, 18.07426158159635, 8.680092016062602, 10.992523594443115, 1.4209382356736358, 6.20269712476931, 13.206271858985511, 8.171387394143194, 4.880447823821549, 6.325843827777947, 11.171045371080472, 22.45796024760318, 7.2186773805414735, 1.3282343123483273, 8.015501875242096, 10.467722961175259, 7.654107610955654, 12.579307682687746, 2.7136596762013374, 8.075997300065174, 8.013523866124249, 11.284984206872005, 6.98729680252555, 4.469798436046547, 12.235127616018996, 11.185677821756691, 6.070727807045664, 6.801890391545882, 6.3981550544877885, 5.093647396759224, 2.186737631712388, 10.10408275025375, 1.1463146446339763, 11.91645896609035, 15.149991374434329, 10.487611613242734, 3.4960337026045947, 1.105003857441163, 8.163296264886329, 15.236894741668491, 3.4176618389497047, 11.881326953981397, 3.3293587921206083, 7.198973519449561, 7.784077097960441]
print(np.sort(a) )
print('Percentile',np.percentile(a, 20))

plt.show()

