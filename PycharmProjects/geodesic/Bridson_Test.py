# # https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.structure_tensor
#
# # Tensor Structure testg
#
# from skimage.feature import structure_tensor
# import numpy as np
#
# square = np.zeros((5, 5))
# square[2, 2] = 1
#
# Arr, Arc, Acc = structure_tensor(square, sigma=0.1)
#
#
# print(square)
# print(Arr)
# print(Arc)
# print(Acc)
#
# # array([[0., 0., 0., 0., 0.],
# #        [0., 1., 0., 1., 0.],
# #        [0., 4., 0., 4., 0.],
# #        [0., 1., 0., 1., 0.],
# #        [0., 0., 0., 0., 0.]])
import math

intensityValues = [255, 200, 128, 100, 50, 30, 10, 1,0]
for intensity in intensityValues:
	print("Intensity:", intensity, "produces", math.exp(intensity/50))

