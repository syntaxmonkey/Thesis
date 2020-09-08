'''Neural style transfer with Keras.

Run the script with:
```
python neural_style_transfer.py path_to_your_base_image.jpg path_to_your_reference.jpg prefix_for_results
```
e.g.:
```
python neural_style_transfer.py img/tuebingen.jpg img/starry_night.jpg results/my_result
```
Optional parameters:
```
--iter, To specify the number of iterations the style transfer takes place (Default is 10)
--content_weight, The weight given to the content loss (Default is 0.025)
--style_weight, The weight given to the style loss (Default is 1.0)
--tv_weight, The weight given to the total variation loss (Default is 1.0)
```

It is preferable to run this script on GPU, for speed.

Example result: https://twitter.com/fchollet/status/686631033085677568

# Details

Style transfer consists in generating an image
with the same "content" as a base image, but with the
"style" of a different picture (typically artistic).

This is achieved through the optimization of a loss function
that has 3 components: "style loss", "content loss",
and "total variation loss":

- The total variation loss imposes local spatial continuity between
the pixels of the combination image, giving it visual coherence.

- The style loss is where the deep learning keeps in --that one is defined
using a deep convolutional neural network. Precisely, it consists in a sum of
L2 distances between the Gram matrices of the representations of
the base image and the style reference image, extracted from
different layers of a convnet (trained on ImageNet). The general idea
is to capture color/texture information at different spatial
scales (fairly large scales --defined by the depth of the layer considered).

 - The content loss is a L2 distance between the features of the base
image (extracted from a deep layer) and the features of the combination image,
keeping the generated image close enough to the original one.

# References
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
'''

from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import os # HSC
from keras.utils.training_utils import multi_gpu_model # HSC
import tensorflow as tf # HSC
import os

from keras.applications import vgg19
from keras import backend as K
from skimage import data
from skimage.color import convert_colorspace
import math

import cv2

from createMask import createSquareGradient

sourceColour = 'RGB'
#targetColour = 'HSV'
targetColour = 'YIQ'

from matplotlib import pyplot as plt
from scipy.misc import face, ascent

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')

parser.add_argument('input_folder', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('input_images', metavar='ref', type=str,
                    help='Comma separated list of images.')
parser.add_argument('output_folder', metavar='ref', type=str,
                    help='Output path for this run.')



# parser.add_argument('base_image_path', metavar='base', type=str,
#                     help='Path to the image to transform.')
#                     
# parser.add_argument('style_reference_image_path', metavar='ref', type=str,
#                     help='Path to the style reference image.')
#                     
# parser.add_argument('result_prefix', metavar='res_prefix', type=str,
#                     help='Prefix for the saved results.')


parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')

parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')
parser.add_argument("-g", "--gpus", type=int, default=2, help="# of GPUs to use for training") # HSC








args = parser.parse_args()
G = vars(args)["gpus"] # HSC


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# colour histogram transfer: https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/

def image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())    
    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def color_transfer(source, target):
    # Will apply the source histogram to the target image.
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
    
    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar
    
    # scale by the standard deviations
    # l = (lStdTar / lStdSrc) * l
    # a = (aStdTar / aStdSrc) * a
    # b = (bStdTar / bStdSrc) * b

    # This is the actually the correc way to scale. HSC.
    l = (lStdSrc / (lStdTar+1e-06)) * l
    a = (aStdSrc / (aStdTar+1e-06)) * a
    b = (bStdSrc / (bStdTar+1e-06)) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc
    
    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    
    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    
    # return the color transferred image
    return transfer



def hist_match(sourceimg, templateimg):
    global img_nrows, img_ncols
    sourceImage = load_img(sourceimg, target_size=(img_nrows, img_ncols))
    templateImage = load_img(templateimg, target_size=(img_nrows, img_ncols))
    sourceImage = img_to_array(sourceImage)
    templateImage = img_to_array(templateImage)
    # ** Really important.  For some reason if we don't clip it to unsigned, the histogram transformation fails.
    source = np.clip(sourceImage, 0, 255).astype('uint8') 
    template = np.clip(templateImage, 0, 255).astype('uint8')  
        
    img = color_transfer(template, source)
    img = img_to_array(img)
    #img = randomizeImage(img) # HSC    
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img



# Histogram matching: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x


def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

def test_histo(sourceimg, templateimg):
    global img_nrows, img_ncols
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """
    print(sourceimg)
    width, height = load_img(sourceimg, grayscale=True).size
    img_nrows = 400 # HSC - NOTE: This is actually the height of the image, not the width.
    img_ncols = int(width * img_nrows / height)

   ##   print(sourceimg)
    print(img_nrows, img_ncols)
    sourceImage = load_img(sourceimg, target_size=(img_nrows, img_ncols))
    templateImage = load_img(templateimg, target_size=(img_nrows, img_ncols))
    sourceImage = img_to_array(sourceImage)
    templateImage = img_to_array(templateImage)
    # ** Really important.  For some reason if we don't clip it to unsigned, the histogram transformation fails.
    source = np.clip(sourceImage, 0, 255).astype('uint8') 
    template = np.clip(templateImage, 0, 255).astype('uint8')

   ##   
    # if True:
    #     source = sourceImage
    #     template = templateImage
    # else:
    #     source = cv2.imread(sourceimg)
    #     template = cv2.imread(templateimg)
    #     # # 'BGR'->'RGB'
    #     source = source[:, :, ::-1]
    #     template = template[:, :, ::-1]


  ###     
    # 
    # print(np.sum(sourceImage - source))
    # print(np.shape(sourceImage))    
    # print(np.shape(source))
    # print(type(sourceImage))
    # print(type(source))
    # print(sourceImage)
    # print('-----')
    # print(source)


    # 
    newImage = color_transfer(template, source)

    #####
    if False:
        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()
    
        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)
    
        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
    
        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values*2.0)
    
        newImage = interp_t_values[bin_idx].reshape(oldshape)
    ####
    
    #newImage = img_to_array(newImage)
    #img = np.expand_dims(newImage, axis=0)
    #img = vgg19.preprocess_input(img)
    
    fig = plt.figure()
    gs = plt.GridSpec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, :])
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    x1, y1 = ecdf(source.ravel())
    x2, y2 = ecdf(template.ravel())
    x3, y3 = ecdf(newImage.ravel())
    
    source = np.clip(source, 0, 255).astype('uint8') 
    template = np.clip(template, 0, 255).astype('uint8')
    newImage = np.clip(newImage, 0, 255).astype('uint8') 
    
        
    ax1.imshow(source)
    ax1.set_title('Source')
    ax2.imshow(template)
    ax2.set_title('template')
    ax3.imshow(newImage)
    ax3.set_title('Matched')
    
    ax4.plot(x1, y1 * 100, '-r', lw=3, label='Source')
    ax4.plot(x2, y2 * 100, '-k', lw=3, label='Template')
    ax4.plot(x3, y3 * 100, '--r', lw=3, label='Matched')
    ax4.set_xlim(x1[0], x1[-1])
    ax4.set_xlabel('Pixel value')
    ax4.set_ylabel('Cumulative %')
    ax4.legend(loc=5)
    plt.show()
    
    return newImage



# util function to open, resize and format pictures into appropriate tensors


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    #img = convert_colorspace(img, sourceColour, targetColour)
    img = img_to_array(img)
    #img = randomizeImage(img) # HSC    
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# util function to open, resize and format pictures into appropriate tensors


def preprocess_image_gray(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols)) #, grayscale=True)
    originalshape = np.shape(img)
    #print("Before shape:", np.shape(img))
    img = img_to_array(img)    
    
    img = rgb2gray(img) # Convert to grayscale.
    img = np.repeat( np.reshape(img, (img_nrows*img_ncols)), 3)
    img = np.reshape(img, (img_nrows,img_ncols,3))  # Expand the image to duplicate pixels.
    #print("After shape:", np.shape(img))    

    #img = randomizeImage(img) # HSC        
    img = np.expand_dims(img, axis=0)
    
    #print(np.shape(img))
    img = vgg19.preprocess_input(img)
    return img

# util function to convert a tensor into a valid image


def deprocess_image(x):
    #x = randomizeImage(x) # HSC
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# HSC
def randomizeImage(x):
    ''' This method will perturb all values by some random integer amount '''
    print("Randomizing")
    image_shape=np.shape(x)
    print(image_shape)
    randomValues = np.random.random_integers( -9, 9, np.prod(image_shape) )
    randomValues = np.reshape(randomValues, image_shape)
    x = x*0.9 + randomValues
    return x

# HSC
def randomImage(x):
    ''' This method will perturb all values by some random integer amount '''
    print("Random Image")
    image_shape=np.shape(x)
    print(image_shape)
    randomValues = np.random.random_integers( -8, 8, np.prod(image_shape) )
    randomValues = np.reshape(randomValues, image_shape)
    x = randomValues
    return x



# HSC
def randomImageCount(x, count):
    ''' This method will zero out a random number of values defined by count. '''
    print("RandomImageCount")
    image_shape=np.shape(x)
    randomValues = np.random.rand( np.prod(image_shape) )
    sortedValues = np.sort(randomValues)
    keepMask = randomValues < sortedValues[ -count ]     
    keepMask = np.reshape(keepMask, image_shape)
    x = x * keepMask
    return x




# Make multiple gpu.
#model = multi_gpu_model(model, gpus=G) # HSC    
#print(outputs_dict)

# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
# To introduce a mask, can we simply find the gram matrix of our mask and use it as a multiplier?
# Mask = gram_matrix(mask_image)
# K.sum(K.square(Mask*(S - C))) / (4. * (channels ** 2) * (size ** 2))
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))



def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        grad_values = np.multiply(grad_values, maskArray.flatten()) # HSC
        return grad_values


def sizeInit(base_image_path):
    global img_nrows, img_ncols, maskArray
    
    # dimensions of the generated picture.
    width, height = load_img(base_image_path, grayscale=True).size
    print(width,height)
    img_nrows = 400 # HSC - NOTE: This is actually the height of the image, not the width.
    img_ncols = int(width * img_nrows / height)
    maskArray = createSquareGradient(img_ncols, img_nrows, horizontal=False)
    maskArray = np.expand_dims(maskArray, axis=0)
    

def processInstance(args, base_image_path, style_reference_image_path, result_prefix, content_weight, style_weight, styleLayers):
    global img_nrows, img_ncols, f_outputs, maskArray
    
    # BETA
    # get tensor representations of our images
    
    
    #base_image = K.variable( preprocess_image(base_image_path))
    #base_image = K.variable( randomizeImage(preprocess_image(base_image_path)) ) #HSC
    #base_image = K.variable( randomImage(preprocess_image(base_image_path)) ) #HSC
    
    

    
    realCount=0
    
    #x = preprocess_image(base_image_path)
    #x = randomizeImage(preprocess_image(base_image_path))
    #x = randomImage(preprocess_image(base_image_path))
    #print("base image shape: ", np.shape(x))
    
    
    #styleLayers = "E"
    
    
    for style_image_path in style_reference_image_path.split(","): # HSC
        # ALPHA
        #style_reference_image_path = args.style_reference_image_path
        style_reference_image_path = style_image_path # HSC
        base_image = K.variable( hist_match(base_image_path, style_reference_image_path))
        x = hist_match(base_image_path, style_reference_image_path)
        #result_prefix = args.result_prefix
        iterations = args.iter
        
        print('Applying Style ', style_image_path)
        
        # # these are the weights of the different loss components
        # total_variation_weight = args.tv_weight
        # style_weight = args.style_weight
        # content_weight = args.content_weight
        
        # # dimensions of the generated picture.
        # width, height = load_img(base_image_path).size
        # img_nrows = 400 # HSC - NOTE: This is actually the height of the image, not the width.
        # img_ncols = int(width * img_nrows / height)
        
        # BETA
        # get tensor representations of our images
        #base_image = K.variable( preprocess_image(base_image_path))
        #base_image = K.variable( randomizeImage(preprocess_image(base_image_path)) ) #HSC
        
        style_reference_image = K.variable(preprocess_image(style_reference_image_path))
        print("style image shape:", np.shape(style_reference_image))
        
        # this will contain our generated image
        # if K.image_data_format() == 'channels_first':
        #     combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
        # else:
        #     combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

        # this will contain our generated image
        if K.image_data_format() == 'channels_first':
            combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
        else:
            combination_image = K.placeholder((1, img_nrows, img_ncols, 3))
        
        # combine the 3 images into a single Keras tensor
        input_tensor = K.concatenate([base_image,
                                    style_reference_image,
                                    combination_image], axis=0)
        
        # build the VGG16 network with our 3 images as input
        # the model will be loaded with pre-trained ImageNet weights
        #model = vgg19.VGG19(input_tensor=input_tensor,
        #                    weights='imagenet', include_top=False)
        
        
        model = vgg19.VGG19(input_tensor=input_tensor,
            weights='imagenet', include_top=False)
                    
        print('Model loaded.')
        
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
        
        # GAMMA
        # combine these loss functions into a single scalar
        loss = K.variable(0.)
        layer_features = outputs_dict['block5_conv2'] # HSC - Should this be block4_conv2?
        #layer_features = outputs_dict['block4_conv2'] # Starting with Results6 onward.
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss += content_weight * content_loss(base_image_features,
                                            combination_features)
        
        if styleLayers=="A":
            feature_layers = ['block1_conv1'] # A
        elif styleLayers=="B":
            feature_layers = ['block1_conv1', 'block2_conv1'] # B
        elif styleLayers=="C":
            feature_layers = ['block1_conv1', 'block2_conv1','block3_conv1'] # C
        elif styleLayers=="D":
            feature_layers = ['block1_conv1', 'block2_conv1','block3_conv1', 'block4_conv1'] # D
        elif styleLayers=="F":
        # Create some other combinations.
            feature_layers = ['block1_conv1','block3_conv1', 'block5_conv1'] # F
        elif styleLayers=="G":
            feature_layers = ['block2_conv1','block4_conv1'] # G
        elif styleLayers=="H":
            feature_layers = ['block2_conv1'] # H
        elif styleLayers=="I":
            feature_layers = ['block3_conv1'] # I
        elif styleLayers=="J":
            feature_layers = ['block4_conv1'] # J
        elif styleLayers=="K":
            feature_layers = ['block5_conv1'] # K                
        elif styleLayers=="L":
            feature_layers = ['block3_conv1', 'block4_conv1'] # L                
        elif styleLayers=="M":
            feature_layers = ['block4_conv1','block5_conv1'] # M                            
        else:
            feature_layers = ['block1_conv1', 'block2_conv1','block3_conv1', 'block4_conv1', 'block5_conv1'] # E
        
        
        # STYLE LOSS CALCULATION
        # DELTA
        # HSC - construct loss calculation.  Could potentially change the weights for each layer.
        for layer_name in feature_layers:
            layer_features = outputs_dict[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, combination_features)
            loss += (style_weight / len(feature_layers)) * sl
        loss += total_variation_weight * total_variation_loss(combination_image)
        
        # get the gradients of the generated image wrt the loss
        grads = K.gradients(loss, combination_image)
        
        print('Shape of grads:', np.shape(grads))
        print(grads)
        print('Shape of maskArray', np.shape(maskArray))
        # Apply maks to grads.
        #grads = np.multiply(grads,maskArray)
        
        outputs = [loss]
        if isinstance(grads, (list, tuple)):
            outputs += grads
        else:
            outputs.append(grads)
        
        f_outputs = K.function([combination_image], outputs)
        
        
        # EPSILON
        evaluator = Evaluator()
        
        # run scipy-based optimization (L-BFGS) over the pixels of the generated image
        # so as to minimize the neural style loss
        #x = preprocess_image(base_image_path)
        save_interval = 1
        
        previousError = 1000.0**10 # HSCs
        
        for i in range(iterations):
            print('Start of iteration', i)
            start_time = time.time()
        #    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
        #                                     fprime=evaluator.grads, maxfun=20)
            
            #steps = int(2.5**i) # HSC
            steps = int(8+1.1*math.exp(1.4*i)) # Make the steps more sensible: 6, 9, 23, 78, 302.  Totals 418.
            #print('Step size:', steps)
            x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                            fprime=evaluator.grads, maxfun=steps)                                     
        
            lossDifferential = (previousError - min_val)
            #print('Current loss value:', min_val)
            print('Current loss value:', min_val, '   ------    Loss Change: ', lossDifferential / previousError) # HSC
            
            # save current generated image    
            #if (i%save_interval) == 0 :
            img = deprocess_image(x.copy())
            #fname = result_prefix + '_at_iteration_%d.png' % i
            fname = result_prefix + '_at_iteration_%d.png' % realCount
            
            img_rgb = img
            #img_rgb = convert_colorspace(img, targetColour,sourceColour)
            imsave(fname, img_rgb)
            print('Image saved as', fname)    
            end_time = time.time()
            print('Iteration %d completed in %ds' % (i, end_time - start_time))
            previousError = min_val # HSC
            realCount += 1
            
    
        #print(np.shape(x))
        #print(np.shape(base_image))
        #base_image=np.reshape(np.copy(x), np.shape(base_image))
        #styleLayers="K" #
        #style_weight = style_weight / 10.0 # Try to reduce the style weight each time we go through
    
    
    fnamePieces = fname.split("/")
    fnamePieces.insert(2,"final")
    finalPath = "/".join(fnamePieces[:-1])
    finalName = fnamePieces[-1]
    
    if not os.path.exists(finalPath): # HSC
        os.mkdir(finalPath) # HSCs
    
    imsave(finalPath + "/" + finalName, img_rgb) # HSC
    K.clear_session()



###################


######################



# these are the weights of the different loss components
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

#styleLayers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
styleLayers = ['E', 'H', 'I', 'J']
#styleLayers = ['E']
#C_values = [content_weight, content_weight*2, content_weight/2]
C_values = [content_weight]
#S_values = [style_weight, style_weight*10, style_weight/10]
S_values = [style_weight]
#S_values = [style_weight, style_weight*10]
input_folder = args.input_folder
output_folder = args.output_folder
input_image_portion = args.input_images

input_images = [
    'animal-blue-pattern-danger-41180_scaled.jpg',
    #'baskelball-surface-pexels-photo-207300.jpg',
    #'bubbles_pexels-photo-459388_scaled.jpg',
    'butterflies_pexels-photo-326055_scaled.jpg',
    'city-pexels-photo-167200.jpg',
    #'city-pexels-photo-421927.jpg',
    'fire-small-pexels-photo-266604.jpg',
    'meeting_pexels-photo-416405_scaled.jpg',
    'metal-diamond-pexels-photo-242616.jpg',
    'money-close-pexels-photo-251287.png',
    #'mushrooms_pexels-photo-382040_scaled.jpg',
    #'nightimage_pexels-photo-57901_scaled.jpg',
    #'paper-crumpled-pexels-photo-220634.jpg',
    #'rocks-grey-pexels-photo-153493.jpg',
    'scales-gold-pexels-photo-260286.jpg',
    'sea-of-fog-fog-ocean-sea-52514.jpg',
    'shoes_pexels-photo-267320_scaled.jpg',
    'stained-glass-pexels-photo-208414.jpg',
    'stained-glass-spiral-circle-pattern-161154.jpg',
    'starry_night_scaled.jpg',
    'texture-designer-grey-fur.jpg',
    'vegetablesfruit_pexels-photo-264537_scaled.jpg',
    #'wood-close-pexels-photo-172289.jpg',
    'wood-cord-pexels-photo-128639.jpg'
]



style_images = [
    'baskelball-surface-pexels-photo-207300.jpg',
    'fire-small-pexels-photo-266604.jpg',
    'metal-diamond-pexels-photo-242616.jpg',
    'money-close-pexels-photo-251287.png',
    'rocks-grey-pexels-photo-153493.jpg',
    'scales-gold-pexels-photo-260286.jpg',
    'stained-glass-pexels-photo-208414.jpg',
    #'stained-glass-spiral-circle-pattern-161154.jpg',
    'starry_night_scaled.jpg',
    'texture-designer-grey-fur.jpg',
    'vegetablesfruit_pexels-photo-264537_scaled.jpg',
    'wood-cord-pexels-photo-128639.jpg',
    'dome-gold-arabic-cathedral-161404.jpg',
    'concrete_stone_pexels-photo-507892.jpg',
    'nail_heads_pexels-photo-210438.jpg',
    'yarn-wool-cords-colorful-67613.jpg'
]


# style_images = [
#      'dome-gold-arabic-cathedral-161404.jpg',
#      'concrete_stone_pexels-photo-507892.jpg',
#      'nail_heads_pexels-photo-210438.jpg',
#      'yarn-wool-cords-colorful-67613.jpg'
# ]

if input_image_portion == "front":
    input_images = input_images[ : int(len(input_images)/2) ]
else:
    input_images = input_images[ int(len(input_images)/2) : ]


output_folder = output_folder + "_" + os.path.basename(__file__)

if not os.path.exists(output_folder): # HSC
    os.mkdir(output_folder) # HSCs

print(args)

if True:
    for input_image in input_images:
        for style_image in style_images:        
            if input_image != style_image:
                for content_weight in C_values:
                    for styleLayer in styleLayers:                
                        for style_weight in S_values:
                            
                            image_output_folder = output_folder + "/" + input_image 
                            if not os.path.exists(image_output_folder): # HSC
                                os.mkdir(image_output_folder) # HSCs                                                
                            
                            result_prefix = image_output_folder + "/" + input_image + "_C_" + str(content_weight) + "_S_" + str(style_weight) + "_L_" + styleLayer + "_" + style_image
                            baseImage = input_folder + '/' + input_image
                            styleImage = input_folder + '/' + style_image
                            sizeInit(baseImage)
    
                            #test_histo(baseImage, styleImage)
                            
                            processInstance(args, baseImage, styleImage, result_prefix, content_weight, style_weight, styleLayer)
else:
    # Create binary mask.
    print("Testing Circle Array")    
    #imageArray = createCircleArray(10,10)
    #print(imageArray)
    baseImage = input_folder + '/' + input_images[0]
    sizeInit(baseImage)
    #print(maskArray)    
    print(maskArray.shape)

