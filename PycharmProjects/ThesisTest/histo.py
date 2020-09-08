import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import face, ascent
from PIL import Image
import cv2
import matplotlib.mlab as mlab
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def color_transfer(source, target):
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

    # This is a better choice.  It seems to match template better.
    l = (lStdSrc / lStdTar) * l
    a = (aStdSrc / aStdTar) * a
    b = (bStdSrc / bStdTar) * b

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



def hist_match(source, template):
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
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


#jpgfile = Image.open("picture.jpg")
#source = face()
#template = ascent()
source = cv2.imread('/Users/hengsun/GitHub/StyleTransfer/img/city-pexels-photo-167200.jpg')
template = cv2.imread('/Users/hengsun/GitHub/StyleTransfer/img/baskelball-surface-pexels-photo-207300.jpg')

# Convert channels to RGB.
source = source[:, :, ::-1]
template = template[:, :, ::-1]


#matched = hist_match(source, template)
#matched = np.copy(source)

def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

matched = color_transfer(template, source)  # Works really well.

# https://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
def plotHisto(image, subplot):
    chans = cv2.split(image)
    colors = ("r", "g", "b")
    # plt.figure()
    subplot.set_title("'Flattened' Color Histogram")
    # subplot.xlabel("Bins")
    # subplot.ylabel("# of Pixels")
    subplot.set_xlabel("Bins")
    subplot.set_ylabel("# of Pixels")

    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # plot the histogram
        subplot.plot(hist, color=color)
        subplot.set_xlim([0, 256])
        subplot.set_ylim([0, 10000])

#
# May want to look at this: https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py
#
# Review the paper and see what it requires.

if __name__ == "__main__":
    # This is a working version of the histogram.
    if True:
        x1, y1 = ecdf(source.ravel())
        x2, y2 = ecdf(template.ravel())
        x3, y3 = ecdf(matched.ravel())

        fig = plt.figure()

        #https: // nickcharlton.net / posts / drawing - animating - shapes - matplotlib.html
        canvas = FigureCanvas(fig)

        gs = plt.GridSpec(4, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1, :])
        for aa in (ax1, ax2, ax3):
            aa.set_axis_off()

        ax1.imshow(source)
        ax1.set_title('Source')
        ax2.imshow(template)
        ax2.set_title('template')
        ax3.imshow(matched)
        ax3.set_title('Matched')

        ax4.plot(x1, y1 * 100, '-r', lw=3, label='Source')
        ax4.plot(x2, y2 * 100, '-k', lw=3, label='Template')
        ax4.plot(x3, y3 * 100, '--r', lw=3, label='Matched')
        ax4.set_xlim(x1[0], x1[-1])
        ax4.set_xlabel('Pixel value')
        ax4.set_ylabel('Cumulative %')
        ax4.legend(loc=5)

        # fig = plt.figure()
        # gs = plt.GridSpec(2, 3)
        hx1 = fig.add_subplot(gs[2, 0])
        hx2 = fig.add_subplot(gs[2, 1], sharex=hx1, sharey=hx1)
        hx3 = fig.add_subplot(gs[2, 2], sharex=hx1, sharey=hx1)
        hx4 = fig.add_subplot(gs[3, :])

        num_bins = 16
        mu=100
        sigma=15
        #source = mu + sigma * np.random.randn(10000)
        # ax1n, ax1bins, ax1patches = hx1.hist(source[::,0], num_bins, normed=1, facecolor='red', alpha=0.5)
        # ax2n, ax2bins, ax2patches = hx1.hist(template[::,0], num_bins, normed=1, facecolor='green', alpha=0.5)
        # ax3n, ax3bins, ax3patches = hx1.hist(matched[::,0]+4, num_bins, normed=1, facecolor='blue', alpha=0.5)

        # ax1n, ax1bins, ax1patches = hx2.hist(source[::,1], num_bins, normed=1, facecolor='green', alpha=0.5)
        # ax2n, ax2bins, ax2patches = hx2.hist(source[::,1], num_bins, normed=1, facecolor='red', alpha=0.5)
        # ax3n, ax3bins, ax3patches = hx2.hist(matched[::,1], num_bins, normed=1, facecolor='green', alpha=0.5)

        # ax1n, ax1bins, ax1patches = hx3.hist(source[::,2]+1, num_bins, normed=1, facecolor='blue', alpha=0.5)
        #ax2n, ax2bins, ax2patches = hx3.hist(template[::,0]+1, num_bins, normed=1, facecolor='blue', alpha=0.5)
        # ax3n, ax3bins, ax3patches = hx3.hist(source[::,2], num_bins, normed=1, facecolor='red', alpha=0.5)


        # add a 'best fit' line
        # ax1y = mlab.normpdf(ax1bins, mu, sigma)
        # ax2y = mlab.normpdf(ax2bins, mu, sigma)
        # ax3y = mlab.normpdf(ax3bins, mu, sigma)

        #ax1.plot(ax1bins, ax1y, 'r--')
        #ax2.plot(ax2bins, ax2y, 'r--')
        #ax3.plot(ax3bins, ax3y, 'r--')
        # print(np.shape(ax1bins))
        # print(np.shape(ax1n))
        #
        # hx4.plot( ax1bins[1:], ax1n[0], '-r', lw=3)
        # hx4.plot(ax1bins[1:], ax1n[1], '-g', lw=3)
        # hx4.plot(ax1bins[1:], ax1n[2], '-b', lw=3)


        # hx4.plot(ax1bins[1:], ax2n, '-g', lw=3)
        # hx4.plot(ax1bins[1:], ax3n, '-b', lw=3)
        # ax4.plot(ax2bins, ax2y, 'g--', secondary=True)
        # ax4.plot(ax3bins, ax3y, 'b--',secondary=True)

        #plt.plot(bins, n, kind='bar')

        #ax1.xlabel('Smarts')
        #ax1.ylabel('Probability')

        # Tweak spacing to prevent clipping of ylabel
        #plt.subplots_adjust(left=0.15)

        plotHisto(source, hx1)
        plotHisto(template, hx2)
        plotHisto(matched, hx3)

        #https: // stackoverflow.com / questions / 35355930 / matplotlib - figure - to - image -as-a - numpy - array
        canvas.draw()
        imageArray = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        print(imageArray.shape)
        print(source.flatten().shape)
        #ax1.imshow(np.reshape(imageArray, source.shape))
    # print("flattened feature vector size: %d" % (np.array(features).flatten().shape))
    plt.show()
