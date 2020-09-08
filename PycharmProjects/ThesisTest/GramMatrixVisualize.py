import numpy as np
import numpy
from PIL import Image, ImageDraw
import matplotlib
from keras.preprocessing.image import img_to_array
import cv2
import math

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = numpy.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll(buf, 3, axis=2)
    return buf





def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    #return Image.fromstring("RGBA", (w, h), buf.tostring())
    return Image.frombytes("RGBA", (w, h), buf.tostring())

def createCircleArray(w, h):
    image = Image.new('RGB', (w, h))
    draw = ImageDraw.Draw(image)
    draw.ellipse(( int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)), fill='white', outline='white')
    # draw.point((100, 100), 'red')
    imageArray = img_to_array(image).astype('uint8')
    # Convert the image to binary map: https://stackoverflow.com/questions/28430904/set-numpy-array-elements-to-zero-if-they-are-below-a-specific-threshold
    indeces = imageArray > 0
    imageArray[indeces] = 1
    print(imageArray)

    return imageArray


def createGram(w,h, blurValue):
    x = np.zeros((w,h))
    # y = np.zeros((6,6))
    #image = 'img/butterflies_pexels-photo-326055_scaled.jpg'
    image = 'img/baskelball-surface-pexels-photo-207300.jpg'
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Canny edge detection: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html

    if True: # Apply Canny edge finder.
        edges = cv2.Canny(gray, 100, 200, apertureSize=3, L2gradient=True)
    else:
        edges = gray

    if False: # Apply Guassian Blur.
        blur = cv2.GaussianBlur(edges, (blurValue,blurValue), 0)
    else:
        blur = edges

    if False:
        laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    else:
        laplacian = blur
    # edges = cv2.Canny(laplacian, 50, 150, apertureSize=3, L2gradient=True)
    x = img_to_array(laplacian).astype('uint8')
    x = np.squeeze(x,axis=2)
    y = np.copy(x)
    b = np.copy(x)
    if False:
        for i in range(0,3):
            # y = np.copy(x)
            for j in range(i,3):
                y[i][j]=64
                # print('X')
                # print(y)
                # z=np.dot(y, y.T)
                # print('Gram')
                # print(z)

        for i in range(0,3):
            # y = np.copy(x)
            for j in range(0,i):
                b[i][j]=64
                # print('X')
                # print(y)
                # c=np.dot(b, b.T)
                # print('Gram')
                # print(c)
    # else:
    #     y[0][1]=1
    #     b[1][1]=1
    #     b[1][2]=1

    originalShape = y.shape
    print(originalShape)
    z=np.dot(y, y.T)
    print(z)

    square = math.sqrt(z.size)
    z.reshape((square,square))
    print('Y')
    print(y)
    print('First Gram')
    print(z)
    # image = Image.fromarray(z)
    cv2.imshow("First Image " + str(blurValue), y)
    cv2.imshow("First Gram " + str(blurValue) , z)

    # print('B')
    # print(b)
    # print('Second Gram')
    # c=np.dot(b, b.T)
    # print(c)
    #
    # cv2.imshow("Second Image", b)
    # cv2.imshow("Second Gram", c)

    # Stacking images: http://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/
    # firstImage=np.vstack((y,z))
    # secondImage=np.vstack((b,c))
    # image = np.hstack((firstImage,secondImage))
    # cv2.imshow("First", image)

    # print('First difference')
    # print((z-c)**2)
    #
    # print('Second difference')
    # print((c-z)**2)
    return



if __name__ == "__main__":
    # https://nickcharlton.net/posts/drawing-animating-shapes-matplotlib.html
    if True:
        # https://stackoverflow.com/questions/20747345/python-pil-draw-circlePy
        # image = Image.new('RGBA', (200, 200))
        # draw = ImageDraw.Draw(image)
        # draw.ellipse((20, 20, 180, 180), fill='blue', outline='blue')
        # #draw.point((100, 100), 'red')

        for i in range(3):
            imageArray = createGram(3,3, int(i*2+1))
        # imageArray = createCircleArray(10,10)

        # Convert the image to binary map: https://stackoverflow.com/questions/28430904/set-numpy-array-elements-to-zero-if-they-are-below-a-specific-threshold

        # print(imageArray)
        # print(imageArray.shape)
        # plt.show()
        cv2.waitKey(0)







