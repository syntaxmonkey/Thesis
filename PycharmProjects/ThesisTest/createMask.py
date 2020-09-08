import numpy as np
import numpy
from PIL import Image, ImageDraw
import matplotlib
from keras.preprocessing.image import img_to_array

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


if __name__ == "__main__":
    # https://nickcharlton.net/posts/drawing-animating-shapes-matplotlib.html
    if False:
        plt.axes()
        fig = plt.figure()
        circle = plt.Circle((0, 0), radius=0.75, fc='y')
        plt.gca().add_patch(circle)


        data = fig2data(fig)
        #im = fig2img(fig)

        print(data)
        print(data.shape)
        #plt.axis('scaled')
        #plt.show()

        #canvas = FigureCanvas(fig)
        #canvas.draw()
    else:
        # https://stackoverflow.com/questions/20747345/python-pil-draw-circlePy
        # image = Image.new('RGBA', (200, 200))
        # draw = ImageDraw.Draw(image)
        # draw.ellipse((20, 20, 180, 180), fill='blue', outline='blue')
        # #draw.point((100, 100), 'red')
        imageArray = createCircleArray(10,10)


        # Convert the image to binary map: https://stackoverflow.com/questions/28430904/set-numpy-array-elements-to-zero-if-they-are-below-a-specific-threshold
        image = Image.fromarray(imageArray)

        plt.imshow(image)

        # print(imageArray)
        # print(imageArray.shape)
        plt.show()







