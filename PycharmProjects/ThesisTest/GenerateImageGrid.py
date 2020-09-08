import os
import re



# https://www.w3schools.com/cSS/tryit.asp?filename=trycss_image_gallery

# Zooming on hover: https://www.w3schools.com/howto/howto_css_zoom_hover.asp

path = "/Users/hengsun/GitHub/StyleTransfer/results110_ColourHisto_SquareMaskGrad_NST_J.py"
BaseImagePath = "../../../img/"

#print(files)

# Need to sort the files by style image.

def generateHTML(root, files):
    # First get unique list of style images.
    style_images = []
    layers = []
    files.sort()
    for filename in files:
        #print(filename.split("_"))
        # Expected format: ['paper-crumpled-pexels-photo-220634.jpg', 'C', '1.0', 'S', '100.0', 'L', 'I', 'city-pexels-photo-167200.jpg', 'at', 'iteration', '6.png']
        #pieces = filename.split("_")
        found = re.search("S_[0-9\.]+_L_([A-K])_([^.]+\..+)_at_iteration_.+\.png", filename)
        if found:
            #print(pieces)
            layers.append(found.group(1)) # The layer name.
            style_images.append(found.group(2)) # The style_image name.


    style_images = list(set(style_images))  # Make the list of style_images unique.
    style_images.sort()
    layers = list(set(layers))  # Make the list of layers unique.
    layers.sort()

    #print(layers)
    #print(style_images)
    #print(len(style_images))


    # Reconstruct the image order based on the matrix.
    #                 layer A   layer B   layer C
    # style 1
    # style 2
    # style 3
    #
    #
    w, h = len(layers), len(style_images)
    image_matrix = [[x for x in range(w)] for y in range(h)]

    #print("files:", files)
    for filename in files:
        #print("filename:", filename)
        found = re.search("S_[0-9\.]+_L_([A-K])_([^.]+\..+)_at_iteration_.+\.png", filename)
        if found:
            #print(pieces)
            baseImageFind = re.search("(.+\.jpg|.+\.png)_C_[0-9\.]+", filename)
            layer = found.group(1) # The layer name.
            style = found.group(2) # The style_image name.
            image_matrix[style_images.index(style)][layers.index(layer)] = filename
            base_image = baseImageFind.group(1)

    ncolumns = len(layers)
    nrows = int(len(files) / ncolumns)


    htmlText = '''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    div.gallery {
        margin: 20px;
        border: 1px solid #ccc;
        float: left;
        width: 180px;
    }

    
    div.gallery:hover {
        border: 1px solid #777;
    
    }
    
    div.gallery img {
        width: 100%;
        height: auto;
    }
    
    div.gallery img:hover {
        width: 100%;
        height: auto;
        transform: scale(2);    
    }
    
    div.desc {
        padding: 15px;
        text-align: center;
    }
    </style>
    </head>
    <body>
    '''

    htmlText += '''
    <table>
    '''

    # Insert empty column.
    htmlText += '''
    <th>BaseImage:<a target="_blank" href="''' + BaseImagePath+base_image + '''">''' + base_image + '''</th>
    
    '''

    # Insert header columns.
    for i in range(ncolumns):
        htmlText += '''
            <th>Layer ''' + layers[i] + '''</th>
        '''


    for i in range(nrows):
        htmlText += "   <tr>"
        htmlText += '''
            <td>
                <div class="gallery">
                  <a target="_blank" href="''' + BaseImagePath+style_images[i] + '''">
                    <img src="''' + BaseImagePath+style_images[i] +'''" alt="Trolltunga Norway" width="300" height="200">
                  </a>
                  <div class="desc">'''+ style_images[i] + '''</div>
                </div>            
            </td>
        '''
        for j in range(ncolumns):
            filename = image_matrix[i][j]
            found = re.search("(C_[0-9\.]+)_(S_[0-9\.]+)_(L_[A-K])_([^.]+\..+)_at_iteration_(.+)\.png", filename)
            Cvalue = found.group(1)
            SValue = found.group(2)
            iterations = found.group(5)
            Description = Cvalue + ":" + SValue + ":iter " + iterations
            htmlText += '''
            <td>
                <div class="gallery">
                  <a target="_blank" href="''' + image_matrix[i][j] + '''">
                    <img src="''' + image_matrix[i][j] +'''" alt="Trolltunga Norway" width="300" height="200">
                  </a>
                  <div class="desc">'''+ Description + '''</div>
                </div>
            </td>                
            '''
        htmlText += "   </tr>"


    htmlText += '''
    </table>
    '''



    # <div class="gallery">
    #   <a target="_blank" href="img_fjords.jpg">
    #     <img src="img_fjords.jpg" alt="Trolltunga Norway" width="300" height="200">
    #   </a>
    #   <div class="desc">Add a description of the image here</div>
    # </div>
    #
    # <div class="gallery">
    #   <a target="_blank" href="img_forest.jpg">
    #     <img src="img_forest.jpg" alt="Forest" width="600" height="400">
    #   </a>
    #   <div class="desc">Add a description of the image here</div>
    # </div>
    #
    # <div class="gallery">
    #   <a target="_blank" href="img_lights.jpg">
    #     <img src="img_lights.jpg" alt="Northern Lights" width="600" height="400">
    #   </a>
    #   <div class="desc">Add a description of the image here</div>
    # </div>
    #
    # <div class="gallery">
    #   <a target="_blank" href="img_mountains.jpg">
    #     <img src="img_mountains.jpg" alt="Mountains" width="600" height="400">
    #   </a>
    #   <div class="desc">Add a description of the image here</div>
    # </div>
    #
    # <div class="gallery">
    #   <a target="_blank" href="">
    #     <img id="spare" src="" alt="Mountains" width="600" height="400">
    #   </a>
    #   <div class="desc">Add a description of the image here</div>
    # </div>


    htmlText += '''
    </body>
    </html>
    '''



    fos = open(root+"/index.html", "w")
    fos.write(htmlText)
    fos.close()

#print(htmlText)

for root, dirs, files in os.walk(path):
    if (len(files) > 0) and root.endswith("final"):
        #print(root, dirs, files)
        print(root)
        generateHTML(root, files)
