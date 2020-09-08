# https://www.learnopencv.com/blob-detection-using-opencv-python-c/

# Standard imports
import cv2
import numpy as np;

# Read image
image = 'img/blob_detection.jpg'
# image = 'img/butterflies_pexels-photo-326055_scaled.jpg'
# image = 'images/la-fi-mo-incandescent-lightbulb-ban-20140101-001.jpg'
im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 1
params.maxThreshold = 200
# params.thresholdStep = 10
# Filter by Area.
params.filterByArea = True
params.minArea = 10
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.1
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.1

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)
print(detector)

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)