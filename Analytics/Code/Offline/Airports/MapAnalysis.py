# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
from PIL import Image
import  StringIO
from matplotlib import pyplot as plt
import pytesseract

def DetectText( myImage ):

    #gray = cv2.cvtColor( myImage, cv2.COLOR_BGR2GRAY ) # grayscale
    #_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) # threshold
    _, thresh = cv2.threshold(myImage, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # get contours

    # for each contour found, draw a rectangle around it on original image
    for contour in contours:
        
        # get rectangle bounding contour
        [x,y,w,h] = cv2.boundingRect(contour)

        # discard areas that are too large
        #if h>300 and w>300:
        # continue

        # discard areas that are too small
        #if h<40 or w<40:
        #    continue

    # draw rectangle around contour on original image
    cv2.rectangle( myImage,(x,y), (x+w,y+h), (255,0,255), 2 )
    
    return myImage

    # write original image with added contours to disk
    #cv2.imwrite('contoured.jpg', image)


def drawMatches(img1, kp1, img2, kp2, matches):

    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Sift( mySmallImage, myLargeImage ):

    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(mySmallImage, None)
    kp2, des2 = sift.detectAndCompute(myLargeImage, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:

        print "This is cool..."
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = mySmallImage.shape
        pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)

        cv2.polylines(myLargeImage, [np.int32(dst)], True, 255, 3)

    else:
        print "Not enough matches are found - %d/%d" % ( len(good), MIN_MATCH_COUNT )
        matchesMask = None
        return

    draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    myProof = drawMatches(mySmallImage, kp1, myLargeImage, kp2, good)

    plt.imshow(myProof, 'gray'),plt.show()


def DrawTransforms( myImage ):

    ret,th1 = cv2.threshold(myImage, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(myImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(myImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [ myImage, th1, th2, th3 ]

    for i in xrange(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()

##########################################################
 
def auto_canny( myImage, mySigma=0.1):

    # compute the median of the single channel pixel intensities
    v = np.median( myImage )
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - mySigma) * v))
    upper = int(min(255, (1.0 + mySigma) * v))
    edged = cv2.Canny( myImage, lower, upper )
 
    # return the edged image
    return edged
    #myResult = Image.fromarray( (edged).astype(np.uint8) )
    #myResult.save('Jennifer_Edges_lowSigma.jpg')

##########################################################

img1 = cv2.imread('./iah_terminal_a_540_nl.png', 0)
img2 = cv2.imread('./iah_terminalA_googleMaps.png', 0)

myNewImage = Image.open(u'./iah_terminal_a_540_nl.png')
myNewImage.load()
print pytesseract.image_to_string( myNewImage )

img1 = DetectText( img1 )
DrawTransforms( img1 )

img1 = cv2.medianBlur(img1, 5)
img2 = cv2.medianBlur(img2, 5)

img1_thres = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
img2_thres = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#_, img2_thres = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
#th2 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
#th3 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

Sift(img1_thres, img2_thres)

'''

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input dataset of images")
args = vars(ap.parse_args())
 
# loop over the images
for imagePath in glob.glob(args["images"] + "/*.png"):
    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
 
    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = auto_canny(blurred)
 
    # show the images
    cv2.imshow("Original", image)
    cv2.imshow("Edges", np.hstack([wide, tight, auto]))
    cv2.waitKey(0)

#myImage = cv2.imread( './Jennifer.jpg' )
#print type(myEdgedImage)
#print myEdgedImage

'''
