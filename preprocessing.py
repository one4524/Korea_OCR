import tempfile
from skimage.filters import threshold_local
from PIL import Image
import pytesseract
import numpy as np
import os
import imutils
import re
import cv2
from skimage.filters.thresholding import threshold_local


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def imageProcesser(imagePath):
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((2, 2), np.uint8)

    # load the image
    image = cv2.imread(imagePath)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    return image

    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # apply gaussian blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Find edges in image using Canny algo
    edged = cv2.Canny(gray, 60, 120)

    cv2.imshow("Output", imutils.resize(edged, height=650))

    # Waits for a key insert to close images.
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = []
    arr = []
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            print("4")
            break
        else:
            print(len(approx))
            for i in range(0, len(approx)):
                arr.append((np.array(approx[i][0])))


    if len(screenCnt) == 4:
        # show the contour (outline) of the piece of paper
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

    if len(arr) != 0:
        min_x, max_x, min_y, max_y = 1000, 0, 1000, 0

        for a in arr:
            if a[0] != 0:
                min_x = min(min_x, a[0])
                max_x = max(max_x, a[0])

            if a[1] != 0:
                min_y = min(min_y, a[1])
                max_y = max(max_y, a[1])

        print(min_x, max_x, min_y, max_y)
        screenCnt.append([min_x, min_y])
        screenCnt.append([max_x, min_y])
        screenCnt.append([max_x, max_y])
        screenCnt.append([min_x, max_y])
        screenCnt = np.array(screenCnt)

    print(screenCnt)

    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    cv2.imwrite('./data/test.jpg', warped)

    return warped

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 191, offset=20, method="gaussian")
    warped = (warped > T).astype("uint8") * 255
    gray = cv2.erode(warped, kernel, iterations=2)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it

    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    cv2.imwrite('./data/test.jpg', gray)

    # show the output images
    cv2.imshow("Original", imutils.resize(orig, height=650))
    cv2.imshow("Output", imutils.resize(gray, height=650))

    # Waits for a key insert to close images.
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return gray

# https://github.com/muratlutfigoncu/receipt-reader 참고
# https://medium.com/cashify-engineering/improve-accuracy-of-ocr-using-image-preprocessing-8df29ec3a033 참고


# filename = imageProcesser("test.jpg")
# print(filename)
