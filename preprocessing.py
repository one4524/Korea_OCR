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
    #h, w, c = image.shape
    ratio = image.shape[0] / 500
    orig = image.copy()
    image = imutils.resize(image, height=500)
    h, w, c = image.shape

    # 한 줄씩 하고 싶을 때 활성화
    # return image


    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply gaussian blur
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # Find edges in image using Canny algo
    edged = cv2.Canny(gray, 60, 120)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = []
    arr = []
    # loop over the contours
    if len(cnts) == 0:
        screenCnt.append([0, 0])
        screenCnt.append([w, 0])
        screenCnt.append([w, h])
        screenCnt.append([0, h])
        screenCnt = np.array(screenCnt)

    else:
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                print("approx = 4")
                break
            else:
                for i in range(0, len(approx)):
                    arr.append((np.array(approx[i][0])))

    if len(arr) != 0 and len(approx) != 4 and len(cnts) != 0:
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

    if len(screenCnt) == 4:
        # show the contour (outline) of the piece of paper
        cv2.drawContours(orig, [screenCnt], -1, (0, 255, 0), 2)

    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    cv2.imwrite('./data/test.jpg', warped)

    image_crop = warped.copy()

    # warped = orig
    # cv2.imwrite('./data/test.jpg', warped)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 191, offset=20, method="gaussian")
    warped = (warped > T).astype("uint8") * 255
    gray = cv2.erode(warped, kernel, iterations=2)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # 한 줄씩 추출
    g_h, g_w = gray.shape
    white = []
    black = []
    for y in range(0, g_h):
        key = 0
        check_num = 0
        for x in range(0, g_w):
            if gray[y, x] < 200:
                check_num += 1

            if check_num >= int(g_w/30):
                black.append(y)
                key = 1
                break

        if key == 0:
            white.append(y)

    # print("텍스트 없는 y줄 =", white)

    pre = 0
    if g_h > 1500:
        tum = 10
    elif g_h > 1000:
        tum = 7
    elif g_h > 500:
        tum = 4
    else:
        tum = 2

    box_line = []
    box = []
    for i in white:
        cv2.line(gray, (0, i), (g_w, i), (0, 0, 0), 3)
        if i - pre > tum:
            # print(i)
            box_line.append(pre + 1)
            box_line.append(i - 1)
            box.append(box_line.copy())
        box_line.clear()

        pre = i

        if i == white[-1] and g_h - i > tum:
            box_line.append(i + 1)
            box_line.append(g_h-1)
            box.append(box_line.copy())

    print("box", box)

    lines = []
    for i, b in enumerate(box):
        crop_line = image_crop[b[0]:b[1], 0:g_w]
        filename2 = "./crop_line/{}.png".format(i)
        cv2.imwrite(filename2, crop_line)
        lines.append(crop_line)

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it

    cv2.imwrite('./data/test2.jpg', gray)

    # return gray
    return lines

# https://github.com/muratlutfigoncu/receipt-reader 참고
# https://medium.com/cashify-engineering/improve-accuracy-of-ocr-using-image-preprocessing-8df29ec3a033 참고
