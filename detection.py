"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def craft_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    return boxes, polys


def get_detector(trained_model, cuda=True):
    # load net
    net = CRAFT()  # initialize

    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    return net


def get_textbox(detector, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, cuda=True, poly=False,
                refine_net=None):
    result = []
    image = imgproc.loadImage(image)

    bboxes, polys = craft_net(detector, image, text_threshold, link_threshold, low_text,
                              cuda, poly, refine_net)

    for i, box in enumerate(polys):
        single_img_result = []
        poly = np.array(box).astype(np.int32).reshape((-1))
        single_img_result.append(poly)

        min_x = min(poly[0::2])
        max_x = max(poly[0::2])
        min_y = min(poly[1::2])
        max_y = max(poly[1::2])

        img_h = max_y - min_y
        img_w = max_x - min_x

        img_trim = image[min_y:min_y + img_h, min_x:min_x + img_w]
        cv2.imwrite('./image/{}.jpg'.format(str(i)), img_trim)

        result.append(single_img_result)

    return result


def get_textbox2(detector, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, cuda=True, poly=False,
                 refine_net=None):
    result = []
    image = imgproc.loadImage(image)

    bboxes, polys = craft_net(detector, image, text_threshold, link_threshold, low_text,
                              cuda, poly, refine_net)

    w, h = 200, 64
    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape(-1)

        poly2 = np.array([[poly[0], poly[1]], [poly[2], poly[3]], [poly[4], poly[5]], [poly[6], poly[7]]], dtype=np.float32)
        dstQuad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        perspect = cv2.getPerspectiveTransform(poly2, dstQuad)
        img_trim = cv2.warpPerspective(image, perspect, (w, h))

        cv2.imwrite('./image2/{}.jpg'.format(str(i)), img_trim)

        result.append(poly)

    return result
