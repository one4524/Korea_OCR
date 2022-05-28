import json
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

## aihub 데이터 annotation을 읽어서 단어 단위로 잘라서 data에 저장하기

data_root_path = '/data/ljh/dataset/ocr/Goods/'
save_root_path = '/data/ljh/OCR/deep-text-recognition-benchmark/data/'

test_annotations = json.load(open('./test_annotation.json'))
gt_file = open(save_root_path + 'gt_test.txt', 'w')
for file_name in tqdm(test_annotations):
    annotations = test_annotations[file_name]
    image = cv2.imread(data_root_path + file_name)
    for idx, annotation in enumerate(annotations):
        x, y, w, h = annotation['bbox']
        if x <= 0 or y <= 0 or w <= 0 or h <= 0:
            continue
        text = annotation['text']
        crop_img = image[y:y + h, x:x + w]
        crop_file_name = file_name[:-4] + '_{:03}.jpg'.format(idx + 1)
        cv2.imwrite(save_root_path + 'test/' + crop_file_name, crop_img)
        gt_file.write("test/{}\t{}\n".format(crop_file_name, text))

validation_annotations = json.load(open('./validation_annotation.json'))
gt_file = open(save_root_path + 'gt_validation.txt', 'w')
for file_name in tqdm(validation_annotations):
    annotations = validation_annotations[file_name]
    image = cv2.imread(data_root_path + file_name)
    for idx, annotation in enumerate(annotations):
        x, y, w, h = annotation['bbox']
        if x <= 0 or y <= 0 or w <= 0 or h <= 0:
            continue
        text = annotation['text']
        crop_img = image[y:y + h, x:x + w]
        crop_file_name = file_name[:-4] + '_{:03}.jpg'.format(idx + 1)
        cv2.imwrite(save_root_path + 'validation/' + crop_file_name, crop_img)
        gt_file.write("validation/{}\t{}\n".format(crop_file_name, text))

train_annotations = json.load(open('./train_annotation.json'))
gt_file = open(save_root_path + 'gt_train.txt', 'w')
for file_name in tqdm(train_annotations):
    annotations = train_annotations[file_name]
    image = cv2.imread(data_root_path + file_name)
    for idx, annotation in enumerate(annotations):
        x, y, w, h = annotation['bbox']
        if x <= 0 or y <= 0 or w <= 0 or h <= 0:
            continue
        text = annotation['text']
        crop_img = image[y:y + h, x:x + w]
        crop_file_name = file_name[:-4] + '_{:03}.jpg'.format(idx + 1)
        cv2.imwrite(save_root_path + 'train/' + crop_file_name, crop_img)
        gt_file.write("train/{}\t{}\n".format(crop_file_name, text))