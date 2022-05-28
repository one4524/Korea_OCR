import json
import random
import os
from tqdm import tqdm

data_type = 'ocr'  # htr, ocr
labeling_filename = 'printed_data_info.json'   # handwriting_data_info1

## Check Json File
file = json.load(open(f'./kor_dataset/aihub_data/{data_type}/{labeling_filename}', encoding='UTF8'))
ocr_good_files = os.listdir(f'./kor_dataset/aihub_data/{data_type}/images/')
len(ocr_good_files) # 37220

random.shuffle(ocr_good_files)

n_train = int(len(ocr_good_files) * 0.7)
n_validation = int(len(ocr_good_files) * 0.15)
n_test = int(len(ocr_good_files) * 0.15)

print(n_train, n_validation, n_test) # 26054 5583 5583

train_files = ocr_good_files[:n_train]
validation_files = ocr_good_files[n_train: n_train+n_validation]
test_files = ocr_good_files[-n_test:]

## train/validation/test 이미지들에 해당하는 id 값을 저장

train_img_ids = {}
validation_img_ids = {}
test_img_ids = {}
num = 0
for image in file['images']:
    num = num + 1
    if num % 3000 == 0:
        print(num)
    if image['file_name'] in train_files:
        train_img_ids[image['file_name']] = image['id']
    elif image['file_name'] in validation_files:
        validation_img_ids[image['file_name']] = image['id']
    elif image['file_name'] in test_files:
        test_img_ids[image['file_name']] = image['id']

## train/validation/test 이미지들에 해당하는 annotation 들을 저장
print('for1 end') # 26054 5583 5583

train_annotations = {f:[] for f in train_img_ids.keys()}
validation_annotations = {f:[] for f in validation_img_ids.keys()}
test_annotations = {f:[] for f in test_img_ids.keys()}

print('annotations 1') # 26054 5583 5583        # 878100

train_ids_img = {train_img_ids[id_]:id_ for id_ in train_img_ids}
validation_ids_img = {validation_img_ids[id_]:id_ for id_ in validation_img_ids}
test_ids_img = {test_img_ids[id_]:id_ for id_ in test_img_ids}

print('for2 start') # 26054 5583 5583
for idx, annotation in enumerate(file['annotations']):
    if idx % 5000 == 0:
        print(idx,'/',len(file['annotations']),'processed')
    if annotation['attributes']['class'] != 'word':
        continue
    if annotation['image_id'] in train_ids_img:
        train_annotations[train_ids_img[annotation['image_id']]].append(annotation)
    elif annotation['image_id'] in validation_ids_img:
        validation_annotations[validation_ids_img[annotation['image_id']]].append(annotation)
    elif annotation['image_id'] in test_ids_img:
        test_annotations[test_ids_img[annotation['image_id']]].append(annotation)

print('for2 end') # 26054 5583 5583

with open('train_annotation.json', 'w') as file:
    json.dump(train_annotations, file)
with open('validation_annotation.json', 'w') as file:
    json.dump(validation_annotations, file)
with open('test_annotation.json', 'w') as file:
    json.dump(test_annotations, file)



## Make gt_xxx.txt files
data_root_path = f'./kor_dataset/aihub_data/{data_type}/images/'
save_root_path = f'./deep-text-recognition-benchmark/{data_type}_data/'

obj_list = ['test', 'train', 'validation']
for obj in obj_list:
  total_annotations = json.load(open(f'./{data_type}_{obj}_annotation.json'))
  gt_file = open(f'{save_root_path}gt_{obj}.txt', 'w')
  for file_name in tqdm(total_annotations):
    annotations = total_annotations[file_name]
    for idx, annotation in enumerate(annotations):
      text = annotation['text']
      gt_file.write(f'{obj}/{file_name}\t{text}\n')