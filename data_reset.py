import os
from glob import glob

import pandas as pd
import numpy as np
import random
import shutil

test_dir = '/opt/ml/input/data/train'
train_ds = pd.read_csv(test_dir+'/train.csv',  index_col='id')

# reset the existing dataset

for path in ['/opt/ml/input/data/val_18class', '/opt/ml/input/data/train_18class']:
    for i in range(18):
        if os.path.exists(os.path.dirname(f'{path}/{i}/')):
            shutil.rmtree(os.path.dirname(f'{path}/{i}/'))
        os.makedirs(os.path.dirname(f'{path}/{i}/'), exist_ok=True)

# make dir for trainset
# 18class
for path in ['/opt/ml/input/data/val_18class', '/opt/ml/input/data/train_18class']:
    for i in range(18):
        os.makedirs(os.path.dirname(f'{path}/{i}/'), exist_ok=True)

# labeling by 18 class
for person in train_ds['path']:
    os.chdir('/opt/ml/input/data')
    img_lst = os.listdir(f'train/images/{person}')
    gender= person.split('_')[1]
    age = int(person.split('_')[3])
    for photo_n in img_lst:
        fl = photo_n[0]
        if fl == 'm':
            if gender != 'female':
                if age < 30:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/0/{person}_{photo_n}')
                elif age < 60:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/1/{person}_{photo_n}')
                else:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/2/{person}_{photo_n}')
            else:
                if age < 30:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/3/{person}_{photo_n}')
                elif age < 60:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/4/{person}_{photo_n}')
                else:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/5/{person}_{photo_n}')
        elif fl == 'i':
            if gender != 'female':
                if age < 30:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/6/{person}_{photo_n}')
                elif age < 60:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/7/{person}_{photo_n}')
                else:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/8/{person}_{photo_n}')
            else:
                if age < 30:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/9/{person}_{photo_n}')
                elif age < 60:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/10/{person}_{photo_n}')
                else:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/11/{person}_{photo_n}')
        elif fl == 'n':
            if gender != 'female':
                if age < 30:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/12/{person}_{photo_n}')
                elif age < 60:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/13/{person}_{photo_n}')
                else:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/14/{person}_{photo_n}')
            else:
                if age < 30:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/15/{person}_{photo_n}')
                elif age < 60:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/16/{person}_{photo_n}')
                else:
                    shutil.copy(f'train/images/{person}/{photo_n}', f'train_18class/17/{person}_{photo_n}')

# integrity checker
integrity = 1

# integrity check
img_lst = glob(os.path.join(f'/opt/ml/input/data/train_18class', '*/*'))

n_photo = len(img_lst)
n_people = len(train_ds)

mask_n = 0
for i in range(6):
    mask_n += len(glob(os.path.join(f"/opt/ml/input/data/train_18class", f"{i}/*")))
# check integrity of mask
integrity = min(mask_n == n_people*5, integrity)

male_n = 0
for i in [0,1,2,6,7,8,12,13,14]:
    male_n += len(glob(os.path.join(f"/opt/ml/input/data/train_18class", f"{i}/*")))
# check integrity of male
integrity = min(male_n == train_ds.groupby('gender').count()['race']['male']*7, integrity)

age_n = 0
for i in [0,3,6,9,12,15]:
    age_n += len(glob(os.path.join(f"/opt/ml/input/data/train_18class", f"{i}/*")))
# check integrity of age
bins = [0, 29, 59, 120]
integrity = min(age_n != train_ds['age'].value_counts(bins=bins, sort=False)[0]*7, integrity)

os.chdir('/opt/ml/input/data')
img_lst = glob(os.path.join(f'/opt/ml/input/data/train_3class', '*/*/*'))

n_photo = len(img_lst)
n_people = len(train_ds)

# photo number integrity check
integrity = min(n_photo == n_people*7*3, integrity)

# age integrity
integrity = min(len(glob(os.path.join(f'/opt/ml/input/data/train_3class/age', '30to60/*'))) == 1227*7, integrity)
# gender integrity
integrity = min(len(glob(os.path.join(f'/opt/ml/input/data/train_3class/gender', 'female/*')))==train_ds.groupby('gender').count()['race']['female']*7, integrity)

check1 = 42
check2 = 14
check3 = 6

# Modify Wrongly Labelled Data
for i in range(0,18):
    img_lst = glob(os.path.join(f'/opt/ml/input/data/train_18class', f'{i}/*'))
    for img_name in img_lst:
        idn = img_name.split('_')[1].split('/')[-1]
        if idn in ['006359', '006360', '006361', '006362', '006363', '006364']:
            #print(img_name)
            #print(f'/opt/ml/input/data/train_18class/{i-3}/r{img_name.split("/")[-1][1:]}')
            shutil.move(img_name, f'/opt/ml/input/data/train_18class/{i-3}/r{img_name.split("/")[-1][1:]}')
            check1 -= 1
        elif idn in ['001498-1', '004432']:
            #print(img_name)
            #print(f'/opt/ml/input/data/train_18class/{i+3}/{img_name.split("/")[-1]}')
            shutil.move(img_name, f'/opt/ml/input/data/train_18class/{i+3}/r{img_name.split("/")[-1][1:]}')
            check2 -= 1
        elif idn in ['000020', '004418', '005227']:
            if 'incorrect' in img_name:
                #print(img_name)
                #print(f'/opt/ml/input/data/train_18class/{i+6}/{img_name.split("/")[-1]}')
                shutil.move(img_name, f'/opt/ml/input/data/train_18class/{i+6}/r{img_name.split("/")[-1][1:]}')
                check3 -=1
            elif 'normal' in img_name:
                #print(img_name)
                #print(f'/opt/ml/input/data/train_18class/{i-6}/{img_name.split("/")[-1]}')
                shutil.move(img_name, f'/opt/ml/input/data/train_18class/{i-6}/r{img_name.split("/")[-1][1:]}')
                check3 -= 1
    
# Modification Integrity check
if check1 ==0 and check2 ==0 and check3 ==0:
    integrity = True
else:
    integrity = False


#  split stratified validation dataset
for i in range(0,6):
    img_lst = glob(os.path.join(f'/opt/ml/input/data/train_18class', f'{i}/*'))
    id_lst = []
    for img_name in img_lst:
        id_lst.append(img_name.split('_')[1].split('/')[-1])
    id_lst = list(set(id_lst))
    random.shuffle(id_lst)
    id_lst = id_lst[:int(len(id_lst)*0.2)]
    
    for j in [0,6,12]:
        new_img_lst = glob(os.path.join(f'/opt/ml/input/data/train_18class', f'{i+j}/*'))
        for ide in id_lst:
            for new_img_name in new_img_lst:
                if new_img_name.split('_')[1].split('/')[-1] == ide:
                    name = new_img_name.split('/')[-1]
                    shutil.move(new_img_name, f'val_18class/{i+j}/{name}')

# check if any same person in both train and validation
for i in range(0,18):
    img_lst = glob(os.path.join(f'/opt/ml/input/data/val_18class', f'{i}/*'))
    v_id_lst = []
    for img_name in img_lst:
        v_id_lst.append(img_name.split('_')[1].split('/')[-1])

    img_lst = glob(os.path.join(f'/opt/ml/input/data/train_18class', f'{i}/*'))
    t_id_lst = []
    for img_name in img_lst:
        t_id_lst.append(img_name.split('_')[1].split('/')[-1])

    if set(v_id_lst) & set(t_id_lst):
        integrity = False
    else:
        integrity = True

# check if 2:8 split is done
for i in range(0,18):
    v_img_lst = glob(os.path.join(f'/opt/ml/input/data/val_18class', f'{i}/*'))
    t_img_lst = glob(os.path.join(f'/opt/ml/input/data/train_18class', f'{i}/*'))
    v = len(v_img_lst)/(len(t_img_lst)+len(v_img_lst))
    integrity = min((0.19 <v<0.21), integrity)

print("dataset created")
# Integrity check print
if integrity:
    print("Integrity check complete. Safe to use the dataset")
else:
    print("Dataset Incomplete. Unsafe to use the dataset")