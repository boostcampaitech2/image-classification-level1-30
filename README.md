# :mask: Image-classification-level1-30

## Main Subject
As COVID-19 largely impacts people’s lives socially and economically, many scientists and engineers contemplated how to utilize technology to alleviate people’s suffer. One of the most effective and easiest ways to prevent the spread is agreed to be wearing masks. Thus, we came up with this project which implements machine learning mechanism to classify people wearing masks either properly or not as well as their gender and age.
<br/><br/>

## Installation
**1. Set up the python environment:**
- Recommended python version 3.8.5
```
$ conda create -n venv python=3.8.5 pip
$ conda activate venv
```
**2. Install other required packages**
  - torch==1.7.1
  - torchvision==0.8.2
  - tensorboard==2.4.1
  - pandas==1.1.5
  - opencv-python==4.5.1.48
  - scikit-learn~=0.24.1
  - matplotlib==3.2.1
  - albumentations==1.0.3

```
$ pip install -r $ROOT/image-classification-level1-30/requirements.txt
```
<br/>

## Classes for Classification
- Three subclasses (mask, gender, and age) are combined to have a total of eighteen classes
<img src=https://i.imgur.com/efDFm0m.png>
<br/>

## Function Description
`main.py`: main 함수

`train.py`: 한 epoch를 학습시키는 코드

`model.py`: EfficientNet 클래스가 포함된 코드

`inference.py`: 추론합니다

`loss.py`: loss 함수와 f1_score metric 계산이 포함된 클래스

`label_smoothing_loss.py`:

`dataset.py`: Dataset 클래스가 포함된 코드

`data_reset.py`: 데이터 리셋합니다

`early_stopping.py`: 일찍 멈춥니다

`utils.py`: EfficientNet에 필요한 코드가 포함됨

`transformation.py`: 이미지 변경합니다

`dashboard.ipynb`: 대쉬보드에 이미지 나옵니다
<br/><br/>

## USAGE
### 1. Data Generation

- Before Data Generation:
```
input
└──data
    ├──eval
    |  ├──images/
    |  └──info.csv
    └──train
        ├──images/
        └──train.csv
```

- Run python file to generate mask classification datasets
```
$ python data_reset.py
```

- After Data Generation:
```
input
└──data
    ├──eval
    |  ├──images/
    |  └──info.csv
    └──train
        ├──images/
        ├──train_18class/
        ├──val_18class/
        └──train.csv
```

### 2. Model Training

- Early stopping applied by (default) 

```
$ python main.py --model 7 --tf yogurt --lr 2e-3 --batch_size 16 --num_workers 4 --patience 10 --cut_mix --epochs 100
```

**Image Transformation**<br>
- argument parser `--tf` can receive types of augmentation
- Transformation functions applied to training datasets and test datasets are different: image transformation for inference should be modified as little as possible

- Consult [transformation.py](https://github.com/boostcampaitech2/image-classification-level1-30/blob/main/transformation.py ) for detailed explanation on the types of transformation

### 3. Inference
```
$ python inference.py --tf yogurt
```
- Running the line above will generate submission.csv as below

```
input
└──data
    ├──eval
    |  ├──images/
    |  ├──submission.csv
    |  └──info.csv
    └──train
        ├──images/
        ├──train_18class/
        ├──val_18class/
        └──train.csv
```


