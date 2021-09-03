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
`main.py`: main module that combines and runs all other sub-modules

`train.py`: trains the model by iterating through specific number of epochs

`model.py`: EfficientNet model from [lukemelas](https://github.com/lukemelas/EfficientNet-PyTorch)

`utils.py`: required by EfficientNet model

`inference.py`: tests the model using the test dataset and outputs the inferred csv file

`loss.py`: calculates loss using cross entropy and f1-score

`label_smoothing_loss.py`: calculates loss using cross entropy with label smoothing and f1-score

`dataset.py`: generates the dataset to feed the model to train

`data_reset.py`: generates the image dataset divided into 18 classes (train and validation)

`early_stopping.py`: Early Stopping function from [Bjarten](https://github.com/Bjarten/early-stopping-pytorch) (patience decides how many epochs to tolerate after val loss exceeds min. val loss)

`transformation.py`: a group of transformation functions that can be claimed by args parser

`dashboard.ipynb`: can observe the images with labels from the inferred csv files
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

- Consult [transformation.py](https://github.com/boostcampaitech2/image-classification-level1-30/blob/main/transformation.py) for detailed explanation on the types of transformation

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


