# :mask: Image-classification-level1-30

## 📖 Main Subject

## :floppy_disk: Installation
**1. Set up the python environment:**

  - Python = 3.6
  - torch = 1.2.0
  - torchVison = 0.4.0
  - OpenCV-python = 4.1.1.26
  - pillow = 6.2.1
  - vispy = 0.6.3
  - scipy = 1.1.0
  - minSdkVersion: 23
  - targetSdkVersion: 29
  - JAVA jdk: 1.8.0_241

```
$ git clone https://github.com/boostcampaitech2/image-classification-level1-30.git
$ pip install -r $ROOT/image-classification-level1-30/requirements.txt
```

## Function Description
`main.py`: main 함수

`train.py`: 한 epoch를 학습시키는 코드

`loss.py`: loss 함수와 f1_score metric 계산이 포함된 클래스

`dataset.py`: Dataset 클래스가 포함된 코드

`model.py`: EfficientNet 클래스가 포함된 코드

`utils.py`: EfficientNet에 필요한 코드가 포함됨

`labeling.ipynb`: ages.csv, genders.csv, masks.csv, labels.csv, images.csv를 생성하는 코드(참고용)

`sample_submission.ipynb`: 기존에 주어진 베이스라인 코드
