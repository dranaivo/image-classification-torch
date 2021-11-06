# Pytorch image classification

This is a framework for producing image classification models.

# Attribution

The codebase is inspired from the course **Deep Learning with PyTorch** by [OpenCV.org](https://opencv.org/courses/). Take a look at the [LICENCE](./LICENCE) if you are using/forking this repository.

# Installation

First, install `torch` and `torchvision` (refer to [pytorch installation guide](https://pytorch.org/get-started/locally/)):
```bash
# for python 3.6 and cuda 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

Then, install the other packages :
```bash
pip install -r requirements.txt
```

# How to use

## Training

You will need a classification dataset to launch a training. Also, there's a configuration file **cfg** ([`trainer/configuration.py`](trainer/configuration.py)).

**Dataset**

The dataset should contain a `training` and `validation` folder. Then, each of these folders should have one sub-folder per class category, with the respective images of this category in the appropriate sub-folder. For an example of **cat - dog - panda** classification :
```bash
cat-dog-panda/
├── training
│   ├── cat
│   ├── dog
│   └── panda
└── validation
    ├── cat
    ├── dog
    └── panda
```
In the **cfg**, you also need to indicate the number of classes :
```python
# trainer/configuration.py
class DataConfig:
    ...
    root_dir: str = "cat-dog-panda/"
    n_classes: int = 3
    ...
```

**Launch training**

In the **cfg**, input your training configurations using the `@dataclass` : 
```python
class OptimizerConfig:
    ...

class TrainerConfig:
    ...
```

Execute : `python train.py`.

## Visualize

TBA...

## Prediction

TBA...
