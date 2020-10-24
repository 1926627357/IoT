# Human Activity Recognition

<!--TOC-->
- [Abstract](# Abstract)
<!--TOC-->

## Abstract

This repo is HAR-CNN code


## Requirement
Pytorch 1.x

Scikit-learn


## Usage
Fistly, we can set some necessary configurations in `config.py`.
Configuration Table
|Config|Value|Description|
|:-:|:-:|:-:|
|epoch|20|-|
|lr|0.001|-|
|momentum|0.9|-|
|batch_size|64|-|
|linke_train|./linke_train/|training dataset directory|
|linke_test|./linke_test/|testing dataset directory|
|result_file|result.csv|the file that stores training result|

Then begin to train network by following command:
```sh
[username:/home/workspace/IoT]$ python main_pytorch.py
```

## Result
Training Accuracy: 91.26%
Testing Accuracy: 90.26%
