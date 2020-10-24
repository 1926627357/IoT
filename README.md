# Human Activity Recognition

## Table of Contents

* 1 [markdown-toc](#markdown-toc)
  * 1.1 [变更日志](#变更日志)
  * 1.2 [Features](#features)
* 2 [环境依赖](#环境依赖)
  * 2.1 [JDK](#jdk)
  * 2.2 [Maven](#maven)
* 3 [快速入门](#快速入门)
  * 3.1 [maven 引入](#maven-引入)
  * 3.2 [md 文件](#md-文件)
  * 3.3 [快速开始](#快速开始)
* 4 [属性配置](#属性配置)
  * 4.1 [属性说明](#属性说明)
  * 4.2 [返回值说明](#返回值说明)
* 5 [测试案例](#测试案例)
* 6 [其他](#其他)

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
