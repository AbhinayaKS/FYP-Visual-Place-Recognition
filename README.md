# Visual Place Recognition - Final Year Project

<div style="text-align: justify">

This repository contains the codebase for the Final Year Project **SCSE22-0277 - Visual Localization on NTU Campus**. This project is submitted in partial fulfillment of the degree requirements for Bachelors of Engineering (Computer Engineering). It was completed under the guidance of Professor Lin Weisi and PhD student Mr. Hu Zhi.

## Project Abstract
Visual localization is a key problem in various computer vision applications such as augmented reality and autonomous driving. Major challenges for visual localization include varying weather conditions, dynamic foregrounds, and varying viewpoints as seen in environments with dynamic objects such as the Nanyang Technological University Campus. 

Some efficient methods to represent images for the Visual Place Recognition task like Fischer Vectors (FV), Scale-Invariant Feature Transform (SIFT), and Vector of Locally Aggregated Descriptors (VLAD) can handle some of these challenges. Although VLAD provides a rich and effective method for image storage and retrieval, it models a static function. NetVLAD modifies the same to create a trainable function that minimizes the Euclidean distance between the query and the correct positive image that is used as baseline in this work. Soft assignment to clusters makes NetVLAD readily pluggable into Convolutional Neural Network architectures for end - to - end training. 

To de-prioritize task - irrelevant features; instead of uniform pooling as in the case of NetVLAD, Attention Pyramid Pooling of Salient Visual Residuals (APPSVR) uses attention, generated based on semantic segmentation. Three levels of attention in the form of local integration, global integration and parametric pooling handle the cases of task - irrelevant features, contextual information and weighting between clusters respectively. 

This paper aims to study the effect of semantic segmentation in visual localization; NetVLAD and APPVSR as potential solutions for visual localization in an indoor location like the Nanyang Technological University (NTU) Campus. Utilizing semantic information to generate attention has shown to be helpful with an increase in Recall@1 rates from 0.8381 to 0.8563.


## Data
The data is provided to the python files via the data directory. Please download the Pittsburgh250k dataset as well as the dataset specifications, place them in the data directory. This file structure is as follows:

#### Data Files
- `data/`
    - `pittsburgh/`
        - `000/`
        - `001/`
        - `002/`
        - `003/`
        - `004/`
        - `005/`
        - `006/`
        - `007/`
        - `008/`
        - `009/`
        - `010/`
        - `datasets/`
            - This contains the dataset specifications that can be downloaded from [here](https://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz).
        - `queries_real/`
            - Contains all the queries for testing the NetVLAD Model.

## Environment Setup
Create a virtual enviroment with python 3.9. The dependencies required by the project are:

| Dependency   | Version Used |
|--------------|--------------|
| Python       | 3.9.16       |
| Pytorch      | 1.13.1       |
| torchvision  | 0.14.1       |
| cudatoolkit  | 11.7.0       |
| Mask2Former  | -            |
| faiss-gpu    | 1.7.2        |
| tensorboardx | 2.2          |

## Testing the Project
The project can be tested by downloading the checkpoint available [here](https://entuedu-my.sharepoint.com/:u:/g/personal/abhinaya002_e_ntu_edu_sg/Ea8yqUQVFY1Pr8DL4DmHrWUB4rJr2iudokhvAVSZWkrrWA?e=dUQeUM). Please execute the following code after extracting the checkpoint.

```
python main.py --mode=test --split=val --resume=Mar13_10-52-00_vgg16_netvlad/
```

## Training the model
To train the model, first run the extraction of clusters.
```
python main.py --mode=cluster --arch=vgg16 --pooling=netvlad --num_clusters=64
```
Now execute the model in training mode. You can specify the number of epochs, learning rate and other hyperparameters. Please refer to main.py for more details.
```
python main.py --mode=train --arch=vgg16 --pooling=netvlad --num_clusters=64
```
In the case of training the model with semantic segmentation, please run the main.py with the label model first to extract and cache the labels.
```
python main.py --mode=label --arch=vgg16 --pooling=netvlad --num_clusters=64
```
Next, for training, execute the above two training commands with the --includeSemantic flag set to True.
