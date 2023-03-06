# Visual Place Recognition - Final Year Project

This repository contains the codebase for the Final Year Project **SCSE22-0277 - Visual Localization on NTU Campus**. This project is submitted in partial fulfillment of the degree requirements for Bachelors of Engineering (Computer Engineering). It was completed under the guidance of Professor Lin Weisi and PhD student Mr. Hu Zhi.

### Data
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

### Environment Setup
- Create a virtual enviroment with python 3.10
- Run the following code to download the dependencies from the requirements.txt file
```
conda install --file requirements.txt -c pytorch -c nvidia -c conda-forge -c auto -c domdfcoding -c anaconda
```
