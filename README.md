# PDE-Identification-Features
This github repository is for our paper "Data-driven Identification of 2D Partial Differential Equations using Extracted Physical Features" published in the journal of Computer Methods in Applied Mechanics and Engineering (CMAME). To read the paper, please visit the following link: https://www.sciencedirect.com/science/article/pii/S0045782521001687

# Data
Data samples were generated using COMSOL Physics PDE solver using FEM, and then normalized and reformatted to samples of size (21,21,500). You can access the data for 8 PDEs considered in this work from: https://drive.google.com/drive/folders/1hVcCHKJ-PWBbEWziGttiSqfruE3SK1Xy?usp=sharing .

# Physical Feature Extraction
Features are visualized and extracted from raw data samples in FEATURE_EXTRACTION.ipynb. The extracted features are then saved in the folder FEATURES. The processed features also can be accessed from the drive: https://drive.google.com/drive/u/0/folders/1hVcCHKJ-PWBbEWziGttiSqfruE3SK1Xy .

# Prediction Model
The pipeline of two experiments explained in the paper for identification of Partial Differential Equations from the extracted features are available in EXPERIMENTS.ipynb

# 3D Convolutional Neural Network
The network and the model for training the network based on the labels of terms in the PDEs are in the 3DCNN folder. For the prediction of unseen equations, the model should be trained for each term and unseen PDE separately. The processed data for this model with shape of (21,21,21) for each sample can be accessed from: https://drive.google.com/drive/folders/1hVcCHKJ-PWBbEWziGttiSqfruE3SK1Xy?usp=sharing .

# Dependencies
* Numpy
* Pandas
* Scipy
* XGBoost
* sklearn
* seaborn
* matplotlib
* PyTorch

# Authors
This work was done by Kazem Meidani under supervision of Professor Amir Barati Farimani at MAIL at Carnegie Mellon University.

# Acknowledgments
The authors would like to thank Francis Ogoke and Zhonglin Cao for their valuable comments and edits. 
