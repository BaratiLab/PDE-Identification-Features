# importing the libraries
import pandas as pd
import numpy as np
import os
# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import pickle


# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.num_classes = 2
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(3**3*64, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, self.num_classes)
        self.relu = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(512)
        self.batch2=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15) 
        self.softmax = nn.Softmax(dim=1)     
        self.extract = [] 
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    
    def forward(self, x , test = False):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        if (test == True):
            self.extract.append(out.cpu().numpy())
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.batch2(out)
        out = self.fc4(out)
        out = self.softmax(out)
        return out
    
    def extract_out(self):
        return np.array(self.extract)