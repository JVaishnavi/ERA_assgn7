#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 08:04:07 2023

@author: vaishnavijanakiraman
"""

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3) # IP: 28, OP: 26, RF: 3
        self.conv2 = nn.Conv2d(16, 32, 3) # IP: 26, OP: 24, RF: 5
        self.pool1 = nn.MaxPool2d(2, 2) # IP: 24, OP: 12, RF: 7
        self.conv3 = nn.Conv2d(32, 4, 1) # IP: 12, OP: 12, RF: 7
        self.conv4 = nn.Conv2d(4, 16, 3) # IP: 12, OP: 10, RF: 11
        self.pool2 = nn.MaxPool2d(2, 2) # IP: 10, OP: 5, RF: 19
        self.conv5 = nn.Conv2d(16, 10, 1) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(10, 10, 5) # 7 > 5 | 30
        
        

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)