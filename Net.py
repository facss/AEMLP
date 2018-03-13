#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


#naive AE model
class AE (nn.Module):
    def __init__(self,inputsize):
        super(AE,self).__init__()
        
        self.encoder=nn.Sequential(
            nn.Linear(inputsize*inputsize,inputsize*int(inputsize/4)),
            nn.ReLU(),
            nn.Linear(inputsize*int(inputsize/4),inputsize*int(inputsize/16)),
        )

        self.decoder=nn.Sequential(
            nn.Linear(inputsize*int(inputsize/16),inputsize*int(inputsize/4)),
            nn.ReLU(),
            nn.Linear(inputsize*int(inputsize/4),inputsize*inputsize),
        )
    def forward(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return encoded,decoded

class MLP(nn.Module):
    def __init__(self,mode,inputsize,bits_len):
        super(MLP,self).__init__()
        if mode=='ORL':
            self.fc1=nn.Sequential(
                nn.Linear(inputsize*int(inputsize/16),inputsize*int(inputsize/4)),
                nn.ReLU(),
                nn.BatchNorm1d(inputsize*int(inputsize/4)),

                nn.Linear(inputsize*int(inputsize/8),1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),

                nn.Linear(1024,512),
                nn.ReLU(),
                nn.BatchNorm1d(512),

                nn.Linear(512,bits_len),
                nn.ReLU(),
                nn.BatchNorm1d(bits_len),
            ) 
        if mode=='CMUPIE':
            self.fc1=nn.Sequential(
                nn.Linear(inputsize*int(inputsize/16),inputsize*int(inputsize/4)),
                nn.ReLU(),
                nn.BatchNorm1d(inputsize*int(inputsize/4)),

                nn.Linear(inputsize*int(inputsize/8),1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),

                nn.Linear(1024,512),
                nn.ReLU(),
                nn.BatchNorm1d(512),

                nn.Linear(512,bits_len),
                nn.ReLU(),
                nn.BatchNorm1d(bits_len),
            )
        if mode=='PUBFIG83':
            self.fc1=nn.Sequential(
                nn.Linear(inputsize*int(inputsize/16),inputsize*int(inputsize/4)),
                nn.ReLU(),
                nn.BatchNorm1d(inputsize*int(inputsize/4)),

                nn.Linear(inputsize*int(inputsize/8),1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),

                nn.Linear(1024,512),
                nn.ReLU(),
                nn.BatchNorm1d(512),

                nn.Linear(512,bits_len),
                nn.ReLU(),
                nn.BatchNorm1d(bits_len),
            )

    def forward(self,input):           
        #full connect network
        input=input.view(input.size()[0],-1)
        output=self.fc1(input)
        output=F.log_softmax(output)
        
        return output


