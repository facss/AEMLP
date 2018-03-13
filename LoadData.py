#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms
from args import get_parser#import args

mp=get_parser()
opts=mp.parse_args()

#orl face dataset
class FaceData(Dataset):
    def __init__(self,arnoldimglist,class_number,mode,transform=None,should_invert=True):
        super(FaceData,self).__init__()
        self.arnoldimglist=arnoldimglist
        self.class_number=class_number
        self.mode=mode#switch mode
        self.imglist=self.loadimglist()
        self.should_invert=should_invert
        self.transform=transform
          
    def loadimglist(self):
        #return image name list
        img_list=list()
        if self.mode=='mlptrain':
            for i in range(self.class_number):#遍历所有的类别
                for j in  range(opts.train_number):
                    img_list.append(self.arnoldimglist[i*opts.all_number+j])
        elif self.mode=='mlpvalidate':
            for i in range(self.class_number):#遍历所有的类别
                for j in  range(opts.train_number,opts.validate_number):
                    img_list.append(self.arnoldimglist[i*opts.all_number+j])
        elif self.mode=='mlptest':
            for i in range(self.class_number):#遍历所有的类别
                for j in  range(opts.validate_number,opts.test_number):
                    img_list.append(self.arnoldimglist[i*opts.all_number+j])
        else:   #autoencoder 
            img_list=self.arnoldimglist
        
        return img_list
    
    def __getitem__(self,item):

        img=self.imglist[item][0]
        label=self.imglist[item][1]   
        if self.transform is not None:
            img=self.transform(img)
        return [img,label]#img is a torch.tensor对象
        
    def __len__(self):
        return len(self.imglist)
