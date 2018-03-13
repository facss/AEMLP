#! /usr/bin/env python
#-*- coding:UTF-8 -*-
import time
import torch
import shutil
import numpy as np 
import torch.nn as nn
import torchvision 
import torch.utils.data 
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F 
from args import get_parser

#================================================
myparser=get_parser()
opts=myparser.parse_args()
#================================================

class AverageMeter(object):
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count

def save_checkpoint(state,is_best,filename='checkpoint.path.tar'):
    """save file to checkpoint"""
    torch.save(state,filename)
    if is_best:
        shutil.copy(filename,opts.model_path)

def accuracy(output, label):
    #top 1 precision
    """Computes the precision@k for the specified values of k"""
    count=0
    reses=np.argmax(output,axis=1)
    lengths=output.shape[0]
    for i in range(lengths):
        if reses[i] == label[i]:
            count=count+1
    
    result=count/lengths
    return result

class Counter(object):
    def __init__(self):
        super(Counter,self).__init__()
        self.iteration_number=0
        self.counter=list()
        self.loss_history=list()
    def myadd(self,loss):
        self.iteration_number+=10
        self.counter.append(self.iteration_number)
        self.loss_history.append(loss)

def mlptrain(AEModel,best_val,MLPModel,MLP_train_loader,MLP_val_loader,Optimizer):
    #train for one epoch
    mycount=Counter()
    for epoch in range(opts.MLP_Start_epoch,opts.MLP_train_number_epochs):       
        mlptrain_epoch(AEModel, mycount,epoch,MLPModel,MLP_train_loader,Optimizer)

        if (epoch+1)%opts.valfre ==0 and epoch !=0:#save the model
            val_res=mlpvalidate(AEModel,MLP_val_loader,MLPModel)
            #save the best model
            is_best=val_res>best_val
            best_val=max(val_res,best_val)
            save_checkpoint({
                'epoch':epoch+1,
                'state_dict':MLPModel.state_dict(),
                'best_val':best_val,
                'optimizer':Optimizer,
                'curr_val':val_res,
            },is_best)
            print('** Validation : %f (best) '%(best_val))

    return mycount
        
def mlptrain_epoch(AEModel,mycount,epoch,MLPModel,train_loader,Optimizer):
    #switch to train model
    MLPModel.train()
    AEModel.eval()
    
    for i,traindata in enumerate(train_loader,0):
        #measure data loading time       
        [img,label]=traindata#load original training data
        opts.cuda=torch.cuda.is_available()
        if opts.cuda:
            img,label=Variable(img).cuda(),Variable(label).cuda()
        
        #load data into models     
        inputimg=img.view(img.size()[0],-1)  
        encoded,decoded=AEModel(inputimg)
        output=MLPModel(encoded)
        loss = F.cross_entropy(output,label)  

        Optimizer.zero_grad()   
        loss.backward()
        Optimizer.step()
        
        if i%1000 ==0:
            print('Epochs number {}\n Current loss:{}\n'.format(epoch,loss.data[0]))
            mycount.myadd(loss.data[0])

def mlpvalidate(AEModel,val_loader,MLPModel):
    top1=AverageMeter()
    losses=AverageMeter()
    
    #switch to evaluate mode
    MLPModel.eval()
    AEModel.eval()

    for i,valdata in enumerate(val_loader,0):
        input_var=list()
        [img,label]=valdata
        opts.cuda=torch.cuda.is_available()
        img_var,label_var=Variable(img,volatile=True),Variable(label,volatile=True)
        if opts.cuda:
            img_var,label_var=img_var.cuda(),label_var.cuda()
        
        img_var=img_var.view(img_var.size()[0],-1)
        encoded,decoded=AEModel(img_var)
        output=MLPModel(encoded)  
        #use euclidean distance of two image             
        loss_=F.cross_entropy(output,label_var)
        losses.update(loss_.data[0], img.size(0))

        prec1=accuracy(output.cpu().data.numpy(),label.numpy())
        top1.update(prec1, img.size(0))
    print('Prec@1 {0:.2f}\t validation Loss {1:.2f}'.format(float(top1.avg), float(losses.avg)))
    return top1.avg


def aetrain(Model,train_loader,Optimizer):
    #train for one epoch
    for epoch in range(0,opts.AE_train_number_epochs):       
        aetrain_epoch(epoch,Model,train_loader,Optimizer)
        
def aetrain_epoch(epoch,Model,train_loader,Optimizer):
    #switch to train model
    Model.train()
    
    for i,traindata in enumerate(train_loader,0):
        #measure data loading time       
        [img,_]=traindata #load original training data
       
        opts.cuda=torch.cuda.is_available()
        if opts.cuda:
            img=Variable(img).cuda()
        
        #load data into models
        inputimg=img.view(img.size()[0],-1)
       
        encoded,decoded = Model(inputimg)
        loss_func=nn.MSELoss()
        loss=loss_func(decoded,inputimg)

        Optimizer.zero_grad()
        
        loss.backward()
        Optimizer.step()
        if i % 1000 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])
        
