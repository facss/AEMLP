#! /usr/bin/env python
#-*- coding:UTF-8 -*-
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
import os 
from PIL import Image,ImageDraw
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot,savefig

class plotfunc(object):
    def __init__(self,iteration,loss):
        super(plotfunc,self).__init__()
        self.iteration=iteration
        self.loss=loss 
        
    def imshow(self,img,text,should_save=False):
        npimg=img.numpy()
        plt.axis("off")
        if text:
            plt.text(75,8,text,style='italic',fontweight='bold',bbox={'facecolor':'white','alpha':0.8,'pad':10})
            plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.show()

    def show_plot(self):
        plt.figure
        plt.plot(self.iteration,self.loss)
        plt.show()

    def show_TSNE(self,X_tsne,label):
        '''show two dimension TSNE'''
        plt.figure
        plt.scatter(X_tsne[:,0],X_tsne[:,1],c=label)
        plt.show()

class  TraverseDataset(object):
    def __init__(self,dataset_dir,n,a=1,b=1):
        super(TraverseDataset,self).__init__()
        self.dataset_dir=dataset_dir
        self.n=n
        self.a=1
        self.b=1
    
    def preprocessORLDataset(self):
        """
        Args:
            dataset:dataset directory
            n:number of arnold transforms 
        Returns:
            image list:[class 1 (10 per class ),class 2(10 per class),...,class n( 10 per class )]
            class_numbner :40
        """
        img_list=list()
        tmp_imglist=list()
        for dirpath,dirnames,filenames in os.walk(self.dataset_dir):
            for d in dirnames:
                subpath=os.path.join(self.dataset_dir,d)
                classname=int(d)-1#子文件夹就是类别名称,这个后续还得改
                class_number=len(dirnames)#类别的总数
                for subdirpath,subdirnames,subfilenames in os.walk(subpath):
                    for f in subfilenames:#获得子文件夹下所有的文件
                        img_name=os.path.join(subpath,f)#image name                      
                        name_class=(img_name,classname)#组成一个tuple类型，包含了(文件绝对路径名，类别名)
                        tmp_imglist.append(name_class)                     
        sorted_list=sorted(tmp_imglist)#sort the image name list
        for imagename,classname in sorted_list:
            img=Image.open(imagename)#PIL image
            arnold_img=self.ArnoldTransform(img)
            print("arnold transforms : ",imagename,"-> class : ",classname)
            img_list.append([arnold_img,classname])
        #print(img_list[0])
        return img_list,class_number#

    def preprocessCMUPIEDataset(self,classnumber,person_eachclass):
        """
        Args:
            Pose05_64x64_files:68个人，每个人49张，总的3332
            Pose07_64x64_files:68个人，每个人24张，总的1632，
            Pose09_64x64_files:68个人，每个人24张，总的1632
            Pose27_64x64_files:68个人，每个人49张，总的3332，
            Pose29_64x64_files:68个人，每个人24张，总的1632
        Returns:
            imglist:[class 1 (10 per class ),class 2(10 per class),...,class n( 10 per class )]
            classnumber:68
        """
        img_list1=list()
        img_list2=list()
        img_list3=list()
        img_list4=list()
        img_list5=list()
        img_list=list()

        for dirpath,dirnames,filenames in os.walk(self.dataset_dir):#获得根目录下的文件
            for d in dirnames:#子文件夹，包含不同的姿势的命名
                subpath=os.path.join(self.dataset_dir,d)#构造出路径
                tmp_namelist=list()
                for subdirpath,subdirnames,subfilenames in os.walk(subpath):#获得每个子文件夹里所有的图片数据
                    for f in subfilenames:#遍历每个子文件下的所有数据
                        img_name=os.path.join(subpath,f)#获得每个子文件夹下的图片名
                        tmp_namelist.append(img_name)
          
                    for i in range(classnumber):#68个class
                        for j in range(person_eachclass):#每个class人是20张图
                            if "Pose05_64x64_files" in subpath:
                                imgname=tmp_namelist[i*49+j]
                                img=Image.open(imgname)
                                img=self.ArnoldTransform(img)
                                print("arnold transforms : ",imgname,"-> class : ",i)
                                img_list1.append((img,i))   #tuple(image,class)               
                            elif "Pose07_64x64_files" in subpath:
                                imgname=tmp_namelist[i*24+j]
                                img=Image.open(imgname)
                                img=self.ArnoldTransform(img)
                                print("arnold transforms : ",imgname,"-> class : ",i)
                                img_list2.append((img,i))    
                            elif "Pose09_64x64_files" in subpath:
                                imgname=tmp_namelist[i*24+j]
                                img=Image.open(imgname)
                                img=self.ArnoldTransform(img)
                                print("arnold transforms : ",imgname,"-> class : ",i)
                                img_list3.append((img,i))    
                            elif "Pose27_64x64_files" in subpath:
                                imgname=tmp_namelist[i*49+j]
                                img=Image.open(imgname)
                                img=self.ArnoldTransform(img)
                                print("arnold transforms : ",imgname,"-> class : ",i)
                                img_list4.append((img,i))    
                            else:   
                                imgname=tmp_namelist[i*24+j]
                                img=Image.open(imgname)  
                                img=self.ArnoldTransform(img)  
                                print("arnold transforms : ",imgname,"-> class : ",i)                  
                                img_list5.append((img,i))   

        #重新进行调整，每个人有100张图，总的有35个人
        for ii in range(classnumber):#0-35
            for jj in range(person_eachclass):
            #img_list1[0:19,20:39:40:59,...,680:699]
                img_list.append(img_list1[20*ii+jj ])
                img_list.append(img_list2[20*ii+jj ])
                img_list.append(img_list3[20*ii+jj ])
                img_list.append(img_list4[20*ii+jj ])
                img_list.append(img_list5[20*ii+jj ])

        img_list1=[]
        img_list2=[]
        img_list3=[]
        img_list4=[]
        img_list5=[]
                
        return img_list,classnumber  

    def preprocessPUBFIGDataset(self):
        """
        Args:
            dataset:dataset directory
            n:number of arnold transforms 
        Returns:
            image list:[class 1 (10 per class ),class 2(10 per class),...,class n( 10 per class )]
            class_numbner :40
        """
        classname=-1    
        img_list=list()
        tmp_imglist=list()
        for dirpath,dirnames,filenames in os.walk(self.dataset_dir):
            for d in dirnames:
                file_number=-1
                subpath=os.path.join(self.dataset_dir,d)
                classname=classname+1#子文件夹就是类别名称,这个后续还得改
                class_number=len(dirnames)#类别的总数
                for subdirpath,subdirnames,subfilenames in os.walk(subpath):
                    for f in subfilenames:#获得子文件夹下所有的文件
                        file_number=file_number+1
                        if file_number<10:#控制每个类中的个数为50张
                            img_name=os.path.join(subpath,f)#image name                      
                            name_class=(img_name,classname)#组成一个tuple类型，包含了(文件绝对路径名，类别名)
                            tmp_imglist.append(name_class)                     
        sorted_list=sorted(tmp_imglist)#sort the image name list

        for imagename,classname in sorted_list:
            img=Image.open(imagename)#PIL image
            arnold_img=self.ArnoldTransform(img)
            print("arnold transforms : ",imagename,"-> class : ",classname)
            img_list.append([arnold_img,classname])
       
        return img_list,class_number#   

    def ArnoldTransform(self,img):
        a=self.a 
        b=self.b 
        n=self.n 
        #cpu model
        width,height=img.size#获得输入图像的大小(宽度，高度)
        if width<height:
            N=width
        else:
            N=height
        #接着对图片进行设置长宽
        img = img.resize((N, N), Image.ANTIALIAS)
        img=img.convert('L')  # 把输入的图片转化为灰度图
        image=Image.new('L',(N,N),(255))#空白的图
        draw=ImageDraw.Draw(image)

        #填充每个像素
        for inc in range(n):
            for y in range(N):
                for x in range(N):
                    xx=(x+b*y)%N
                    yy=(a*x+(a*b+1)*y)%N
                    temp=img.getpixel((x,y))
                    draw.point((xx,yy),fill=img.getpixel((x,y)))       
            img=image
        
        return image
        

