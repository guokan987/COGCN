import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import torch.autograd as autograd
 
import numpy as np
import math
import random
from torch.nn import BatchNorm2d, BatchNorm1d, Conv1d, Conv2d, ModuleList, Parameter,LayerNorm,InstanceNorm2d, Dropout2d
import util

from utils import ST_BLOCK_7
from utils import NTXentLoss
from utils import NTXentLoss1

class COGCN(nn.Module):
    def __init__(self,device, num_nodes, CL,dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(COGCN,self).__init__()
        tem_size=length
        self.num_nodes=num_nodes
        self.block1=ST_BLOCK_7(in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_7(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_7(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        self.conv1=Conv2d(2*dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        
        self.projection_head=nn.Sequential(
            Conv2d(dilation_channels,dilation_channels//4,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True),
            nn.BatchNorm2d(dilation_channels//4),
            nn.ReLU(inplace=True),
            Conv2d(dilation_channels//4,dilation_channels,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True),
            
            )
        
        self.loss=NTXentLoss(device,64,0.05,True)
        #self.loss1=NTXentLoss1(device,64,0.1,True)
        
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h1=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h1, a=0, b=0.0001)
        self.CL=CL
    def forward(self,input):
        x=input
        mask=(self.supports[0]!=0).float()
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A=F.dropout(A,0.3,self.training)
        
        mask=(self.supports[0]!=0).float()
        B=self.h*mask
        d1=1/(torch.sum(B,-1)+0.0001)
        D1=torch.diag_embed(d1)
        B=torch.matmul(D1,B)
        B=F.dropout(B,0.3,self.training)
        v=self.block1(x,A)
        v=self.block2(v,A)
        v=self.block3(v,A)
        
        z=self.block1(x,B)
        z=self.block2(z,B)
        z=self.block3(z,B)
        
        loss1=0 
        
        if self.training and self.CL=='True':
            p=(self.projection_head(v))
            p1=(self.projection_head(z))
            loss1=self.loss(p,p1,self.supports[0])
        
        x=torch.cat((v,z),1)
        x=self.conv1(x)
        return x,loss1,B     


