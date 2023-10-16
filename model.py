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
from utils import NTXentLoss_time

from utils import ST_BLOCK_5 #OGCRNN
from utils import multi_gcn #gwnet
"""
the parameters:
x-> [batch_num,in_channels,num_nodes,tem_size],
"""
def rand_mask(x,alpha,self):
    rand1=torch.rand(x.shape).to(x.device)
    mask=(rand1>alpha).int().float()
    x=x*mask
    return x

class OGCRNN(nn.Module):      
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(OGCRNN,self).__init__()
       
        self.block1=ST_BLOCK_5(in_dim,dilation_channels,num_nodes,length,K,Kt)
        
        self.tem_size=length
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1,length),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
    def forward(self,input):
        x=input
        if self.training:
            x=rand_mask(x,0.2,self)
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A=F.dropout(A,0.5)
        x=self.block1(x,A)
        
        x=self.conv1(x)
        return x,A,A   

class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1,out_dim=12,residual_channels=32,
                 dilation_channels=32,skip_channels=256,
                 end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        
        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len +=1
            



        
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                
                new_dilation *=2
                receptive_field += additional_scope
                
                additional_scope *= 2
                
                self.gconv.append(multi_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1=BatchNorm2d(in_dim,affine=False)

    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        if self.training==True:
            x=rand_mask(x,0.8,self)
       
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)           

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x,adp,adp


class COGCN(nn.Module):
    def __init__(self,device, num_nodes, CL,l,dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(COGCN,self).__init__()
        tem_size=length
        self.num_nodes=num_nodes
        self.CL=CL
        self.l=l
        
        self.block = nn.ModuleList()
        self.block1=ST_BLOCK_7(in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        for i in range(l-1):
            self.block.append(ST_BLOCK_7(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt))
        
        
        self.conv1=Conv2d(2*dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.projection_head=nn.Sequential(
            Conv2d(dilation_channels,dilation_channels//4,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True),
            nn.BatchNorm2d(dilation_channels//4),
            nn.ReLU(inplace=True),
            Conv2d(dilation_channels//4,dilation_channels,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True),
            
            )
        

        self.loss=NTXentLoss(device,64,0.05,True)
        self.loss1=NTXentLoss_time(device,64,0.05,True) #04:0.3 08:0.05
        self.mae=nn.MSELoss(reduction='mean')
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h1=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h1, a=0, b=0.0001)
        
        self.bn_1=BatchNorm2d(in_dim,affine=False)
        
    def forward(self,input):
        #input: batch_size x channels x nodes x time_length
        shape=input.shape
        
        input1=(input)
        input1 = F.normalize(input1, p=2, dim=-1).reshape(shape[0],-1).contiguous()
        time_scores=torch.matmul(input1,input1.permute(1,0)) #nodes x batch_size x batch_size
        time_scores=F.normalize(time_scores, p=2, dim=-1)
        diag=(1-torch.eye(shape[0])).cuda()
        #diag1=(torch.eye(shape[0])).cuda()
        time_scores=time_scores*diag
        
        k=1
        mask_neg=0
        if k>0:
            _,index=torch.topk(time_scores, k=k, dim=-1, largest=True, sorted=True)
            #_,index1=torch.topk(time_scores1, k=shape[0]-k-1, dim=-1, largest=False, sorted=False)
            mask_neg=F.one_hot(index,shape[0]).sum(1)
        
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
        for i in range(self.l-1):
            v=self.block[i](v,A)
        
        z=self.block1(x,B)
        for i in range(self.l-1):
            z=self.block[i](z,B)
        
        loss1=0.0 

        if self.training and self.CL=='True':
            p=(self.projection_head(v))
            p1=(self.projection_head(z))
            if k==0:
                loss1=self.loss(p,p1,mask_neg)
                
            else:
                loss1=self.loss1(p,p1,mask_neg,k)
                
        x=torch.cat((v,z),1)
        x=self.conv1(x)
        return x,loss1,A        


