 # -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:20:23 2018
 
@author: gk
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, Conv3d, ModuleList, Parameter, LayerNorm, BatchNorm1d, BatchNorm3d


"""
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

class cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(cheby_conv,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 


    
class ST_BLOCK_7(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_7,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.gcn=cheby_conv(c_out,2*c_out,K,1)
        
        self.c_out=c_out
        self.conv_1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
    def forward(self,x,supports):
        x_input1=self.conv_1(x)
        x1=self.conv1(x)   
        x2=self.gcn(x1,supports)
        filter,gate=torch.split(x2,[self.c_out,self.c_out],1)
        x=(filter+x_input1)*torch.sigmoid(gate)
        return x    
    
    

        
#**Siam

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cos):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        
    def forward(self, zis, zjs, adj):
        shape=zis.shape
        zis=zis.reshape(shape[0],-1)
        zjs=zjs.reshape(shape[0],-1)
        
        zis1 = F.normalize(zis, p=2, dim=-1)
        zjs1 = F.normalize(zjs, p=2, dim=-1)
        
        similarity_matrix = torch.matmul(zis1,zjs1.permute(1,0))
        
        shape=similarity_matrix.shape
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix)
        positives = l_pos#.view(shape[0],self.batch_size, 1)
        positives = positives/self.temperature
        
        diag = np.eye(self.batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask).type(torch.bool).cuda()
        negatives = similarity_matrix[mask].view(self.batch_size,self.batch_size-1)
        negatives =negatives /self.temperature
        
        loss=-torch.log((torch.exp(positives))/(torch.exp(positives)+torch.sum(torch.exp(negatives),-1,True)))
        return loss.sum()/(shape[0])


    

class NTXentLoss1(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cos):
        super(NTXentLoss1, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        
    def forward(self, zis, zjs, adj):
        shape=zis.shape
        zis=zis.permute(2,0,1,3).contiguous().reshape(shape[2],shape[0],-1)
        zjs=zjs.permute(2,0,1,3).contiguous().reshape(shape[2],shape[0],-1)
        zis1 = F.normalize(zis, p=2, dim=-1)
        zjs1 = F.normalize(zjs, p=2, dim=-1)
        similarity_matrix = torch.matmul(zis1,zjs1.permute(0,2,1))#torch.sqrt(F.relu(2-2*torch.matmul(zis1,zjs1.permute(0,2,1))))
        #similarity_matrix1 = torch.matmul(zis1,zis1.permute(0,2,1))
        shape=similarity_matrix.shape
        # filter out the scores from the positive samples
        l_pos = torch.diagonal(similarity_matrix,dim1=1,dim2=2)
        positives = l_pos.view(shape[0],self.batch_size, 1)#torch.cat([l_pos, r_pos]).view(shape[0],2 * self.batch_size, 1)
        positives /= self.temperature
        #positives =torch.einsum('mn,nbl->mbl',adj,positives) #spatial siam
        
        diag = np.eye(self.batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask).type(torch.bool).cuda()
        negatives = similarity_matrix[:,mask].view(shape[0],self.batch_size,self.batch_size-1)
        negatives /= self.temperature
        
        loss=-torch.log((torch.exp(positives))/(torch.exp(positives)+torch.sum(torch.exp(negatives),-1,True)))
        return loss.sum()/(shape[0]*shape[1])

