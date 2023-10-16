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

##gwnet
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2)
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class multi_gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(multi_gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


###GRCN(ICLR)
class gcn_conv_hop(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ] - input of one single time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : gcn_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(gcn_conv_hop,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv1d(c_in_new, c_out, kernel_size=1,
                          stride=1, bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode  = x.shape
        
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
        x = torch.einsum('bcn,knq->bckq', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode)
        out = self.conv1(x)
        return out 



class ST_BLOCK_5(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_5,self).__init__()
        self.gcn_conv=gcn_conv_hop(c_out+c_in,c_out*4,K,1)
        self.c_out=c_out
        self.tem_size=tem_size
        
        
    def forward(self,x,supports):
        shape = x.shape
        h = Variable(torch.zeros((shape[0],self.c_out,shape[2]))).cuda()
        c = Variable(torch.zeros((shape[0],self.c_out,shape[2]))).cuda()
        out=[]
        
        for k in range(self.tem_size):
            input1=x[:,:,:,k]
            tem1=torch.cat((input1,h),1)
            fea1=self.gcn_conv(tem1,supports)
            i,j,f,o = torch.split(fea1, [self.c_out, self.c_out, self.c_out, self.c_out], 1)
            new_c=c*torch.sigmoid(f)+torch.sigmoid(i)*torch.tanh(j)
            new_h=torch.tanh(new_c)*(torch.sigmoid(o))
            c=new_c
            h=new_h
            out.append(new_h)
        x=torch.stack(out,-1)
        return x 

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



    
    
class nconv_batch(nn.Module):
    def __init__(self):
        super(nconv_batch,self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2)
        #try:
       #     x = torch.einsum('ncvl,vw->ncwl',(x,A))
        #except:
        x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        return x.contiguous()
    
class linear_time(nn.Module):
    def __init__(self,c_in,c_out,Kt):
        super(linear_time,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class multi_gcn_time(nn.Module):
    def __init__(self,c_in,c_out,Kt,dropout,support_len=3,order=2):
        super(multi_gcn_time,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear_time(c_in,c_out,Kt)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h



class multi_gcn_batch(nn.Module):
    def __init__(self,c_in,c_out,Kt,dropout,support_len=3,order=2):
        super(multi_gcn_batch,self).__init__()
        self.nconv = nconv_batch()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear_time(c_in,c_out,Kt)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:            
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gate(nn.Module):
    def __init__(self,c_in):
        super(gate,self).__init__()
        self.conv1=Conv2d(c_in, c_in//2, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
        
        
    def forward(self,seq,seq_cluster):
        
        #x=torch.cat((seq_cluster,seq),1)     
        #gate=torch.sigmoid(self.conv1(x)) 
        out=torch.cat((seq,(seq_cluster)),1)    
        
        return out
class gate1(nn.Module):
    def __init__(self,c_in):
        super(gate1,self).__init__()
        self.conv1=Conv2d(c_in, c_in//2, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
        
        
    def forward(self,seq,seq_cluster):
        
        x=torch.cat((seq_cluster,seq),1)     
        gate=torch.sigmoid(self.conv1(x)) 
        out=seq*gate+seq_cluster*(1-gate)    
        
        return out
           
    
class Transmit(nn.Module):
    def __init__(self,c_in,tem_size,transmit,num_nodes,cluster_nodes):
        super(Transmit,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(tem_size, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(tem_size,c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(num_nodes,cluster_nodes), requires_grad=True)
        self.c_in=c_in
        self.transmit=transmit
        
    def forward(self,seq,seq_cluster):
        
        c1 = seq
        f1 = self.conv1(c1).squeeze(1)#b,n,l
        
        c2 = seq_cluster.permute(0,3,1,2)#b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)#b,c,n
        logits=torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        a = torch.mean(logits, 1, True)
        logits = logits - a
        logits = torch.sigmoid(logits)
        
        coefs = (logits)*self.transmit
        return coefs    

class T_cheby_conv_ds_1(nn.Module):
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
        super(T_cheby_conv_ds_1,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).repeat(nSample,1,1).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 
    
class dynamic_adj(nn.Module):
    def __init__(self,c_in,num_nodes):
        super(dynamic_adj,self).__init__()
        
        self.SATT=SATT_pool(c_in,num_nodes)
        self.LSTM=nn.LSTM(num_nodes,num_nodes,batch_first=True)#b*n,l,c
    def forward(self,x):
        S_coef=self.SATT(x)        
        shape=S_coef.shape
        h = Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        c=Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        hidden=(h,c)
        S_coef=S_coef.permute(0,2,1,3).contiguous().view(shape[0]*shape[2],shape[1],shape[3])
        S_coef=F.dropout(S_coef,0.5,self.training) #2020/3/28/22:17,试验下效果
        _,hidden=self.LSTM(S_coef,hidden)
        adj_out=hidden[0].squeeze().view(shape[0],shape[2],shape[3]).contiguous()
        
        return adj_out
    
    

class TATT_1(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(TATT_1,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(num_nodes, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(num_nodes,c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(tem_size,tem_size), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(tem_size,tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn=BatchNorm1d(tem_size)
        
    def forward(self,seq):
        c1 = seq.permute(0,1,3,2)#b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()#b,l,n
        
        c2 = seq.permute(0,2,1,3)#b,c,n,l->b,n,c,l
        #print(c2.shape)
        f2 = self.conv2(c2).squeeze()#b,c,n
         
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)
        logits = logits.permute(0,2,1).contiguous()
        logits=self.bn(logits).permute(0,2,1).contiguous()
        coefs = torch.softmax(logits,-1)
        return coefs   
           
class GCNPool(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool,self).__init__()
        self.time_conv=Conv2d(c_in, 2*c_out, kernel_size=(1, Kt),padding=(0,0),
                          stride=(1,1), bias=True,dilation=2)
        
        self.multigcn=multi_gcn_time(c_out,2*c_out,Kt,dropout,support_len,order)
        
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
        
        
    
    def forward(self,x,support):
        residual = self.conv1(x)
        
        x=self.time_conv(x)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        
        
        x=self.multigcn(x,support)        
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2)) 
        #x=F.dropout(x,0.3,self.training)
        
        T_coef=self.TAT(x)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)
        out=self.bn(x+residual[:, :, :, -x.size(3):])
        return out
    
    
###cogcn
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
        
    def forward(self, zis, zjs, mask):
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

class NTXentLoss_time(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cos):
        super(NTXentLoss_time, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        
        
    def forward(self, zis, zjs, mask,k):
        shape=zis.shape
        zis=zis.reshape(shape[0],-1)
        zjs=zjs.reshape(shape[0],-1)
        
        zis1 = F.normalize(zis, p=2, dim=-1)
        zjs1 = F.normalize(zjs, p=2, dim=-1)
        
        similarity_matrix = torch.matmul(zis1,zjs1.permute(1,0))
        
        shape=similarity_matrix.shape
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix)
        positives = l_pos.view(self.batch_size, 1)
        positives = positives/self.temperature
        
        diag = np.eye(self.batch_size)
        mask1 = torch.from_numpy((diag)).cuda()+mask
        mask1 = (1 - mask1).type(torch.bool)
        negatives = similarity_matrix[mask1].view(self.batch_size,self.batch_size-1-k)
        negatives =negatives /self.temperature
        
        loss=-torch.log((torch.exp(positives))/(\
                        torch.exp(positives)+torch.sum(torch.exp(negatives),-1,True)))
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


    

    
class NTXentLoss2(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cos):
        super(NTXentLoss2, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        
    def forward(self, zis, zjs, mask_time, k):
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
        positives1 = l_pos.view(shape[0],self.batch_size, 1)#torch.cat([l_pos, r_pos]).view(shape[0],2 * self.batch_size, 1)
        positives2 = similarity_matrix[mask_time].view(shape[0],self.batch_size,k)
        positives = torch.stack((positives1,positives2),-1)
        positives /= self.temperature
        #positives =torch.einsum('mn,nbl->mbl',adj,positives) #spatial siam
        
        diag = np.eye(self.batch_size)
        mask1 = torch.from_numpy((diag))
        mask1 = (1 - mask1).type(torch.bool).cuda()
        mask = mask1+mask_time
        negatives = similarity_matrix[mask].view(shape[0],self.batch_size,self.batch_size-1-k)
        negatives /= self.temperature
        
        loss=-torch.log((torch.exp(positives))/(torch.exp(positives)+torch.sum(torch.exp(negatives),-1,True)))
        return loss.sum()/(shape[0]*shape[1])