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

from utils import NTXentLoss,NTXentLoss_time
from utils import NTXentLoss1

from utils import ST_BLOCK_7 #COGCN
from utils import ST_BLOCK_5 #OGCRNN
from utils import multi_gcn #gwnet
from utils import GCNPool #H_GCN
from utils import Transmit
from utils import gate
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
        x=self.bn(input)
        if self.training==True:
            x=rand_mask(x,0.8,self)
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
        input=self.bn_1(input)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        if self.training==True:
            x=rand_mask(x,1.0,self)
       
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
    

class H_GCN(nn.Module):
    def __init__(self,device, num_nodes, cluster_nodes,dropout=0.3, supports=None,supports_cluster=None,transmit=None,length=12, 
                 in_dim=1,in_dim_cluster=3,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(H_GCN, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.transmit=transmit
        self.cluster_nodes=cluster_nodes
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.start_conv_cluster = nn.Conv2d(in_channels=in_dim_cluster,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.supports_cluster = supports_cluster
        
        self.supports_len = 0
        self.supports_len_cluster = 0
        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_cluster+=len(supports_cluster)

        
        if supports is None:
            self.supports = []
            self.supports_cluster = []
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster=Parameter(torch.zeros(cluster_nodes,cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.supports_len +=1
        self.supports_len_cluster +=1
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)  
        self.nodevec1_c = nn.Parameter(torch.randn(cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_c = nn.Parameter(torch.randn(10,cluster_nodes).to(device), requires_grad=True).to(device)  
        
        
        self.block1=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block2=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.block_cluster1=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-6,3,dropout,cluster_nodes,
                            self.supports_len)
        self.block_cluster2=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-9,2,dropout,cluster_nodes,
                            self.supports_len)
        
        self.skip_conv1=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        
        
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.conv_cluster1=Conv2d(dilation_channels,out_dim,kernel_size=(1,3),
                          stride=(1,1), bias=True)
        self.bn_cluster=BatchNorm2d(in_dim_cluster,affine=False)
        self.gate1=gate(2*dilation_channels)
        self.gate2=gate(2*dilation_channels)
        self.gate3=gate(2*dilation_channels)
        
        self.transmit1=Transmit(dilation_channels,length,transmit,num_nodes,cluster_nodes)
        self.transmit2=Transmit(dilation_channels,length-6,transmit,num_nodes,cluster_nodes)
        self.transmit3=Transmit(dilation_channels,length-9,transmit,num_nodes,cluster_nodes)
       

    def forward(self, input, input_cluster):
        x=self.bn(input)
        shape=x.shape
        input_c=input_cluster
        x_cluster=self.bn_cluster(input_c)
        if self.supports is not None:
            #nodes
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d=1/(torch.sum(A,-1))
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            
            new_supports = self.supports + [A]
            #region
            A_cluster=F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
            d_c=1/(torch.sum(A_cluster,-1))
            D_c=torch.diag_embed(d_c)
            A_cluster=torch.matmul(D_c,A_cluster)
            
            new_supports_cluster = self.supports_cluster + [A_cluster]
        
        #network
        transmit=self.transmit              
        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)
        transmit1 = self.transmit1(x,x_cluster)
        x_1=(torch.einsum('bmn,bcnl->bcml',transmit1,x_cluster))   
        
        x=self.gate1(x,x_1)
        
       
        skip=0
        skip_c=0
        #1
        x_cluster=self.block_cluster1(x_cluster,new_supports_cluster) 
        x=self.block1(x,new_supports)   
        transmit2 = self.transmit2(x,x_cluster)
        x_2=(torch.einsum('bmn,bcnl->bcml',transmit2,x_cluster)) 
        
        x=self.gate2(x,x_2) 
        
        
        s1=self.skip_conv1(x)
        skip=s1+skip 
        
       
        #2       
        x_cluster=self.block_cluster2(x_cluster,new_supports_cluster)
        x=self.block2(x,new_supports) 
        transmit3 = self.transmit3(x,x_cluster)
        x_3=(torch.einsum('bmn,bcnl->bcml',transmit3,x_cluster)) 
        
        x=self.gate3(x,x_3)
           
        
        s2=self.skip_conv2(x)      
        skip = skip[:, :, :,  -s2.size(3):]
        skip = s2 + skip        
        
       
        
        #output
        x = F.relu(skip)      
        x = F.relu(self.end_conv_1(x))            
        x = self.end_conv_2(x)              
        return x,transmit3,A
    
    
class COHGCN(nn.Module):
    def __init__(self,device, num_nodes, cluster_nodes,CL,dropout=0.3, supports=None,supports_cluster=None,transmit=None,length=12, 
                 in_dim=1,in_dim_cluster=3,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(COHGCN, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.transmit=transmit
        self.cluster_nodes=cluster_nodes
        
        self.CL=CL
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.start_conv_cluster = nn.Conv2d(in_channels=in_dim_cluster,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.supports_cluster = supports_cluster
        
        self.supports_len = 0
        self.supports_len_cluster = 0
        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_cluster+=len(supports_cluster)

        
        if supports is None:
            self.supports = []
            self.supports_cluster = []
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster=Parameter(torch.zeros(cluster_nodes,cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.supports_len +=1
        self.supports_len_cluster +=1
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)  
        self.nodevec1_c = nn.Parameter(torch.randn(cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_c = nn.Parameter(torch.randn(10,cluster_nodes).to(device), requires_grad=True).to(device)  
        
        
        self.block1=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block2=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.block_cluster1=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-6,3,dropout,cluster_nodes,
                            self.supports_len)
        self.block_cluster2=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-9,2,dropout,cluster_nodes,
                            self.supports_len)
        
        self.skip_conv1=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        
        self.end_conv_1 = nn.Conv2d(in_channels=2*skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        
        
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.conv_cluster1=Conv2d(dilation_channels,out_dim,kernel_size=(1,3),
                          stride=(1,1), bias=True)
        self.bn_cluster=BatchNorm2d(in_dim_cluster,affine=False)
        self.gate1=gate(2*dilation_channels)
        self.gate2=gate(2*dilation_channels)
        self.gate3=gate(2*dilation_channels)
        
        self.transmit1=Transmit(dilation_channels,length,transmit,num_nodes,cluster_nodes)
        self.transmit2=Transmit(dilation_channels,length-6,transmit,num_nodes,cluster_nodes)
        self.transmit3=Transmit(dilation_channels,length-9,transmit,num_nodes,cluster_nodes)
       
        self.loss=NTXentLoss(device,64,0.1,True)
        self.loss1=NTXentLoss_time(device,64,0.1,True)
        self.projection_head=nn.Sequential(
            Conv2d(skip_channels,skip_channels//4,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True),
            nn.BatchNorm2d(skip_channels//4),
            nn.ReLU(inplace=True),
            Conv2d(skip_channels//4,skip_channels,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True),
            
            )
        
        
    def forward(self, input, input_cluster):
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
        
        
        x=self.bn(input)
        shape=x.shape
        input_c=input_cluster
        x_cluster=self.bn_cluster(input_c)
        if self.supports is not None:
            #nodes
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))*self.supports[0]
            d=1/(torch.sum(A,-1)+0.000001)
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            
            new_supports1 = self.supports + [A]
            
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))+self.supports[0]
            d=1/(torch.sum(A,-1)+0.000001)
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            
            new_supports2 = self.supports + [A]
            
            #region
            A_cluster=F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))*self.supports_cluster[0]
            d_c=1/(torch.sum(A_cluster,-1)+0.000001)
            D_c=torch.diag_embed(d_c)
            A_cluster=torch.matmul(D_c,A_cluster)
            
            new_supports_cluster1 = self.supports_cluster + [A_cluster]
            
            A_cluster=F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))+self.supports_cluster[0]
            d_c=1/(torch.sum(A_cluster,-1)+0.000001)
            D_c=torch.diag_embed(d_c)
            A_cluster=torch.matmul(D_c,A_cluster)
            
            new_supports_cluster2 = self.supports_cluster + [A_cluster]
        
        #network
        transmit=self.transmit              
        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)
        transmit1 = self.transmit1(x,x_cluster)
        x_1=(torch.einsum('bmn,bcnl->bcml',transmit1,x_cluster))   
        
        x=self.gate1(x,x_1)
        
       
        skip1=0
        skip2=0
        ##0
        #1
        x_cluster1=self.block_cluster1(x_cluster,new_supports_cluster1) 
        x1=self.block1(x,new_supports1)   
        transmit21 = self.transmit2(x1,x_cluster1)
        x_21=(torch.einsum('bmn,bcnl->bcml',transmit21,x_cluster1)) 
        
        x1=self.gate2(x1,x_21) 
        
        
        s11=self.skip_conv1(x1)
        skip1=s11+skip1 
        
       
        #2       
        x_cluster1=self.block_cluster2(x_cluster1,new_supports_cluster1)
        x1=self.block2(x1,new_supports1) 
        transmit31 = self.transmit3(x1,x_cluster1)
        x_31=(torch.einsum('bmn,bcnl->bcml',transmit31,x_cluster1)) 
        
        x1=self.gate3(x1,x_31)
           
        
        s21=self.skip_conv2(x1)      
        skip1 = skip1[:, :, :,  -s21.size(3):]
        skip1 = s21 + skip1
        
        
        ##1
        #1
        x_cluster2=self.block_cluster1(x_cluster,new_supports_cluster2) 
        x2=self.block1(x,new_supports2)   
        transmit22 = self.transmit2(x2,x_cluster2)
        x_22=(torch.einsum('bmn,bcnl->bcml',transmit22,x_cluster2)) 
        
        x2=self.gate2(x2,x_22) 
        
        
        s12=self.skip_conv1(x2)
        skip2=s12+skip2
        
       
        #2       
        x_cluster2=self.block_cluster2(x_cluster2,new_supports_cluster2)
        x2=self.block2(x2,new_supports2) 
        transmit32 = self.transmit3(x2,x_cluster2)
        x_32=(torch.einsum('bmn,bcnl->bcml',transmit32,x_cluster2)) 
        
        x2=self.gate3(x2,x_32)
           
        
        s22=self.skip_conv2(x2)      
        skip2 = skip2[:, :, :,  -s22.size(3):]
        skip2 = s22 + skip2
        
        
        loss1=0.0 

        if self.training and self.CL=='True':
            p=(self.projection_head(skip1))
            p1=(self.projection_head(skip2))
            if k==0:
                loss1=self.loss(p,p1,mask_neg)
                
            else:
                loss1=self.loss1(p,p1,mask_neg,k)
            
        
        
        skip=torch.cat((skip1,skip2),1)
        
        #output
        x = F.relu(skip)      
        x = F.relu(self.end_conv_1(x))            
        x = self.end_conv_2(x)              
        return x,loss1,A


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
        

        self.loss=NTXentLoss(device,64,0.1,True)
        self.loss1=NTXentLoss_time(device,64,0.1,True)
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
        
        #if self.training==True:
            #input=rand_mask(input,0,self)
        x=input
        
        mask=(self.supports[0]!=0).float()
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A=F.dropout(A,0.5,self.training)
        
        mask=(self.supports[0]!=0).float()
        B=self.h*mask
        d1=1/(torch.sum(B,-1)+0.0001)
        D1=torch.diag_embed(d1)
        B=torch.matmul(D1,B)
        B=F.dropout(B,0.5,self.training)
        
        
        
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