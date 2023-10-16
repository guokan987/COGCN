# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 11:15:02 2023

@author: gk
"""

import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import *
import os
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/JiNan_City',help='data path')
parser.add_argument('--adjdata',type=str,default='data/JiNan_City/adj_mat.pkl',help='adj data path')
parser.add_argument('--adjdatacluster',type=str,default='data/JiNan_City/adj_mat_cluster.pkl',help='adj data path')
parser.add_argument('--transmit',type=str,default='data/JiNan_City/transmit.csv',help='data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--in_dim_cluster',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=561,help='number of nodes')
parser.add_argument('--cluster_nodes',type=int,default=20,help='number of cluster')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0000,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=50,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument("--force", type=str, default=False,help="remove params dir", required=False)
parser.add_argument('--save',type=str,default='./garage/JiNan_City',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--model',type=str,default='gwnet',help='adj type')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')
parser.add_argument('--CL', type=str, default=True,help="remove params dir", required=False)
parser.add_argument('--l', type=int,default=3,help='block layers')
args = parser.parse_args()
##model repertition
def setup_seed(seed):
    #np.random.seed(seed) # Numpy module
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # multi-GPU

seed = 1
setup_seed(seed)
def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    sensor_ids_cluster, sensor_id_to_ind_cluster, adj_mx_cluster = util.load_adj(args.adjdatacluster,args.adjtype)
    dataloader = util.load_dataset_cluster(args.data, args.batch_size, args.batch_size, args.batch_size)
    #scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    supports_cluster = [torch.tensor(i).to(device) for i in adj_mx_cluster]
    transmit_np=np.float32(np.loadtxt(args.transmit,delimiter=','))
    transmit=torch.tensor(transmit_np).to(device)
    
    print(args)
    
    if args.model=='COHGCN':
        engine = trainer7( args.in_dim,args.in_dim_cluster, args.seq_length, args.num_nodes,args.cluster_nodes,args.CL,args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports,supports_cluster,transmit,args.decay
                         )
    if args.model=='HGCN':
        engine = trainer6( args.in_dim,args.in_dim_cluster, args.seq_length, args.num_nodes,args.cluster_nodes,args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports,supports_cluster,transmit,args.decay
                         )
    
     
    # check parameters file
    params_path=args.save+"/"+args.model
    print("start testing...",flush=True)
    
    #testing
    engine.model.load_state_dict(torch.load(params_path+"/"+args.model+"_epoch_1"+".pth")) #relplace COGCN:2.83 COGCN_NL:2.85
    
    
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]
    engine.model.eval()
    for iter, (x,y,x_cluster,y_cluster) in enumerate(dataloader['test_loader_cluster'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        testx_cluster = torch.Tensor(x_cluster).to(device)
        testx_cluster = testx_cluster.transpose(1, 3)
        with torch.no_grad():
            preds,_,_ = engine.model(testx,testx_cluster)
            preds=preds.transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    

    amae = []
    amape = []
    armse = []
    prediction=yhat
    for i in range(12):
        pred = prediction[:,:,i]
        #pred = scaler.inverse_transform(yhat[:,:,i])
        #prediction.append(pred)
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
    
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    # prediction_path=params_path+"/"+args.model+"_prediction_results_test"
    # ground_truth=realy.cpu().detach().numpy()
    # prediction=prediction.cpu().detach().numpy()
    # #spatial_at=spatial_at.cpu().detach().numpy()
    # parameter_adj=parameter_adj.cpu().detach().numpy()
    # np.savez_compressed(
    #         os.path.normpath(prediction_path),
    #         prediction=prediction,
    #         #spatial_at=spatial_at,
    #         parameter_adj=parameter_adj,
    #         ground_truth=ground_truth
    #     )


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
