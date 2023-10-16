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
parser.add_argument('--device',type=str,default='cuda:0',help='graphics card')
parser.add_argument('--data',type=str,default='data/PEMS04',help='data path')
parser.add_argument('--adjdata',type=str,default='data/PEMS04/adj_pems04.pkll',help='adj data path')
parser.add_argument('--seq_length',type=int,default=12,help='prediction length')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=307,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='Gradient Clipping')
parser.add_argument('--lr_decay_rate', type=float, default=1, help='learning rate')
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--model',type=str,default='COGCN',help='model selection(CLGCN or Pretrain_GCN)')
parser.add_argument('--print_every',type=int,default=100,help='')
parser.add_argument('--save',type=str,default='./garage/pems04',help='save path')
parser.add_argument('--force', type=str, default=False,help="remove params dir", required=False)
parser.add_argument('--CL', type=str, default=False,help="remove params dir", required=False)
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
    device = torch.device(args.device)
    adj_mx = util.load_adj(args.adjdata)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    if args.model=='COGCN':
        engine = trainer1( scaler,args.in_dim, args.seq_length, args.num_nodes,args.CL,args.l, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.lr_decay_rate
                         )
    if args.model=='COGCN_noloss':
        engine = trainer1( scaler,args.in_dim, args.seq_length, args.num_nodes,args.CL,args.l,  args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.lr_decay_rate
                         )
    if args.model=='gwnet':
        engine = trainer2( scaler,args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.lr_decay_rate
                         )
    if args.model=='OGCRNN':
        engine = trainer3( scaler,args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.lr_decay_rate
                         )        
     
    # check parameters file
    params_path=args.save+"/"+args.model
    print("start testing...",flush=True)
    
    #testing
    engine.model.load_state_dict(torch.load(params_path+"/"+args.model+"_best"+".pth")) #relplace COGCN:2.83 COGCN_NL:191
    
    
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]
    engine.model.eval()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds,spatial_at,parameter_adj = engine.model(testx)
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
        pred = scaler.inverse_transform(yhat[:,:,i])
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
