import torch
import numpy as np
import argparse
import time
import util
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
parser.add_argument('--weight_decay',type=float,default=0.000001,help='weight decay rate')
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
print(args)

def setup_seed(seed):
    #np.random.seed(seed) # Numpy module
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # multi-GPU

seed = 1
setup_seed(seed)

def main():

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
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
        
        
    params_path=args.save+"/"+args.model
    

    if os.path.exists(params_path) and not args.force:
        raise SystemExit("Params folder exists! Select a new params path please!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path)) 
    
    print("start training...",flush=True)
    his_loss =[]
    his_train_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        print('***** Epoch: %03d START *****' % i)
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).cuda()
            
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).cuda()
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        # if args.CL and args.model1=='None':
        #     engine.scheduler1.step()
        # else:
        #     engine.scheduler.step()

        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).cuda()
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).cuda()
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        his_train_loss.append(mtrain_loss)
        
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)
        

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        
        torch.save(engine.model.state_dict(), params_path+"/"+args.model+"_epoch_"+str(i)+".pth")

        print('***** Epoch: %03d END *****' %i)
        print('\n')

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    
    #save traing loss and validation loss
    his_loss1=np.mat(his_loss)
    his_train_loss1=np.mat(his_train_loss)
    np.savetxt( params_path+"/"+args.model+"_his_loss.csv", his_loss1, delimiter="," )
    np.savetxt( params_path+"/"+args.model+"_his_train_loss.csv", his_train_loss1, delimiter="," )
    
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(params_path+"/"+args.model+"_epoch_"+str(bestid+1)+".pth"))
    engine.model.eval()
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).cuda()
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).cuda()
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds,_,_ = engine.model(testx)
            preds=preds.transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    print("Best model epoch:", str(bestid+1))

    amae = []
    amape = []
    armse = []
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
    torch.save(engine.model.state_dict(), params_path+"/"+args.model+"_best"+".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
