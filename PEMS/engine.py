import torch.optim as optim
from model import *
import util
class trainer1():
    def __init__(self,scaler, in_dim, seq_length, num_nodes, CL, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = COGCN(device, num_nodes, CL,dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.CL=CL
    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        
        output,Siam_loss,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        # print(real.shape)
        # print(predict.shape)
        loss = self.loss(predict, real,0.0)
        
        #print(loss)
        #print(Siam_loss)
        if self.CL=='True':
            (loss+1.0*Siam_loss).backward() #08:0.5; 04:
        else:
            (loss).backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
