# from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch

# https://www.jianshu.com/p/d91d6d06766b 数据预处理的解释看这里
### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):
    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size = 64, grid_size = (13,3)):
        # traj去掉前两个维度后就是tracks(没有datasetID, vehID) 输入处理的水太深，先不搞这块，不影响我学习网络，先抓重点，怎么train起来一个网络
        self.D = scp.loadmat(mat_file)['traj'] #loadmat返回是dict格式 mat是字典嵌套字典的格式
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences缩小处理的特征数据量 https://blog.csdn.net/hxxjxw/article/details/106175155
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid 3根车道 前后6车，一共约13辆车的车长

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        # 把datasetID　和 vehID转成int格式，因为默认是float格式
        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,8:] #一共3*13，标号从左下，往上；存的方法 一列代表一个格子，里面存vehID
        neighbors = [] #list格式

        # Get 每辆车自己的track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId,t,vehId,dsId) #为什么用两个vehID什么区别？？？？？？
        fut = self.getFuture(vehId,t,dsId)

        # Get 在周围大长方形grid里的所有其他车的track histories of all neighbours
        # 'neighbors' = [ndarray,[],ndarray,ndarray]
        # ！！！！！之后可以改，这里只看了周围车的历史轨迹，没看预测的未来轨迹的反馈，加入其他车的未来的轨迹进输入特征里
        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t,vehId,dsId))

        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        # 要转成独热码，因为，标签隐含了距离信息，比如1、2比1、3之间更近
        lon_enc = np.zeros([2]) # 刹车和非刹车 初始化[0,0]
        lon_enc[int(self.D[idx, 7] - 1)] = 1 # 特别妙这里，好好体会：normal是[1,0] brake是[0,1]
        lat_enc = np.zeros([3]) #右3 左2 保持1 所以是右[0,0,1] 左[0,1,0] [1,0,0]
        lat_enc[int(self.D[idx, 6] - 1)] = 1

        return hist,fut,neighbors,lat_enc,lon_enc #neighbours是所有周围车辆的轨迹

    ## Helper function to get track history
    def getHistory(self,vehId,t,refVehId,dsId):
        if vehId == 0: #为什么还有0的啊？？？
            return np.empty([0,2]) #就是空的[]
        else:
            if self.T.shape[1]<=vehId-1: #track的列数<vehID，why？？？？
                return np.empty([0,2]) #就是空的[]
            refTrack = self.T[dsId-1][refVehId-1].transpose() #用来算refpos
            vehTrack = self.T[dsId-1][vehId-1].transpose()  #ref和veh区别，ref是存当前的，veh是用来算历史的，为什么要转置？？？？？？？
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3] #取reftrack这辆车的所有行，取第0列frameID==t这时刻的，这一行的local_X/Y/速度v

            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h) #start起点是3s前的序号，防止没有前3s，所以设个0
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1 #end是vehtrack当前这一帧t的序号，为什么+1？？？？？？
                # 改成下面这样试试看 ？？？？？？？？上面那里+1的地方看不懂
                # enpt = np.argwhere(vehTrack[:, 0] == t).item()
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos #历史的轨迹=历史-当前这一帧，为什么 看不懂这里

            if len(hist) < self.t_h//self.d_s + 1: #降采样，不懂为啥降采样？？
                return np.empty([0,2])
            return hist

    ## Helper function to get track future
    def getFuture(self, vehId, t,dsId):
        vehTrack = self.T[dsId-1][vehId-1].transpose() #为啥还转置，不理解，这个是self.T是dict嵌套吗？？？？
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3] #取当前这一帧 第0行，第1到3列
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s #为什么要降采样，在当前帧的索引加2帧降采样（序号的索引） stpt：start point
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1) # enpt: end point 那种加完50帧以后超了的，就要取min
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos #去掉本身当前这一帧这一行，所以要减refPos，取未来的
        return fut

    ## Collate（gather and examine整合） function for dataloader data从nparray转成dataloader送入网络的格式
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _,_,nbrs,_,_ in samples:
            nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))])
        maxlen = self.t_h//self.d_s + 1  #为什么历史长度要//降采样长度？？？？？？？？？？？？
        nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)


        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size)
        mask_batch = mask_batch.byte()


        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen,len(samples),2)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        lat_enc_batch = torch.zeros(len(samples),3)
        lon_enc_batch = torch.zeros(len(samples), 2)


        count = 0
        for sampleId,(hist, fut, nbrs, lat_enc, lon_enc) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut),sampleId,:] = 1
            lat_enc_batch[sampleId,:] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)

            # Set up neighbor, neighbor sequence length, and mask batches:
            for id,nbr in enumerate(nbrs):
                if len(nbr)!=0:
                    nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
                    count+=1

        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch

## Custom activation for output layer (Graves, 2015) #为什么要这一层激活层，不加会怎么样？？？？？？？？？？？
def outputActivation(x):
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    return out


## Batchwise NLL loss, uses mask for variable output lengths# 看git原仓库的issue讨论区
def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3]
    rho = y_pred[:,:,4]
    ohr = torch.pow(1-torch.pow(rho,2),-0.5)
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    # If we represent likelihood in feet^(-1):
    out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
    # If we represent likelihood in m^(-1):
    # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes = 2,use_maneuvers = True, avg_along_time = False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:,l]*lon_pred[:,k]
                wts = wts.repeat(len(fut_pred[0]),1)
                y_pred = fut_pred[k*num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                # If we represent likelihood in feet^(-1):
                out = -(0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + 0.5*torch.pow(sigY, 2)*torch.pow(y-muY, 2) - rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379)
                # If we represent likelihood in m^(-1):
                # out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160)
                acc[:, :, count] =  out + torch.log(wts)
                count+=1
        acc = -logsumexp(acc, dim = 2)
        acc = acc * op_mask[:,:,0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        # If we represent likelihood in feet^(-1):
        out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2 * rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
        # If we represent likelihood in m^(-1):
        # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:,:,0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal,counts

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
