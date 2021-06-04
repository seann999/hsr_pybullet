import os
import cv2
import numpy as np
import os
import torch
import json

import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from fcn_model import FCN
#from train_agent import phi


def phi(x):
    return (x - 0.2) / 0.2


class SegData:
    def __init__(self, train):
        self.root = 'pretrain_data2'
        self.files = os.listdir(self.root)

        if train:
            self.files = self.files[:16000]
        else:
            self.files = self.files[16000:20000]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hmap = cv2.imread(os.path.join(self.root, self.files[idx], 'hmap.png'), -1)

        angle = np.random.randint(low=0, high=360)
        image_center = (112, 112)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        # hmap = cv2.warpAffine(hmap, rot_mat, (224, 224))

        hmap = (hmap / 1000.0).astype(np.float32)#[None]
        #gtmap = cv2.imread(os.path.join(self.root, self.files[idx], 'maskmap.png'))[:, :, 0] > 0
        segmap = cv2.imread(os.path.join(self.root, self.files[idx], 'segmap.png'))[:, :, 0]
        info = json.load(open(os.path.join(self.root, self.files[idx], 'ids.json'), 'r'))
        # furns = info['furn_ids']
        # gtmap = np.logical_or.reduce([segmap == furns[i] for i in [11, 12, 13, 14]])
        objs = [i for i in info['obj_ids'] if (segmap == i).sum() > 0]
        if len(objs) > 0:
            coords = [np.stack(np.where(segmap == i)).T.mean(0) for i in objs]
            dists = np.linalg.norm(np.array(coords) - np.array([[112, 0]]), axis=1)
            nearest = objs[np.argmin(dists)]
            gtmap = segmap == nearest
        else:
            gtmap = np.zeros_like(segmap)
        gtmap = gtmap.astype(np.float32)
        # gtmap = cv2.warpAffine(gtmap, rot_mat, (224, 224))

        return np.stack([hmap, hmap, hmap]), gtmap


def create_fig(y_hat, y):
    fig, ax = plt.subplots(3, 8)
    #y_hat = F.sigmoid(y_hat).detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    y = y.cpu().numpy()

    for i in range(8):
        ax[0, i].imshow(x[i, 0])
        ax[1, i].imshow(y_hat[i, 0], vmin=0, vmax=1)
        ax[2, i].imshow(y[i, 0], vmin=0, vmax=1)
        #ax[3, i].imshow(g[i, 0], vmin=0, vmax=1)
        #ax[4, i].imshow(p[i, 0], vmin=0, vmax=1)

    fig.set_size_inches(24, 15)
    fig.tight_layout()

    return fig

model = FCN(num_rotations=16, use_fc=True, fast=True, debug=True)
train_loader = DataLoader(SegData(True), batch_size=32, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)
val_loader = DataLoader(SegData(False), batch_size=8, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)

model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

train_losses, val_losses = [], []
root = 'pretrain_results/fast'

try:
    os.makedirs(root)
except:
    pass

def reg_loss(p):
    loss = 0

    #for p in model.fc.parameters():
    #    loss += p.abs().sum()
    loss = 0 * torch.abs(p).sum(3).sum(2).sum(1).mean()

    return loss

for ep in range(1001):
    total_loss = 0
    model.train()

    for i, (x, y) in enumerate(train_loader):
        y_hat = model.forward(phi(x).cuda())
        y = y.unsqueeze(1).cuda()
        #loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none').sum(3).sum(2).sum(1).mean()
        loss = (y_hat - y).pow(2).sum(3).sum(2).sum(1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.cpu().detach().numpy()
        total_loss += loss

        print(i, len(train_loader), loss)

    loss_avg = total_loss / len(train_loader)
    print(loss_avg)
    train_losses.append(min(1000,loss_avg))

    fig = create_fig(y_hat, y)
    fig.savefig(root + '/train_{:05d}.png'.format(ep))

    del y_hat, loss
    torch.cuda.empty_cache()

    total_loss = 0
    model.eval()

    for i, (x, y) in enumerate(val_loader):
        y_hat = model.forward(phi(x).cuda())
        y = y.unsqueeze(1).cuda()
        #loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none').sum(3).sum(2).sum(1).mean()
        loss = (y_hat - y).pow(2).sum(3).sum(2).sum(1).mean()

        loss = loss.cpu().detach().numpy()
        total_loss += loss

        print(i, len(val_loader), loss)

    loss_avg = total_loss / len(val_loader)
    print(loss_avg)
    val_losses.append(min(1000,loss_avg))

    fig = create_fig(y_hat, y)
    fig.savefig(root + '/val_{:05d}.png'.format(ep))

    plt.clf()
    plt.figure(figsize=(4, 3))
    plt.yscale('log')
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.tight_layout()
    plt.savefig(root + '/loss.png')

    if ep % 10 == 0:
        torch.save(model.state_dict(), root + '/weights_{:03d}.p'.format(ep))
