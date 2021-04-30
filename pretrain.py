import os
import cv2
import numpy as np
import os
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from fcn_model import FCN
from train_agent import phi


class SegData:
    def __init__(self, train):
        self.root = 'pretrain_data'
        self.files = os.listdir(self.root)

        if train:
            self.files = self.files[:8000]
        else:
            self.files = self.files[8000:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hmap = cv2.imread(os.path.join(self.root, self.files[idx], 'hmap.png'), -1) / 1000.0
        hmap = hmap.astype(np.float32)[None]
        gtmap = cv2.imread(os.path.join(self.root, self.files[idx], 'maskmap.png'))[:, :, 0] > 0
        gtmap = gtmap.astype(np.float32)

        return hmap, gtmap


def create_fig(y_hat, y):
    fig, ax = plt.subplots(3, 8)
    #y_hat = F.sigmoid(y_hat).detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    y = y.cpu().numpy()

    for i in range(8):
        ax[0, i].imshow(x[i, 0])
        ax[1, i].imshow(y_hat[i, 0], vmin=0, vmax=1)
        ax[2, i].imshow(y[i, 0], vmin=0, vmax=1)

    fig.set_size_inches(18, 6)
    fig.tight_layout()

    return fig

model = FCN(num_rotations=1)
train_loader = DataLoader(SegData(True), batch_size=32, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)
val_loader = DataLoader(SegData(False), batch_size=8, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)

model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

train_losses, val_losses = [], []

for ep in range(30):
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
    train_losses.append(loss_avg)

    fig = create_fig(y_hat, y)
    fig.savefig('pretrain_results/train_{:05d}.png'.format(ep))

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
    val_losses.append(loss_avg)

    fig = create_fig(y_hat, y)
    fig.savefig('pretrain_results/val_{:05d}.png'.format(ep))

    plt.clf()
    plt.figure(figsize=(4, 3))
    plt.yscale('log')
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pretrain_results/loss.png')

    if ep % 10 == 0:
        torch.save(model.state_dict(), 'pretrain_results/weights_{:03d}.p'.format(ep))