from scipy.spatial.transform import Rotation as R

import os
import cv2
import numpy as np
import os
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import matplotlib.pyplot as plt
import json
import env_utils as eu
import ravens.utils.utils as ru
import matplotlib.pyplot as plt
from fcn_model import FCN

BOUNDS = np.array([[0, 3], [-1.5, 1.5], [-0.05, 0.3]])


class SegData:
    def __init__(self, train=True):
        self.files = sorted(os.listdir('data'))
        self.debug_rgb = True
        self.root = 'data'
        self.noise = train

        if train:
            self.files = self.files[:4500]
        else:
            self.files = self.files[4500:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        depth = cv2.imread(os.path.join(self.root, self.files[idx], 'depth.png'), -1) / 1000.0

        if self.noise:
            if np.random.uniform() < 0.5:
                depth = eu.distort(depth, noise=np.random.uniform(0, 1))

        # plt.imshow(depth, vmax=5, cmap=plt.get_cmap('plasma'))
        # plt.show()

        seg = cv2.imread(os.path.join(self.root, self.files[idx], 'seg.png'))

        with open(os.path.join('data', self.files[idx], 'config.json')) as f:
            config = json.loads(f.read())

            if self.noise:
                config['position'] = np.array(config['position']) + np.random.normal(0, 0.01, 3)

                rvec = np.random.normal(0, 1, 3)
                rvec /= np.linalg.norm(rvec)
                mag = 1 / 180.0 * np.pi

                euler = R.from_quat(config['rotation']).as_euler('xyz')
                # euler[2] = np.pi * np.random.uniform(-0.9, -0.1)

                rot = R.from_euler('xyz', euler)  # R.from_quat(config['rotation'])
                # rot = R.from_euler('z', np.random.uniform(-np.pi*0.25, np.pi*0.25)) * rot
                config['rotation'] = (R.from_rotvec(mag * rvec) * rot).as_quat()

        hmaps, segmaps = ru.reconstruct_heightmaps(
            [seg], [depth], [config], BOUNDS, 0.01)

        hmap = hmaps[0].astype(np.float32)
        hmap = np.dstack([hmap, hmap, hmap])

        gtmap = np.logical_and(segmaps[0][:, :, 0] >= 3, segmaps[0][:, :, 0] <= 81)
        gtmap = gtmap.astype(np.float32)

        hmap = np.transpose(hmap, (2, 0, 1))

        return hmap, gtmap


def create_fig(x, y_hat, y):
    fig, ax = plt.subplots(3, 10)
    y_hat = F.sigmoid(y_hat).detach().cpu().numpy()
    y = y.cpu().numpy()

    for i in range(10):
        ax[0, i].imshow(x[i, 0])
        ax[1, i].imshow(y_hat[i, 0], vmin=0, vmax=1)
        ax[2, i].imshow(y[i, 0], vmin=0, vmax=1)

    fig.set_size_inches(18, 6)
    fig.tight_layout()

    return fig


def train_fcn():
    model = FCN()
    train_loader = DataLoader(SegData(True), batch_size=32, num_workers=4, shuffle=True, pin_memory=True,
                              drop_last=True, worker_init_fn=lambda x: np.random.seed())
    val_loader = DataLoader(SegData(False), batch_size=32, num_workers=4, shuffle=True, pin_memory=True, drop_last=True,
                            worker_init_fn=lambda x: np.random.seed())

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    train_losses, val_losses = [], []

    for ep in range(10000):
        total_loss = 0
        model.train()

        for i, (x, y) in enumerate(train_loader):
            y_hat = model.forward(x.cuda())
            y = y.unsqueeze(1).cuda()
            loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none').sum(3).sum(2).sum(1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            total_loss += loss

            print(i, len(train_loader), loss)

        loss_avg = total_loss / len(train_loader)
        print(loss_avg)
        train_losses.append(loss_avg)

        fig = create_fig(x, y_hat, y)
        fig.savefig('results/train_{:05d}.png'.format(ep))

        # del y_hat, loss
        # torch.cuda.empty_cache()

        total_loss = 0
        model.eval()

        for i, (x, y) in enumerate(val_loader):
            y_hat = model.forward(x.cuda())
            y = y.unsqueeze(1).cuda()
            loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none').sum(3).sum(2).sum(1).mean()

            loss = loss.cpu().detach().numpy()
            total_loss += loss

            print(i, len(val_loader), loss)

        loss_avg = total_loss / len(val_loader)
        print(loss_avg)
        val_losses.append(loss_avg)

        fig = create_fig(x, y_hat, y)
        fig.savefig('results/val_{:05d}.png'.format(ep))

        plt.clf()
        plt.figure(figsize=(4, 3))
        plt.yscale('log')
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.legend()
        plt.tight_layout()
        plt.savefig('loss.png')

        torch.save(model.state_dict(), 'weights_{:03d}.p'.format(ep), _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    train_fcn()

    # data = SegData()
    # for _ in range(5):
    #     x = data[1]
    #     plt.imshow(x[0][0])
    #     plt.show()
    #     plt.imshow(x[1])
    #     plt.show()
