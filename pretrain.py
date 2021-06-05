import os
import cv2
import numpy as np
import os
import torch
import json
import argparse

from matplotlib.colors import ListedColormap
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from fcn_model import FCN
#from train_agent import phi


CLASSES = {
    'walls': 3,
    'shelf': 4,
    'bin_left': 5,
    'bin_right': 6,
    'stair_drawer': 7,
    'drawer_bottom': 8,
    'knob_bottom': 9,
    'drawer_left': 10,
    'knob_left': 11,
    'drawer_top': 12,
    'knob_top': 13,
    'tall_table': 14,
    'long_table': 15,
    'long_table_placing': 16,
    'tray_left': 17,
    'tray_right': 18,
    'container_left': 19,
    'container_right': 20,
}


class SegData:
    def __init__(self, root, train, classify=False, hmap=False):
        self.root = root
        self.files = os.listdir(self.root)
        self.classify = classify
        self.hmap = hmap

        if train:
            self.files = self.files[:8000]
        else:
            self.files = self.files[8000:10000]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hmap = cv2.imread(os.path.join(self.root, self.files[idx], 'hmap.png' if self.hmap else 'noisy_depth.png'), -1)

        #angle = np.random.randint(low=0, high=360)
        #image_center = (112, 112)
        #rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        # hmap = cv2.warpAffine(hmap, rot_mat, (224, 224))

        hmap = (hmap / 1000.0).astype(np.float32)[None]
        #gtmap = cv2.imread(os.path.join(self.root, self.files[idx], 'maskmap.png'))[:, :, 0] > 0
        segmap = cv2.imread(os.path.join(self.root, self.files[idx], 'segmap.png' if self.hmap else 'seg.png'), -1)#.astype(np.float32)
        info = json.load(open(os.path.join(self.root, self.files[idx], 'ids.json'), 'r'))
        # furns = info['furn_ids']
        # gtmap = np.logical_or.reduce([segmap == furns[i] for i in [11, 12, 13, 14]])
        if self.classify:
            placed = info.get('placed_obj_ids', [])
            gtmap = np.zeros_like(segmap).astype(np.int64)

            for id in info['obj_ids']:
                if id in placed:
                    gtmap[segmap == id] = 2
                else:
                    gtmap[segmap == id] = 1

            gtmap[segmap == info['robot_id']] = 21

            for name, cls_idx in CLASSES.items():
                gtmap[segmap == info['furn_ids'][name]] = cls_idx
        else:
            placed = info.get('placed_obj_ids', [])
            objs = [i for i in info['obj_ids'] if i not in placed and (segmap == i).sum() > 0]
            if len(objs) > 0:
                coords = [np.stack(np.where(segmap == i)).T.mean(0) for i in objs]
                dists = np.linalg.norm(np.array(coords) - np.array([[112, 0]]), axis=1)
                nearest = objs[np.argmin(dists)]
                gtmap = segmap == nearest
            else:
                gtmap = np.zeros_like(segmap)
            gtmap = gtmap.astype(np.float32)
            # gtmap = cv2.warpAffine(gtmap, rot_mat, (224, 224))

        return hmap, gtmap


def create_fig(y_hat, y, classification=False):
    fig, ax = plt.subplots(3, 8)
    #y_hat = F.sigmoid(y_hat).detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    y = y.cpu().numpy()

    colors = plt.get_cmap('tab20')
    newcolors = colors(np.arange(20))
    newcolors = np.concatenate([np.array([[0, 0, 0, 1.0]]), newcolors], 0)
    newcmp = ListedColormap(newcolors)

    for i in range(8):
        ax[0, i].imshow(x[i, 0])

        if classification:
            y = y.astype(np.uint8)
            ax[1, i].imshow(y_hat[i].argmax(axis=0), cmap=newcmp, interpolation='nearest')
            ax[2, i].imshow(y[i], cmap=newcmp, interpolation='nearest')
        else:
            ax[1, i].imshow(y_hat[i, 0], vmin=0, vmax=1)
            ax[2, i].imshow(y[i, 0], vmin=0, vmax=1)
        #ax[3, i].imshow(g[i, 0], vmin=0, vmax=1)
        #ax[4, i].imshow(p[i, 0], vmin=0, vmax=1)

    fig.set_size_inches(24, 15)
    fig.tight_layout()

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='pretrain_shapenet')
    parser.add_argument('--outdir', type=str, default='pretrain_results/test00')
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--no-hmap', action='store_true')
    args = parser.parse_args() 

    def phi(x):
        if args.no_hmap:
            return x

        return (x - 0.2) / 0.2

    model = FCN(num_rotations=22 if args.classify else 16, use_fc=True, fast=True, debug=True)
    train_loader = DataLoader(SegData(args.data, True, args.classify, hmap=not args.no_hmap), batch_size=16 if args.no_hmap else 32, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(SegData(args.data, False, args.classify, hmap=not args.no_hmap), batch_size=8, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    train_losses, val_losses = [], []
    root = args.outdir

    try:
        os.makedirs(root)
    except:
        pass

    def calc_loss(y_hat, y):
        if args.classify:
            loss = F.cross_entropy(y_hat, y.cuda(), reduction='none').sum(2).sum(1).mean()
        else:
            y = y.unsqueeze(1).cuda()
            loss = (y_hat - y).pow(2).sum(3).sum(2).sum(1).mean()

        return loss

    for ep in range(101):
        total_loss = 0
        model.train()

        for i, (x, y) in enumerate(train_loader):
            y_hat = model.forward(phi(x).cuda())
            #loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none').sum(3).sum(2).sum(1).mean()
            loss = calc_loss(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.cpu().detach().numpy()
            total_loss += loss

            print(i, len(train_loader), loss)

        loss_avg = total_loss / len(train_loader)
        print(loss_avg)
        train_losses.append(loss_avg)

        fig = create_fig(y_hat, y, classification=args.classify)
        fig.savefig(os.path.join(root, 'train_{:05d}.png'.format(ep)))

        del y_hat, loss
        torch.cuda.empty_cache()

        total_loss = 0
        model.eval()

        for i, (x, y) in enumerate(val_loader):
            y_hat = model.forward(phi(x).cuda())
            #loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none').sum(3).sum(2).sum(1).mean()
            loss = calc_loss(y_hat, y)

            loss = loss.cpu().detach().numpy()
            total_loss += loss

            print(i, len(val_loader), loss)

        loss_avg = total_loss / len(val_loader)
        print(loss_avg)
        val_losses.append(loss_avg)

        fig = create_fig(y_hat, y, classification=args.classify)
        fig.savefig(os.path.join(root, 'val_{:05d}.png'.format(ep)))

        plt.clf()
        plt.figure(figsize=(4, 3))
        plt.yscale('log')
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(root, 'loss.png'))

        if ep % 10 == 0:
            torch.save(model.state_dict(), os.path.join(root, '/weights_{:03d}.p'.format(ep)))
