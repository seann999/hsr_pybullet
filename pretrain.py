import os
import cv2
import numpy as np
import os
import torch
import json
import argparse
import random

from matplotlib.colors import ListedColormap
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from fcn_model import FCN
#from train_agent import phi
import albumentations as A


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


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
colors = plt.get_cmap('tab20')
newcolors = colors(np.arange(20))
COLORS = np.concatenate([np.array([[0, 0, 0, 1.0]]), newcolors, np.array([[1, 1, 1, 1.0]])], 0)
#newcmp = ListedColormap(newcolors)

def visualize(out):
    x = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)

    for i in range(22):
        c = COLORS[i]
        x[out == i] = tuple(np.uint8(c[:3] * 255))

    return x


def analyze(root, files):
    crops = [[] for _ in range(16)]
    for f in files:
        info = json.load(open(os.path.join(root, f, 'ids.json'), 'r'))
        hmap = cv2.imread(os.path.join(root, f, 'hmap.png'), -1)
        if info['pick']['success']:
            try:
                x, y = info['pick']['pick_px']
                crop = hmap[y-20:y+20, x-20:x+20]
                if crop.shape != (40, 40):
                    continue
                crop = crop.astype(np.float32)
                #crop -= crop.mean()
                crops[info['pick']['pick_rot_idx']].append(crop)
            except Exception as e:
                print(e)

    for i in range(16):
        plt.clf()
        x = np.array(crops[i])
        plt.imshow(x.mean(0))
        plt.savefig('avg_{}.png'.format(i))


def analyze2(root, files):
    X = []
    for f in files:
        info = json.load(open(os.path.join(root, f, 'ids.json'), 'r'))
        x, y = info['pick']['pick_px']
        r = info['pick']['pick_rot_idx']
        X.append([x, y, r, int(info['pick']['success'])])

    X = np.array(X)

    plt.clf()
    success = X[:, 3] == 1
    plt.scatter(X[X[:, 3] == 0][:, 0], X[X[:, 3] == 0][:, 1], marker='x', label='fail')
    for i in range(16):
        mask = np.logical_and(success, X[:, 2] == i)
        plt.scatter(X[mask][:, 0], X[mask][:, 1], marker='x', label='success')
    plt.legend()
    plt.savefig('scatter.png')


class SegData:
    def __init__(self, root, train, classify=False, placing=False, picking=False, hmap=False, balance=False):
        self.root = root
        self.files = sorted(os.listdir(self.root))
        self.classify = classify
        self.placing = placing
        self.picking = picking
        self.balance = balance
        self.hmap = hmap
        self.train = train
        self.offset = 0

        if picking:
            self.transform = A.Compose([
                # A.Cutout(max_h_size=10, max_w_size=10),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0),
            ])
        else:
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=360 if placing else 30, border_mode=cv2.BORDER_CONSTANT, value=0),
            ])

        valid = []
        success = []
        for f in self.files:
            try:
                info = json.load(open(os.path.join(self.root, f, 'ids.json'), 'r'))
                pdata = info['pick']
                success.append(float(pdata['success'] and not pdata['furniture_collision']))
                valid.append(f)
            except json.decoder.JSONDecodeError:
                print(f, 'is corrupt')

        self.files = valid
        self.success = np.array(success)
        # analyze(root, self.files)
        # analyze2(root, self.files)
        # exit()

        if train:
            self.files = self.files[:80000]
            self.success = self.success[:80000]
        else:
            self.files = self.files[80000:100000]
            self.success = self.success[80000:100000]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if self.balance:
            s = 1#int(np.random.random() > 0.5)
            valid = np.where(self.success == s)[0]
            idx = valid[0]#np.random.choice(valid)
        else:
            idx = i

        hmap = cv2.imread(os.path.join(self.root, self.files[idx], 'hmap.png' if self.hmap else 'noisy_depth.png'), -1)

        #angle = np.random.randint(low=0, high=360)
        #image_center = (112, 112)
        #rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        # hmap = cv2.warpAffine(hmap, rot_mat, (224, 224))

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
        elif self.placing:
            gtmap = np.zeros(list(segmap.shape[:2]) + [3], dtype=np.float32)
            pdata = info['place']
            x, y = pdata['loc_px']
            x, y = np.clip(x, 0, 223), np.clip(y, 0, 223)
            val = (float(pdata['contact_loc']), float(pdata['contact_neighbor']), float(pdata['contact_other']))
            gtmap[y, x, :] = val

            mask = np.zeros(segmap.shape[:2], dtype=np.float32)
            mask[y, x] = 1.0
        elif self.picking:
            gtmap = np.zeros(list(segmap.shape[:2]) + [16], dtype=np.float32)
            pdata = info['pick']
            x, y = pdata['pick_px']
            x, y = np.clip(x, 0, 223), np.clip(y, 0, 223)
            val = np.zeros(16)
            rot_idx = self.offset#np.random.randint(16)
            r = (pdata['pick_rot_idx'] - rot_idx + 16) % 16

            mask = np.zeros(gtmap.shape, dtype=np.float32)

            rot = int(360 / 16 * rot_idx)
            # M = cv2.getRotationMatrix2D(tuple(np.int32(np.array(gtmap.shape[:2]) / 2)), rot, 1)
            M = cv2.getRotationMatrix2D((112, 112), rot, 1)
            x, y = np.round(M.dot(np.array([x, y, 1]))[:2]).astype(np.int32)
            w_xy, w_r = 2, 0 # window

            if x >= 0 and y >= 0 and x < 224 and y < 224:
                y0, y1 = max(0, y - w_xy), min(y + w_xy, 224)
                x0, x1 = max(0, x - w_xy), min(x + w_xy, 224)
                
                gtmap[y0:y1, x0:x1, r] = float(pdata['success'] and not pdata['furniture_collision'])
                mask[y0:y1, x0:x1, r] = 1

            hmap = cv2.warpAffine(hmap, M, hmap.shape[:2], flags=cv2.INTER_NEAREST)
            #gtmap = cv2.warpAffine(gtmap, M, gtmap.shape[:2], flags=cv2.INTER_NEAREST)
            #mask = cv2.warpAffine(mask, M, mask.shape[:2], flags=cv2.INTER_NEAREST)
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

        if self.train and self.classify:
            re = self.transform(image=hmap, mask=gtmap)
            hmap, gtmap = re['image'], re['mask']
            gtmap = gtmap.astype(np.int64)
        elif self.train and self.placing:
            re = self.transform(image=hmap, masks=[gtmap, mask])
            hmap, gtmap, mask = re['image'], re['masks'][0], re['masks'][1]
        elif self.train and self.picking:
            re = self.transform(image=hmap, masks=[gtmap, mask])
            hmap, gtmap, mask = re['image'], re['masks'][0], re['masks'][1]

        if self.picking:
            gtmap = np.transpose(gtmap, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))
        elif self.placing:
            gtmap = np.transpose(gtmap, (2, 0, 1))

        hmap = (hmap / 1000.0).astype(np.float32)[None]

        if self.placing or self.picking:
            return hmap, gtmap, mask

        return hmap, gtmap


def create_fig(y_hat, y, classification=False, placing=False, picking=False):
    fig, ax = plt.subplots(3, 8)
    #y_hat = F.sigmoid(y_hat).detach().cpu().numpy()
    #y_hat = y_hat.detach()

    if placing:
        y_hat = F.sigmoid(y_hat).permute(0, 2, 3, 1)
        #y_hat = torch.clip(y_hat.permute(0, 2, 3, 1), 0, 1)
        y = y.permute(0, 2, 3, 1)

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
            ax[1, i].imshow(visualize(y_hat[i].argmax(axis=0)), interpolation='nearest')
            ax[2, i].imshow(visualize(y[i]), cmap=newcmp, interpolation='nearest')
        elif placing:
            ax[1, i].imshow(np.uint8(y_hat[i] * 255))
            ax[2, i].imshow(np.uint8(y[i] * 255))
        elif picking:
            import colorsys
            y_hat_idx = y_hat[i].argmax(0)
            y_hat_prob = np.clip(y_hat[i].max(0), 0, 1)
            y_idx = y[i].argmax(0)
            y_prob = np.clip(y[i].max(0), 0, 1)
            
            y_rgb = np.zeros(list(y_hat[i].shape[1:]) + [3], dtype=np.float32)
            y_hat_rgb = np.zeros(list(y_hat[i].shape[1:]) + [3], dtype=np.float32)

            N = y_hat[i].shape[0]
            for k in range(N):
                rgb = colorsys.hsv_to_rgb(k / N, 1.0, 1.0)
                y_hat_rgb[y_hat_idx == k] = rgb
                y_rgb[y_idx == k] = rgb

            y_rgb *= y_prob[:, :, None]
            y_hat_rgb *= y_hat_prob[:, :, None]

            ax[1, i].imshow(np.uint8(y_hat_rgb * 255))
            ax[2, i].imshow(np.uint8(y_rgb * 255))
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
    parser.add_argument('--picking', action='store_true')
    parser.add_argument('--placing', action='store_true')
    parser.add_argument('--no-hmap', action='store_true')
    args = parser.parse_args() 

    def phi(x):
        if args.no_hmap:
            return x

        return (x - 0.2) / 0.2

    if args.classify:
        output_channels = 22
    elif args.placing:
        output_channels = 3
    elif args.picking:
        output_channels = 16
    else:
        output_channels = 16
   
    bs = 32
    if args.no_hmap:
        bs = 16
    #elif args.picking:
    #    bs = 128

    def init_fn(worker_id):
        process_seed = torch.initial_seed()
        # Back out the base_seed so we can use all the bits.
        base_seed = process_seed - worker_id
        ss = np.random.SeedSequence([worker_id, base_seed])
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))

    d = SegData(args.data, True, args.classify, args.placing, args.picking, hmap=not args.no_hmap, balance=True)
    plt.clf()
    R = 8
    for i in range(R):
        d.offset = i
        x, y, m = d[0]

        import colorsys
        y_idx = y.argmax(0)
        y_prob = np.clip(y.max(0), 0, 1)

        y_rgb = np.zeros(list(y.shape[1:]) + [3], dtype=np.float32)

        N = y.shape[0]
        for k in range(N):
            rgb = colorsys.hsv_to_rgb(k / N, 1.0, 1.0)
            if np.sum(y[k,:,:]) > 0:
                print('>', i, k, np.sum(y[k,:,:]))
            y_rgb[y_idx == k] = rgb

        y_rgb *= y_prob[:, :, None]

        plt.subplot(2, R, i+1)
        plt.imshow(x[0])
        ys, xs = np.where(y_prob == 1)
        py = ys.mean()
        px = xs.mean()
        plt.plot([px], [py], 'r*', markersize=10)
        plt.subplot(2, R, i+R+1)
        plt.imshow(np.uint8(y_rgb * 255))
    plt.gcf().set_size_inches(32, 8)
    plt.gcf().tight_layout()
    plt.savefig('augs.png')
    exit()

    model = FCN(num_rotations=output_channels, use_fc=True, fast=True, debug=True, dilation=args.classify)
    train_loader = DataLoader(SegData(args.data, True, args.classify, args.placing, args.picking, hmap=not args.no_hmap, balance=True), batch_size=bs, num_workers=8, shuffle=True, pin_memory=True, drop_last=True, worker_init_fn=init_fn)
    val_loader = DataLoader(SegData(args.data, False, args.classify, args.placing, args.picking, hmap=not args.no_hmap), batch_size=8, num_workers=8, shuffle=True, pin_memory=True, drop_last=True, worker_init_fn=init_fn)

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses = [], []
    root = args.outdir

    try:
        os.makedirs(root)
    except:
        pass

    def calc_loss(y_hat, y, m):
        if args.classify:
            loss = F.cross_entropy(y_hat, y.cuda(), reduction='none').sum(2).sum(1).mean()
        elif args.placing:
            m = m.cuda().unsqueeze(1)
            loss = (F.binary_cross_entropy_with_logits(y_hat, y.cuda(), reduction='none') * m).sum(3).sum(2).sum(1).mean()
            #loss = ((y_hat - y.cuda()).pow(2) * m).sum(3).sum(2).sum(1).mean()
        elif args.picking:
            y = y.cuda()
            w = 1#torch.where(y == 1, 1, 0.06)
            loss = ((y_hat - y).pow(2) * m.cuda() * w).sum(3).sum(2).sum(1).mean()
        else:
            y = y.cuda().unsqueeze(1)
            loss = (y_hat - y).pow(2).sum(3).sum(2).sum(1).mean()

        return loss

    m = None

    for ep in range(101):
        # np.random.seed()
        total_loss = 0
        model.train()

        for i, d in enumerate(train_loader):
            if args.placing or args.picking:
                x, y, m = d
            else:
                x, y = d

            y_hat = model.forward(phi(x).cuda())
            #loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none').sum(3).sum(2).sum(1).mean()
            loss = calc_loss(y_hat, y, m)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.cpu().detach().numpy()
            total_loss += loss

            #fig = create_fig(y_hat, y, classification=args.classify)
            #fig.savefig(os.path.join(root, 'train_{:05d}.png'.format(ep)))

            print(i, len(train_loader), loss)

        loss_avg = total_loss / len(train_loader)
        print(loss_avg)
        train_losses.append(loss_avg)

        fig = create_fig(y_hat, y, classification=args.classify, placing=args.placing, picking=args.picking)
        fig.savefig(os.path.join(root, 'train_{:05d}.png'.format(ep)))

        del y_hat, loss
        torch.cuda.empty_cache()

        total_loss = 0
        model.eval()

        for i, d in enumerate(val_loader):
            if args.placing or args.picking:
                x, y, m = d
            else:
                x, y = d

            y_hat = model.forward(phi(x).cuda())
            #loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none').sum(3).sum(2).sum(1).mean()
            loss = calc_loss(y_hat, y, m)

            loss = loss.cpu().detach().numpy()
            total_loss += loss

            print(i, len(val_loader), loss)

        loss_avg = total_loss / len(val_loader)
        print(loss_avg)
        val_losses.append(loss_avg)

        fig = create_fig(y_hat, y, classification=args.classify, placing=args.placing, picking=args.picking)
        fig.savefig(os.path.join(root, 'val_{:05d}.png'.format(ep)))

        plt.clf()
        plt.figure(figsize=(4, 3))
        plt.yscale('log')
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(root, 'loss.png'))

        if ep % 1 == 0:
            torch.save(model.state_dict(), os.path.join(root, 'weights_{:03d}.p'.format(ep)))
