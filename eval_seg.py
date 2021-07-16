from pretrain import N_CLASSES
from fcn_model import FCN
import torch
import cv2
from instance_post_processing import get_panoptic_segmentation
import matplotlib.pyplot as plt
import numpy as np

output_channels = N_CLASSES + 1 + 2

model = FCN(num_rotations=output_channels, fast=True, dilation=True)
model.cuda()
model.eval()

state = torch.load('pretrain_results/pan05s/weights_018.p')
model.load_state_dict(state)

for data_i in range(30):
    depth = cv2.imread('pretrain_data/test/{:07d}/depth.png'.format(data_i), -1) / 1000.0
    x = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0).cuda()

    y_pred = model(x).detach()

    print(y_pred.shape)

    y_cls = y_pred[:, :N_CLASSES]
    y_center = y_pred[:, N_CLASSES:N_CLASSES+1]
    y_offset = y_pred[:, N_CLASSES+1:N_CLASSES+3] * 640
    y_offset = torch.cat([y_offset[:, 1:2], y_offset[:, 0:1]], 1)

    pan, centers = get_panoptic_segmentation(y_cls, y_center, y_offset, [2, 3], 30, 50, 0, threshold=0.2)

    centers = centers[0].cpu().numpy()
    # plt.clf()
    # plt.imshow(y_center[0, 0].cpu().numpy())
    # plt.plot(centers[:, 1], centers[:, 0], 'r*')
    # plt.colorbar()
    # plt.show()
    pan = pan[0].cpu().numpy()
    ids = np.zeros_like(pan)

    for i, id in enumerate(np.unique(pan)):
        ids[pan == id] = i

    rgb = cv2.imread('pretrain_data/test/{:07d}/rgb.png'.format(data_i))[:, :, ::-1]

    plt.clf()
    plt.subplot(131)
    plt.imshow(rgb)
    plt.subplot(132)
    plt.imshow(y_center[0, 0].cpu().numpy())
    # plt.plot(centers[:, 1], centers[:, 0], 'r*')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(ids, cmap='tab20', interpolation='nearest')
    plt.plot(centers[:, 1], centers[:, 0], 'k*')
    plt.colorbar()
    plt.show()
