import torchvision.models as models
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, num_rotations=16):
        super().__init__()

        self.num_rotations = num_rotations
        self.use_cuda = True

        modules = list(models.resnet18().children())[:-5]
        self.backbone = nn.Sequential(*modules)
        self.end = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, 1, 1),
        )

    def forward(self, x):
        bs = len(x)
        # y = self.end(self.backbone(x))

        # assert x.shape[-2:] == y.shape[-2:], 'input =/= output shape {} {}'.format(x.shape, y.shape)
        output_prob = []

        if self.num_rotations == 1:
            return self.end(self.backbone(x))
        else:
            for rotate_idx in range(self.num_rotations):
                rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                # Compute sample grid for rotation BEFORE neural network
                affine_mat_before = np.asarray(
                    [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                affine_mat_before = affine_mat_before.repeat(bs, 1, 1)

                #print(affine_mat_before.shape, x.shape)
                #print(affine_mat_before.is_cuda, x.is_cuda)

                if self.use_cuda:
                    flow_grid_before = F.affine_grid(affine_mat_before.cuda(),
                                                     x.size())
                else:
                    flow_grid_before = F.affine_grid(affine_mat_before.detach(), x.size())

                # Rotate images clockwise
                if self.use_cuda:
                    rotate_depth = F.grid_sample(x.detach().cuda(), flow_grid_before, mode='nearest')
                else:
                    rotate_depth = F.grid_sample(x.detach(), flow_grid_before, mode='nearest')

                # Compute intermediate features
                output_map = self.end(self.backbone(rotate_depth))

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray(
                    [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_mat_after = affine_mat_after.repeat(bs, 1, 1)

                if self.use_cuda:
                    flow_grid_after = F.affine_grid(affine_mat_after.detach().cuda(),
                                                    output_map.size())
                else:
                    flow_grid_after = F.affine_grid(affine_mat_after.detach(),
                                                    output_map.size())

                # Forward pass through branches, undo rotation on output predictions, upsample results
                output_prob.append(F.grid_sample(output_map, flow_grid_after, mode='nearest'))

        return output_prob

