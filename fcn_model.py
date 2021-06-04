from torch import nn
import numpy as np
import torch
import torch.nn.functional as F

import resnet


class FCN(nn.Module):
    def __init__(self, num_rotations=16, use_fc=False, fast=False, debug=False):
        super().__init__()

        self.num_rotations = num_rotations
        self.use_cuda = True
        self.use_fc = use_fc
        self.debug = debug
        self.fast = fast

        #modules = list(models.resnet18().children())[:-5]
        #self.backbone = nn.Sequential(*modules)
        self.backbone = resnet.resnet18(num_input_channels=3)#models.resnet18()
        #backbone = resnet.resnet18(num_input_channels=3, num_classes=1)
        #self.resnet.cuda()
        #self.backbone = backbone.features
        
        def decoder(n, out):
          return nn.Sequential(
            nn.Conv2d(n, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, out, 1, 1),
          )

        self.decoder = decoder(512, 16)
        #self.fcn = deeplabv3_resnet50(pretrained=False, num_classes=16)
        #self.head = nn.Sequential(
        #    nn.Conv2d(32, 32, 1, 1),
        #    nn.BatchNorm2d(32),
        #    nn.ReLU(),
        #    nn.Conv2d(32, num_rotations if fast else 1, 1, 1)
        #)

    def cat_grid(self, input, affine_grid=None):
        x = torch.abs(torch.linspace(-0.5, 0.5, steps=input.shape[-2])).cuda() # side
        y = torch.tensor(torch.linspace(0, 1, steps=input.shape[-1])).cuda()  # forward
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        grid_x = grid_x.repeat(len(input), 1, 1, 1)
        grid_y = grid_y.repeat(len(input), 1, 1, 1)
        grid = torch.cat([grid_x, grid_y], 1)

        if affine_grid is not None:
            flow_grid = F.affine_grid(affine_grid, input.size())
            grid = F.grid_sample(grid, flow_grid, mode='nearest')

        x = torch.cat([input, grid], 1)

        return x

    def forward(self, x):
        bs = len(x)
        output_prob = []

        if self.num_rotations == 1 or self.fast:
            #h = self.backbone(self.cat_grid(x))
            #g = self.end(h)
            g = self.decoder(self.backbone.features(x))
            #h = self.fcn(x)['out']
            #g = self.head(torch.cat([h, g], 1))

            return g
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
                    affine_mat_before = affine_mat_before.cuda()
                    flow_grid_before = F.affine_grid(affine_mat_before, x.size())
                    #flow_grid_vit = F.affine_grid(affine_mat_before, vit_h.size())
                else:
                    affine_mat_before = affine_mat_before.detach()
                    flow_grid_before = F.affine_grid(affine_mat_before, x.size())

                # Rotate images clockwise
                if self.use_cuda:
                    rotate_depth = F.grid_sample(x.detach().cuda(), flow_grid_before, mode='nearest')
                    #rotate_vit_h = F.grid_sample(vit_h, flow_grid_vit, mode='nearest')
                else:
                    rotate_depth = F.grid_sample(x.detach(), flow_grid_before, mode='nearest')

                # Compute intermediate features
                #output_map = self.end(torch.cat([rotate_vit_h, self.backbone(rotate_depth)], 1))
                h = self.backbone(rotate_depth)
                #if self.use_fc:
                #    a = self.fc(h[:, 0].view(-1, 56*56)).view(-1, 1, 56, 56)
                #    h = torch.cat([h, a], 1)
                output_map = self.end(h)

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
                h = F.grid_sample(output_map, flow_grid_after, mode='nearest')
                if self.use_fc:
                    h = torch.minimum(p, h)
                output_prob.append(h)

        out = torch.stack(output_prob)  # R x N x 1 x H x W
        out = out.squeeze(2)  # R x N x H x W
        out = out.permute(1, 0, 2, 3)

        if self.debug:
            return out, p

        return out


if __name__ == '__main__':
    model = FCN()
    model.cuda()
    model.eval()

    while True:
        y = model(torch.rand((1, 3, 224, 224)).cuda())
        print(torch.stack(y).shape)
