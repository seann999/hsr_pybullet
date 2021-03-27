import torchvision.models as models
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import time

from self_attention_cv import TransformerEncoder
from einops import rearrange


class ViT(nn.Module):
    def __init__(self, img_dim,
                 in_channels=3,
                 patch_dim=16,
                 num_classes=10,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 dropout=0, transformer=None, classification=True):
        """
        Args:
            img_dim: the spatial image size
            in_channels: number of img channels
            patch_dim: desired patch dim
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
            classification: creates an extra CLS token
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible'
        self.p = patch_dim
        self.classification = classification
        tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)
        self.dim = dim
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)
        if self.classification:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, dim))
            self.mlp_head = nn.Linear(dim, num_classes)
        else:
            self.pos_emb1D = nn.Parameter(torch.randn(tokens, dim))

        if transformer is None:
            self.transformer = TransformerEncoder(dim, blocks=blocks, heads=heads,
                                                  dim_head=self.dim_head,
                                                  dim_linear_block=dim_linear_block,
                                                  dropout=dropout)
        else:
            self.transformer = transformer

    def expand_cls_to_batch(self, batch):
        """
        Args:
            batch: batch size
        Returns: cls token expanded to the batch size
        """
        return self.cls_token.expand([batch, -1, -1])

    def forward(self, img, mask=None):
        batch_size = img.shape[0]
        img_patches = rearrange(
            img, 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)
        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        if self.classification:
            img_patches = torch.cat(
                (self.expand_cls_to_batch(batch_size), img_patches), dim=1)

        patch_embeddings = self.emb_dropout(img_patches + self.pos_emb1D)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)

        if self.classification:
            # we index only the cls token for classification. nlp tricks :P
            return self.mlp_head(y[:, 0, :])
        else:
            return y


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
        self.vit = ViT(56, in_channels=66, patch_dim=1, dim=64, blocks=1, heads=1,
                       dim_linear_block=64, classification=False)

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

    def self_attention(self, x):
        x = self.vit(x)
        x = x.view(-1, 56, 56, 64)
        x = x.permute(0, 3, 1, 2)

        return x

    def forward(self, x):
        bs = len(x)
        # y = self.end(self.backbone(x))

        # assert x.shape[-2:] == y.shape[-2:], 'input =/= output shape {} {}'.format(x.shape, y.shape)
        output_prob = []

        # x = x[:, 0:1]
        # x = self.cat_meshgrid(x)

        if self.num_rotations == 1:
            out = self.end(self.self_attention(self.cat_grid(self.backbone(x))))
            return out
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
                else:
                    affine_mat_before = affine_mat_before.detach()
                    flow_grid_before = F.affine_grid(affine_mat_before, x.size())

                # Rotate images clockwise
                if self.use_cuda:
                    rotate_depth = F.grid_sample(x.detach().cuda(), flow_grid_before, mode='nearest')
                else:
                    rotate_depth = F.grid_sample(x.detach(), flow_grid_before, mode='nearest')

                # Compute intermediate features
                output_map = self.end(self.self_attention(self.cat_grid(self.backbone(rotate_depth), affine_mat_before)))

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


if __name__ == '__main__':
    model = FCN()
    model.cuda()
    model.eval()

    while True:
        y = model(torch.rand((1, 3, 224, 224)))
        print(torch.stack(y).shape)