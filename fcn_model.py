import torchvision.models as models
from torch import nn


class FCN(nn.Module):
    def __init__(self):
        super().__init__()

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
        y = self.end(self.backbone(x))

        assert x.shape[-2:] == y.shape[-2:], 'input =/= output shape {} {}'.format(x.shape, y.shape)

        return y
