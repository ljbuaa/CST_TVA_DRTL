import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,with_bn=True,with_relu=True,**args):
        super().__init__()
        self.with_bn=with_bn
        self.with_relu=with_relu
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,**args)
        self.batchNorm=None
        self.relu=None
        if with_bn:
            self.batchNorm=nn.BatchNorm2d(out_channels)
        if with_relu:
            self.relu=nn.ELU()
    def forward(self, x):
        out=self.conv2d(x)
        if self.with_bn:
            out=self.batchNorm(out)
        if self.with_relu:
            out=self.relu(out)
        return out

class BasicDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,with_bn=True,with_relu=True,**args):
        super().__init__()
        self.with_bn=with_bn
        self.with_relu=with_relu
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,**args)
        self.batchNorm=None
        self.relu=None
        if with_bn:
            self.batchNorm=nn.BatchNorm2d(out_channels)
        if with_relu:
            self.relu=nn.ELU()
    def forward(self, x):
        out=self.conv2d(x)
        if self.with_bn:
            out=self.batchNorm(out)
        if self.with_relu:
            out=self.relu(out)
        return out
