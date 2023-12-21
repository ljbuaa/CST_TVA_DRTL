import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from Blocks import ConvBlock
from CSMHA import CSMultiHeadAttention

class CSTransormer(nn.Module):
    def __init__(self, tlen,num_heads=5,drop_p=0.5,expansion=4):
        super().__init__()
        self.tlen = tlen
        self.layerNorm=nn.LayerNorm(tlen)
        self.csmha=CSMultiHeadAttention(tlen, num_heads, drop_p)
        self.ffb=nn.Sequential(
            nn.LayerNorm(tlen),
            nn.Linear(tlen, expansion * tlen),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * tlen, tlen),
            nn.Dropout(drop_p)
        )

    def forward(self, x):
        res=x
        x = self.layerNorm(x)
        x = self.csmha(x)
        x+=res
        res=x
        x = self.ffb(x)
        x += res
        return x

class CSTUnit(nn.Module):
    def __init__(self,channelNum):
        super().__init__()
        self.channelNum=channelNum
        self.conv1x=nn.Sequential(
            ConvBlock(1,20,(1,61),stride=(1,2),padding=(0,30)),
            ConvBlock(20,40,(self.channelNum,1)),
        )
        self.conv2x=nn.Sequential(
            ConvBlock(1,20,(1,31),stride=(1,2),padding=(0,15)),
            ConvBlock(20,40,(self.channelNum,1)),
        )
        self.conv4x=nn.Sequential(
            ConvBlock(1,20,(1,15),stride=(1,2),padding=(0,7)),
            ConvBlock(20,40,(self.channelNum,1)),
        )

        self.getWeight1x=nn.Conv2d(40, 1, 1)
        self.getWeight2x=nn.Conv2d(40, 1, 1)
        self.getWeight4x=nn.Conv2d(40, 1, 1)

        self.CSTrans=nn.Sequential(
            Rearrange("b c 1 t -> b (c 1) t"),
            CSTransormer(60,2),
            Rearrange("b (c 1) t-> b c 1 t"),
        )

    def forward(self, x):
        out1x = self.conv1x(x)
        out2x = self.conv2x(x)
        out4x = self.conv4x(x)

        w1 = self.getWeight1x(out1x)
        w2 = self.getWeight2x(out2x)
        w4 = self.getWeight4x(out4x)
        weights=torch.sigmoid(torch.cat((w1,w2,w4),dim=1))
        out1x=weights[:,0:1]*out1x
        out2x=weights[:,1:2]*out2x
        out4x=weights[:,2:3]*out4x

        out=torch.cat((out1x,out2x,out4x),dim=1)
        out = self.CSTrans(out)
        out1,out2,out3=out.chunk(3,dim=1)
        out=out1+out2+out3
        return out
