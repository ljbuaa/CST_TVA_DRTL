import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from Blocks import ConvBlock

class TVAUnit(nn.Module):
    def __init__(self,channelNum,speNum):
        super().__init__()
        self.channelNum=channelNum
        self.speNum=speNum
        self.speConv= nn.Sequential(
            nn.Conv2d(self.channelNum,self.channelNum,(self.speNum,1)),
            Rearrange('b c s t -> b (c s) t'),
        )
        self.spaConv = nn.Sequential(
            Rearrange('b c s t -> b s c t'),
            nn.Conv2d(self.speNum,self.speNum,(self.channelNum,1)),
            Rearrange('b c s t -> b (c s) t'),
        )

        self.tempConv= nn.Sequential(
            nn.Conv2d(self.channelNum,self.channelNum,(1,120)),
            Rearrange('b c s t -> b c (s t)'),
        )
        self.conv0 = nn.Sequential(
            ConvBlock(3*self.channelNum,30,(self.speNum,1),groups=1), #
            ConvBlock(30,40,(1,13),stride=(1,2),padding=(0,6)),
        )

    def forward(self, x):
        spe_out = self.speConv(x)
        spa_out = self.spaConv(x)
        temp_out = self.tempConv(x)
        cs = torch.sigmoid(torch.einsum("bct, bst->bcs", spe_out, spa_out)).unsqueeze(-1).expand(x.shape)
        ct = torch.sigmoid(torch.einsum("bcs, bst->bct", temp_out, spa_out)).unsqueeze(-2).expand(x.shape)
        st = torch.sigmoid(torch.einsum("bcs, bct->bst", temp_out, spe_out)).unsqueeze(-3).expand(x.shape)
        out=torch.cat((cs*x,ct*x,st*x),1)
        out=self.conv0(out)

        return out

