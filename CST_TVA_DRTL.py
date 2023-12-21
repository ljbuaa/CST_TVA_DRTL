import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from Blocks import ConvBlock
from CST import CSTUnit
from TVA import TVAUnit

class FeatureExtractBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.channelNum=config["tfimages_shape"][-3]
        self.speNum=config["tfimages_shape"][-2]
        self.x_chNum=config["xdawn_shape"][-2]
        self.cst = CSTUnit(self.x_chNum)
        self.tva = TVAUnit(self.channelNum,self.speNum)

    def forward(self, x,xcwt):
        x=x.unsqueeze(1)
        out1= self.cst(x)
        out2= self.tva(xcwt)
        out=torch.cat((out1,out2),dim=1)
        return out

class CST_TVA_DRTL(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.domain_nums=config["domain_nums"]
        zdim=config["zdim"]
        self.baseFea=FeatureExtractBlock(config)
        self.DSEncoder = nn.Sequential(
            ConvBlock(80,60,(1,11),stride=(1,2),padding=(0,5)),
            nn.AdaptiveAvgPool2d((1,6)),
            Rearrange("b c 1 t -> b (c 1 t)"),
            nn.Linear((60)*6, zdim),
            nn.ReLU(),
        )
        self.DIEncoder = nn.Sequential(
            ConvBlock(80,60,(1,11),stride=(1,2),padding=(0,5)),
            nn.AdaptiveAvgPool2d((1,6)),
            Rearrange("b c 1 t -> b (c 1 t)"),
            nn.Linear((60)*6, zdim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(zdim, 2),
        )
        self.domain_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(zdim, self.domain_nums),
        )
        self.domain_discer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(zdim, self.domain_nums),
        )
    def forward(self, x,xcwt,outfea=False):
        baseFeas=self.baseFea(x,xcwt)
        dsFea=self.DSEncoder(baseFeas)
        diFea=self.DIEncoder(baseFeas)
        pre1 =self.classifier(dsFea+diFea)
        if outfea:
            return pre1,dsFea,diFea
        else: 
            return pre1

if __name__ == '__main__':
    xdawn=torch.randn(10,8,120).cuda() 
    tfimages=torch.randn(10,8,19,120).cuda()
    mod_config={"xdawn_shape":xdawn.shape,"tfimages_shape":tfimages.shape,"domain_nums":63,"zdim":512}
    model=CST_TVA_DRTL(mod_config).cuda() 
    pre_y=model(xdawn,tfimages)
    print("pre_y.shape:",pre_y.shape)
