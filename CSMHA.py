import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CSMultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.5):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys1 = nn.Linear(emb_size, emb_size)
        self.queries1 = nn.Linear(emb_size, emb_size)
        self.values1 = nn.Linear(emb_size, emb_size)
        self.keys2 = nn.Linear(emb_size, emb_size)
        self.queries2 = nn.Linear(emb_size, emb_size)
        self.values2 = nn.Linear(emb_size, emb_size)
        self.keys3 = nn.Linear(emb_size, emb_size)
        self.queries3 = nn.Linear(emb_size, emb_size)
        self.values3 = nn.Linear(emb_size, emb_size)
        self.att_drop1 = nn.Dropout(dropout)
        self.att_drop2 = nn.Dropout(dropout)
        self.att_drop3 = nn.Dropout(dropout)
        self.projection1 = nn.Linear(emb_size, emb_size)
        self.projection2 = nn.Linear(emb_size, emb_size)
        self.projection3 = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        x1,x2,x3=torch.chunk(x,3,1)
        queries1 = rearrange(self.queries1(x1), "b n (h d) -> b h n d", h=self.num_heads) 
        keys1 = rearrange(self.keys1(x1), "b n (h d) -> b h n d", h=self.num_heads)
        values1 = rearrange(self.values1(x1), "b n (h d) -> b h n d", h=self.num_heads)
        queries2 = rearrange(self.queries2(x2), "b n (h d) -> b h n d", h=self.num_heads) 
        keys2 = rearrange(self.keys2(x2), "b n (h d) -> b h n d", h=self.num_heads)
        values2 = rearrange(self.values2(x2), "b n (h d) -> b h n d", h=self.num_heads)
        queries3 = rearrange(self.queries3(x3), "b n (h d) -> b h n d", h=self.num_heads) 
        keys3 = rearrange(self.keys3(x3), "b n (h d) -> b h n d", h=self.num_heads)
        values3 = rearrange(self.values3(x3), "b n (h d) -> b h n d", h=self.num_heads)
        energy12 = torch.einsum('bhqd, bhkd -> bhqk', queries1, keys2)  #
        energy23 = torch.einsum('bhqd, bhkd -> bhqk', queries2, keys3)  #
        energy31 = torch.einsum('bhqd, bhkd -> bhqk', queries3, keys1)

        scaling = self.emb_size ** (1 / 2)
        att1 = F.softmax(energy23 / scaling, dim=-1)
        att1 = self.att_drop1(att1)
        out1 = torch.einsum('bhal, bhlv -> bhav ', att1, values3) 
        out1 = rearrange(out1, "b h n d -> b n (h d)")
        out1 = self.projection1(out1)

        att2 = F.softmax(energy31 / scaling, dim=-1)
        att2 = self.att_drop2(att2)
        out2 = torch.einsum('bhal, bhlv -> bhav ', att2, values1)
        out2 = rearrange(out2, "b h n d -> b n (h d)")
        out2 = self.projection2(out2)

        att3 = F.softmax(energy12 / scaling, dim=-1)
        att3 = self.att_drop3(att3)
        out3 = torch.einsum('bhal, bhlv -> bhav ', att3, values2)
        out3 = rearrange(out3, "b h n d -> b n (h d)")
        out3 = self.projection3(out3)

        return torch.cat((out1,out2,out3),dim=1)
