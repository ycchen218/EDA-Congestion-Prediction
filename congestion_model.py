import torch
import torch.nn as nn
from CoordConv import AddCoords
from FPN import FPN101



class Decoder(nn.Module):
    def __init__(self,device):
        super(Decoder, self).__init__()
        if device =='cpu':
            self.addcoord = AddCoords(rank=2,w=256,h=256,with_r=False,skiptile=True,use_cuda=False)
            self.tile16 = AddCoords(rank=2,w=16,h=16,with_r=False,skiptile=False,use_cuda=False)
        else:
            self.addcoord = AddCoords(rank=2,w=256,h=256,with_r=False,skiptile=True)
            self.tile16 = AddCoords(rank=2,w=16,h=16,with_r=False,skiptile=False)
        self.convT1 = nn.ConvTranspose2d(256+2, 128, kernel_size=4, stride=2, padding=1)
        self.convT2 = nn.ConvTranspose2d(256+2, 128, kernel_size=4, stride=2, padding=1)
        self.convT3 = nn.ConvTranspose2d(256+2, 64, kernel_size=4, stride=2, padding=1)
        self.convT4 = nn.ConvTranspose2d(64+128+2, 1, kernel_size=4, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        # add coord for every feature
        d1 = self.act(self.convT1(self.tile16(feature[-1])))
        d1 = torch.cat([d1, self.addcoord(feature[-2])], dim=1)
        d2 = self.act(self.convT2(d1))
        d2 = torch.cat([d2, self.addcoord(feature[-3])], dim=1)
        d3 = self.act(self.convT3(d2))
        d3 = torch.cat([d3, self.addcoord(feature[-4])], dim=1)
        output = self.convT4(d3)
        # output = self.sigmoid(self.convT4(d3))
        return output


class CongestionModel(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.encoder = FPN101(device)
        self.decoder = Decoder(device)
    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out

    def sigmoid(self,x):
        x = torch.sigmoid(x)
        return x

