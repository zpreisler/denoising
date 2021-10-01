import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader


class UNet(nn.Module):
    def __init__(self, kernel_size = 3):
        super().__init__()

        self.conv_in = nn.Conv1d(in_channels=1, out_channels=128, padding=0, kernel_size=kernel_size,bias=True,stride=1)

        self.conv0 =  nn.Conv1d(in_channels=128, out_channels=128, padding=0, kernel_size=kernel_size,bias=True,stride=1)
        self.conv1 =  nn.Conv1d(in_channels=256, out_channels=256, padding=0, kernel_size=kernel_size,bias=True,stride=1)
        self.conv2 =  nn.Conv1d(in_channels=192, out_channels=192, padding=0, kernel_size=kernel_size,bias=True,stride=1)

        self.conv_out = nn.ConvTranspose1d(in_channels=256, out_channels=1, padding=0, kernel_size=kernel_size,bias=True,stride=1)

        self.down = nn.Sequential(
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2)
            )

        self.center = nn.Sequential(
            nn.ELU()
            )

        self.up = nn.Sequential(
            nn.ELU(),
            nn.Upsample(scale_factor=2)
            )

    def forward(self,x):

        x = self.conv_in(x)
        l1 = self.center(x)
        print('',x.shape)

        x = self.conv0(l1)
        l2 = self.down(x)

        print('',x.shape)
        x = self.conv0(l2)
        l3 = self.down(x)

        print('',x.shape)
        x = self.conv0(l3)
        x = self.down(x)

        print('',x.shape)
        x = self.conv0(x)
        x = self.center(x)

        print('',x.shape)
        x = self.conv0(x)
        x = self.up(x)

        x = torch.cat((x,l3[:,:,5:-5]),1)
        print('',x.shape)
        x = self.conv1(x)
        x = self.up(x)

        #x = torch.cat((x,l2[:,:,13:-13]),1)
        print('',x.shape)
        x = self.conv1(x)
        x = self.up(x)

        #x = torch.cat((x,l1[:,:,29:-29]),1)
        #print('',x.shape)
        x = self.conv_out(x)

        print('',x.shape)
        x = x[:,:,227:-227]
        print('crop',x.shape)
        return x

class UNet2(nn.Module):
    def __init__(self, kernel_size = 3):
        super().__init__()

        self.conv_in = nn.Conv1d(in_channels=1, out_channels=128, padding=0, kernel_size=kernel_size,bias=True,stride=1)

        self.conv0 =  nn.Conv1d(in_channels=128, out_channels=128, padding=0, kernel_size=kernel_size,bias=True,stride=1)
        self.conv1 =  nn.Conv1d(in_channels=256, out_channels=256, padding=0, kernel_size=kernel_size,bias=True,stride=1)
        self.conv2 =  nn.Conv1d(in_channels=192, out_channels=192, padding=0, kernel_size=kernel_size,bias=True,stride=1)

        self.conv_out = nn.ConvTranspose1d(in_channels=256, out_channels=1, padding=0, kernel_size=kernel_size,bias=True,stride=1)

        self.down = nn.Sequential(
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2)
            )

        self.center = nn.Sequential(
            nn.ELU()
            )

        self.up = nn.Sequential(
            nn.ELU(),
            nn.Upsample(scale_factor=2)
            )

        self.linear = nn.Linear(2048,2048) 

    def forward(self,x):

        x = self.conv_in(x)
        l1 = self.center(x)
        print('',x.shape)

        x = self.conv0(l1)
        l2 = self.down(x)

        print('',x.shape)
        x = self.conv0(l2)
        l3 = self.down(x)

        print('',x.shape)
        x = self.conv0(l3)
        x = self.down(x)

        print('',x.shape)
        x = self.conv0(x)
        x = self.center(x)

        print('',x.shape)
        x = self.conv0(x)
        x = self.up(x)

        x = torch.cat((x,l3[:,:,5:-5]),1)
        print('',x.shape)
        x = self.conv1(x)
        x = self.up(x)

        #x = torch.cat((x,l2[:,:,13:-13]),1)
        print('',x.shape)
        x = self.conv1(x)
        x = self.up(x)

        #x = torch.cat((x,l1[:,:,29:-29]),1)
        #print('',x.shape)
        x = self.conv_out(x)

        print('',x.shape)
        x = x[:,:,227:-227]
        print('crop',x.shape)
        x = self.linear(x)
        return x

class Encode(nn.Module):
    def __init__(self,channels = 128, kernel_size = 3):
        super().__init__()

        self.encode = nn.Sequential(

                nn.Conv1d(in_channels=1, out_channels=channels, padding=0, kernel_size=kernel_size,bias=True,stride=1),
                nn.ELU(),
                nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(in_channels=channels, out_channels=channels, padding=0, kernel_size=kernel_size,bias=True,stride=1),
                nn.ELU(),
                nn.MaxPool1d(kernel_size=2),

                #nn.Conv1d(in_channels=channels, out_channels=channels, padding=0, kernel_size=kernel_size,bias=True,stride=1),
                #nn.ELU(),
                #nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(in_channels=channels, out_channels=channels, padding=0, kernel_size=kernel_size,bias=True,stride=1),
                nn.ELU(),

                )

    def forward(self,x):
        return self.encode(x)

class Decode(nn.Module):
    def __init__(self,channels = 128, kernel_size = 3):
        super().__init__()

        self.decode = nn.Sequential(
                nn.ConvTranspose1d(in_channels=channels, out_channels=channels, padding=0, kernel_size=kernel_size,bias=True,stride=1),
                nn.ELU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(in_channels=channels, out_channels=channels, padding=0, kernel_size=kernel_size,bias=True),
                nn.ELU(),
                nn.Upsample(scale_factor=2),

                #nn.ConvTranspose1d(in_channels=channels, out_channels=channels, padding=0, kernel_size=kernel_size,bias=True),
                #nn.ELU(),
                #nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(in_channels=channels, out_channels=1, padding=0, kernel_size=kernel_size,bias=True,stride=1),

                )

        #self.linear = nn.Linear(2048,1938,bias=False)

    def forward(self,x):
        return self.decode(x) 

class Denoise(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = Encode()
        self.decode = Decode()

    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)
        x = x[:,:,255:-255]
        print('Denoise shape:',x.shape)
        return x


