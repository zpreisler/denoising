#!/usr/bin/env python
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader

from scipy.stats import norm
from numpy import linspace,arange,zeros,asarray,pad
from matplotlib.pyplot import plot,show,figure,gca

from numpy.random import normal,randint,random
from model import Denoise,UNet
from data import RndData

def main():
    """
    Main
    """
    torch.set_num_threads(1)

    #model = Denoise()
    model = UNet()

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        #print(param_tensor, "\t", model.state_dict()[param_tensor])

    #torch.nn.init.constant_(model.state_dict()['decode.linear.weight'],2.0) 
    #torch.nn.init.zeros_(model.state_dict()['decode.linear.bias']) 

    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr = 0.0001)
            
    gen_curve = RndData(256,32)

    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)

    model.to(device)

    offset=256

    for i,batch in enumerate(gen_curve):
        x,y = batch

        x = pad(x,((0,0),(offset,offset)),mode='reflect') 

        x = x.reshape((-1,1,x.shape[-1]))
        y = y.reshape((-1,1,y.shape[-1]))

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        print('out shape:',outputs.shape)

        loss = (criterion(outputs,y))

        print(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss},
    'checkpoint.pth'
    )


    x = x.detach().cpu().numpy()
    print(x.shape)
    x = x[:,:,offset:-offset]
    print(x.shape)
    x = x.reshape(-1,2048)

    y = y.detach().cpu().numpy()
    y = y.reshape(-1,2048)

    outputs = outputs.detach().cpu().numpy()
    outputs = outputs.reshape(-1,2048)

    figure()
    plot(x[:5].T)
    plot(y[:5].T,':')

    figure()
    plot(y[:5].T)

    figure()
    plot(outputs[:5].T)
    gca().set_prop_cycle(None)
    plot(y[:5].T,'--')
    show()

if __name__ == '__main__':
    main()
