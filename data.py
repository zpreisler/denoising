#!/usr/bin/env python
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader

from scipy.stats import norm
from numpy import linspace,arange,zeros,asarray
from matplotlib.pyplot import plot,show,figure

from numpy.random import normal,randint,random
from model import Denoise

class RndData():

    def __init__(self,n,batch_size):
        self.n = n
        self.batch_size = batch_size
        self.count = 0

    def gen_curve(self):

        nb = randint(1,3)
        n0 = randint(1,4)
        n1 = randint(0,3)

        x = arange(2048)
        z = zeros(2048)
        zb = zeros(2048)
        y = zeros(2048)

        for i in range(nb):
            loc = randint(0,1200)
            scale = randint(300,1000)
            rv = norm(loc,scale)

            r = 0.33 * random()
            z += rv.pdf(x) * scale * r
            r = 0.33 * random()
            zb += z * normal(0,r,2048)

        for i in range(n0):
            loc = randint(200,1000)
            scale = randint(6,32)
            rv = norm(loc,scale)
            y += rv.pdf(x) * scale * (random() * 0.6 + 0.4)

        for i in range(n1):
            loc = randint(200,1000)
            scale = randint(6,32)
            rv = norm(loc,scale)
            y += rv.pdf(x) * scale * random() * 0.3

        y0 = y.copy()
        y1 = y.copy()
        y1 += z
        y = y1 + zb

        y[y<0] = 0.0

        return y,y0,y1,z,n0,n1

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.count < self.n:
            self.count += 1

            x = []
            y = []
            for i in range(self.batch_size):
                _y,_y0,_y1,_z,_n0,_n1 = self.gen_curve()
                x += [_y]   
                y += [_y1]   

            return asarray(x),asarray(y)

        else:
            raise StopIteration()
