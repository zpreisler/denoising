#!/usr/bin/env python

from numpy import loadtxt,zeros,arcsin,pi,digitize,linspace,zeros,asarray
from glob import glob
from matplotlib.pyplot import show,figure,plot,vlines,xlim,ylim,fill_between,title
from numpy import fft
from scipy.stats import norm

names = sorted(glob('Database/*.cif'))

G = {}
for name in names:
    d = {}
    d['name'] = name 
    with open(name,'r') as f:
        for line in f:
            x = line.split()
            if x:
                if x[0] == '_chemical_formula_sum':
                    d[x[0]] = ' '.join(x[1:]).replace("'",'')
                if x[0] == '_chemical_name_mineral':
                    d[x[0]] = ' '.join(x[1:]).replace("'",'')
                if x[0] == '_chemical_name_common':
                    d[x[0]] = x[1:]

                if x[0] == '_pd_peak_intensity':
                    z = loadtxt(f,unpack=True,dtype=float)
                    d[x[0]] = z


    formula = d['_chemical_formula_sum']
    if '_chemical_name_mineral' in d:
        mineral = d['_chemical_name_mineral']
    else:
        mineral = formula

    if mineral in G:
        G[mineral] += [d] 
    else:
        G[mineral] = [d]

for k,v in G.items():
    print(k,len(v),len(v[0]['_pd_peak_intensity'][0]))

figure()
for v in G['Pseudomalachite']:
    print(v['_chemical_formula_sum'])

    d,y = v['_pd_peak_intensity']

    l = 1.54
    g = l / (2 * d)
    theta = 360 * arcsin(g)/pi
    vlines(theta,zeros(len(theta)),y,'k')
    xlim(0,90)
    ylim(0,1000)

def get_theta(v,l=1.54):
    d,y = v['_pd_peak_intensity']

    g = l / (2 * d)
    theta = 360 * arcsin(g) / pi
    return theta,y

def plot_diffraction(phase):
    Z = []
    for v in phase:

        theta,y = get_theta(v) 
        z = zeros(1280)
        x = linspace(0,55,1280)

        #vlines(thetas,zeros(len(thetas)),y,'k',lw=0.5,alpha=0.1)
        for loc,_y in zip(theta,y):
            rv = norm(loc=loc,scale=0.2)
            z += rv.pdf(x) * _y
        Z += [z]
        #plot(x,z / z.max() * 1000,alpha=0.4)
        fill_between(x,z/z.max()*1000,y2=0,alpha=0.5/len(G['Calcite']),color='r')


    xlim(0,55)
    ylim(0,1100)

    Z = asarray(Z).sum(axis=0)
    plot(x,Z / Z.max() * 1000)

print(len(G))

figure()
plot_diffraction(G['Calcite'])

#for k,v in G.items():
#    print(k)
#    figure()
#    plot_diffraction(v)
#    title(k)

show()
