#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl
from math import *
from numpy import *
import argparse
import scipy.special
import numpy as np
from numpy import linalg
from sympy import *

parser = argparse.ArgumentParser()
parser.add_argument('-save', action='store_true', default=False)
args = parser.parse_args()

mpl.rc('text',usetex=True)
mpl.rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
mpl.rc('font',size=12)
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

dataDMRG = loadtxt('DMRGevn.txt')
dataBCS = loadtxt('BCSevn.txt')

plot_N_DMRG = []
plot_N_BCS = []
plot_DMRG = []
plot_BCS = []

for i in range(len(dataDMRG[:-2,0])):
	plot_N_DMRG.append(dataDMRG[i,0])
	plot_DMRG.append(dataDMRG[i+1,1]-dataDMRG[i,1])

for i in range(len(dataBCS[:-2,0])):
	plot_N_BCS.append(dataBCS[i,0])
	plot_BCS.append(dataBCS[i+1,1]-dataBCS[i,1])

for i in range(len(plot_DMRG)):
	print(plot_DMRG[i],plot_BCS[i])

plt.plot(plot_N_DMRG, (np.asarray(plot_DMRG)-np.asarray(plot_BCS)), marker='.', label='BCS error, 2-neutron separation')
#plt.plot(plot_N_BCS, plot_BCS, marker='.', label='BCS')

plt.xlabel('$N$')

plt.savefig('bandstructure.png')
plt.legend()
plt.grid()
plt.show()

