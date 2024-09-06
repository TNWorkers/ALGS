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
parser.add_argument('-el', action='store', default='Sn')
args = parser.parse_args()

fig, (ax1L, ax2L) = plt.subplots(2)
ax1R = ax1L.twinx()
ax2R = ax2L.twinx()

axL = [ax1L, ax2L]
axR = [ax1R, ax2R]

mpl.rc('text',usetex=True)
mpl.rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
mpl.rc('font',size=16)
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

cabs = 'k'#'#d9355a' #[,,] #'tab:red'
crel = 'purple' #'#7e2f85'#'#EC1557' #'tab:purple'
N_map = {'Sn':50, 'Pb':82}
Ninit_map = {'Sn':0, 'Pb':15}
Nfinal_map = {'Sn':85, 'Pb':55}
elements = ['Sn','Pb']

for iel,el in enumerate(elements):
	
	dataDMRG = loadtxt('E0_'+el+'_DMRG.dat')
	dataBCS = loadtxt('E0_'+el+'_BCS.dat')

	plot_N_DMRG = []
	plot_N_BCS = []
	plot_DMRG = []
	plot_BCS = []

	for i in range(1,len(dataDMRG[1:-1,0])+1):
		plot_N_DMRG.append(dataDMRG[i,0])
		Delta3 = dataDMRG[i,1]-0.5*(dataDMRG[i-1,1]+dataDMRG[i+1,1])
		plot_DMRG.append(Delta3)
		print(i,el,'DMRG','N=',dataDMRG[i,0],'E=',dataDMRG[i,1],dataDMRG[i-1,1],dataDMRG[i+1,1],'Delta=',Delta3)

	for i in range(1,len(dataBCS[1:-1,0])+1):
		plot_N_BCS.append(dataBCS[i,0])
		Delta3 = dataBCS[i,1]-0.5*(dataBCS[i-1,1]+dataBCS[i+1,1])
		plot_BCS.append(Delta3)
		print(i,el,'BCS','N=',dataBCS[i,0],'E=',dataBCS[i,1],dataBCS[i-1,1],dataBCS[i+1,1],'Delta=',Delta3)

	print(el,'len=',len(dataBCS[1:-1,0]))
	#for i in range(len(plot_DMRG)):
	#	print(plot_DMRG[i],plot_BCS[i])

	plot_N_DMRG = np.asarray(plot_N_DMRG)+N_map[el]
	plot_N_BCS = np.asarray(plot_N_BCS)+N_map[el]

	plot_DMRG = np.asarray(plot_DMRG)
	plot_BCS = np.asarray(plot_BCS)

	#plt.plot(plot_N_DMRG[15:55], plot_DMRG[15:55], marker='.', label='exact (DMRG)', c='#20B254')
	#plt.plot(plot_N_BCS[15:55], plot_BCS[15:55], marker='x', ls='', label='BCS', c='tab:red') #1.3*

	print(iel)
	axL[iel].plot(plot_N_DMRG[Ninit_map[el]:Nfinal_map[el]], abs(plot_DMRG[Ninit_map[el]:Nfinal_map[el]]-plot_BCS[Ninit_map[el]:Nfinal_map[el]]), marker='.', c=cabs)
	axR[iel].plot(plot_N_DMRG[Ninit_map[el]:Nfinal_map[el]], abs(plot_DMRG[Ninit_map[el]:Nfinal_map[el]]-plot_BCS[Ninit_map[el]:Nfinal_map[el]])/abs(plot_DMRG[Ninit_map[el]:Nfinal_map[el]]), marker='.', c=crel)
	
	print(el,'DMRG',plot_DMRG[Ninit_map[el]:Nfinal_map[el]])
	print(el,'BCS',plot_BCS[Ninit_map[el]:Nfinal_map[el]])
	#print(plot_N_BCS)
	#print(plot_BCS)

axL[1].set_xlabel('$N$')
axL[0].set_ylabel('absolute error [MeV]', color=cabs)
axR[0].set_ylabel('relative error', color=crel)
axL[1].set_ylabel('absolute error [MeV]', color=cabs)
axR[1].set_ylabel('relative error', color=crel)

#|\Delta^{(3)}_{\\text{DMRG}}-\Delta^{(3)}_{\\text{BCS}}|

axL[0].tick_params(axis='y', labelcolor=cabs)
axR[0].tick_params(axis='y', labelcolor=crel)
axL[1].tick_params(axis='y', labelcolor=cabs)
axR[1].tick_params(axis='y', labelcolor=crel)

axL[0].text(53, 0.25, 'Sn')
axL[1].text(101, 0.1, 'Pb')

#axL[0].grid()
#axL[1].grid()

if args.save:
	#plt.savefig('deviation_BCS_DMRG.png')
	plt.savefig('11_BCS.pdf')

plt.show()

