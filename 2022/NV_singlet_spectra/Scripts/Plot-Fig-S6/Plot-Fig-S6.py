#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from matplotlib.gridspec import GridSpec
import os
from sklearn.metrics import auc
from matplotlib.cbook import get_sample_data

fig = plt.figure(figsize=(11, 3))

gs = GridSpec(nrows=1, ncols=2, height_ratios=[1],
                                width_ratios=[1, 1],
                                hspace=0.3, wspace=0.2,
                                left=0.05, right=0.9,
                                bottom=0.02, top=0.98)

ax00 = fig.add_subplot(gs[0,0])
ax01 = fig.add_subplot(gs[0,1])

################
# PL 3E -> 3A2 #
################

fnames = [
'PBE-dq-PBE-ph-PL-All-B-3A2-3E-13823-rc2-5-cph-5.dat',
'DDH-dq-PBE-ph-PL-All-B-3A2-3E-13823-rc2-5-cph-5.dat',
'DDH-dq-DDH-ph-PL-All-B-3A2-3E-13823-rc2-5-cph-5.dat'
]

# energy of the ZPL: meV
EZPL = 1945

eneaxis = np.loadtxt(fnames[0], usecols=0)
lsp = []
for i in range(len(fnames)):
    p = np.loadtxt(fnames[i], usecols=1)
    lsp.append(p)
lsp = np.array(lsp)

# \omega^3 prefactor
for i in range(len(fnames)):
    lsp[i,:] = lsp[i,:] * (1945 - eneaxis[:])**3
    # normalization
    lsp[i] = lsp[i] / sum(lsp[i]) / (eneaxis[1] - eneaxis[0]) * 1000

# Experimental data
fname_exp = 'NV-3E-3A2-exp.csv'
r_exp_eneaxis = np.loadtxt(open(fname_exp, "rb"), delimiter=",",
                         usecols=1, skiprows=1) * 1000 - 1946.34
r_exp_lsp = np.loadtxt(open(fname_exp, "rb"), delimiter=",",
                         usecols=2, skiprows=1)

exp_eneaxis = r_exp_eneaxis[1544:14046]
exp_lsp = r_exp_lsp[1544:14046]

# normalization
exp_lsp = exp_lsp / sum(exp_lsp) / (r_exp_eneaxis[1] - r_exp_eneaxis[0]) * 1000

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
labels = [
           'PBE$-\Delta Q$, PBE$-ph$',
           'DDH$-\Delta Q$, PBE$-ph$',
           'DDH$-\Delta Q$, DDH$-ph$',
         ]
linestyles = ['-', '-', '-', '--', '--', '--']

for i in range(len(fnames)):
    if i==0 or i==1 or i==2 or i==3:
        ax00.plot(-eneaxis, lsp[i], color=colors[i], label=labels[i], linewidth=1.5, linestyle=linestyles[i])
ax00.fill_between(exp_eneaxis, exp_lsp, color='gray', label='Expt.', alpha=0.4)

ax00.set_xlim((-400,50))
ax00.set_ylim((0,7))
ax00.set_yticklabels([])
ax00.set_xticks([-400,-300,-200,-100,0])
ax00.tick_params(direction='in')
ax00.legend(fontsize=12,loc='upper left',edgecolor='black', labelspacing=0.2, handlelength=1.5, borderpad=0.2)
ax00.xaxis.set_ticks_position('both')
ax00.yaxis.set_ticks_position('both')
ax00.set_xlabel( "Energy Shift from ZPL (meV)" )
ax00.set_ylabel( "Photoluminescence (arb. unit)" )

ax00.text(x=(50 - -400)*5/9 + -400, y=7*0.8, s='$^3E \\to ^3A_2$', fontsize=14)

ax00.text(x=-450, y=7*1.07, s='a', fontsize=15, weight='bold')

#################
# Abs 1E -> 1A1 #
#################

fnames = [
         'PBE-dq-PBE-ph-Abs-All-B-1A1-1E-13823-rc2-5-cph-5.dat',
         'DDH-dq-PBE-ph-Abs-All-B-1A1-1E-13823-rc2-5-cph-5.dat',
         'DDH-dq-DDH-ph-Abs-All-B-1A1-1E-13823-rc2-5-cph-5.dat'
         ]

# energy of the ZPL: meV
EZPL = 1190

eneaxis = np.loadtxt(fnames[0], usecols=0)
lsp = []
for i in range(len(fnames)):
    p = np.loadtxt(fnames[i], usecols=1)
    lsp.append(p)
lsp = np.array(lsp)

# \omega prefactor
for i in range(len(fnames)):
    lsp[i,:] = lsp[i,:] * (EZPL + eneaxis[:])
    # normalization
    lsp[i] = lsp[i] / auc(eneaxis,lsp[i]) * 1000

# Experimental data
fname_exp = 'NV-1E-1A1-exp.csv'
r_exp_eneaxis = np.loadtxt(open(fname_exp, "rb"), delimiter=",",
                         usecols=0, skiprows=1)
r_exp_lsp = np.loadtxt(open(fname_exp, "rb"), delimiter=",",
                         usecols=1, skiprows=1)

exp_eneaxis = r_exp_eneaxis[:]
exp_lsp = r_exp_lsp[:] * (EZPL + r_exp_eneaxis[:])

# normalization
nn = auc(exp_eneaxis,exp_lsp)
nn = nn / 0.6
exp_lsp = exp_lsp / nn * 1000

labels = [
          '$^1E \\to ^1A_1$, PBE$-\Delta Q$, PBE$-ph$',
          '$^1E \\to ^1A_1$, DDH$-\Delta Q$, PBE$-ph$',
          '$^1E \\to ^1A_1$, DDH$-\Delta Q$, DDH$-ph$'
         ]

for i in range(len(fnames)):
    if i==0 or i==1 or i==2:
        ax01.plot(eneaxis, lsp[i], color=colors[i], label='', linewidth=1.5, linestyle=linestyles[i])
ax01.fill_between(exp_eneaxis, exp_lsp, color='gray', label='Expt.$^b$', alpha=0.4)

ax01.set_xlim((-50,400))
ax01.set_xticks([0, 100, 200, 300, 400])
ax01.set_ylim((0,12))
ax01.set_yticklabels([])
ax01.tick_params(direction='in')
ax01.xaxis.set_ticks_position('both')
ax01.yaxis.set_ticks_position('both')
ax01.set_xlabel( "Energy Shift from ZPL (meV)" )
ax01.set_ylabel( "Absorption (arb. unit)" )

ax01.text(x=(400 - -50)*5/9 + -50, y=12*0.8, s='$^1E \\to ^1A_1$', fontsize=14)

ax01.text(x=-100, y=12*1.07, s='b', fontsize=15, weight='bold')

plt.savefig('Fig-S6.pdf',dpi=300,bbox_inches='tight')
plt.show()
