#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from sklearn.metrics import auc
from scipy import constants

#########
# Input #
#########

fnames = [
'1A1-PL-All-B-1A1-3E.dat',
'1A1-PL-All-B-1A1-3E.dat-a1-modes.dat',
'1A1-SCa1-PL-All-B-1A1-3E.dat',
'1A1-SCa1-PL-All-B-1A1-3E.dat-a1-modes.dat',
'3A2-PL-All-B-3A2-3E.dat',
'3A2-PL-All-B-3A2-3E.dat-a1-modes.dat',
'3A2-SCa1-PL-All-B-3A2-3E.dat',
'3A2-SCa1-PL-All-B-3A2-3E.dat-a1-modes.dat',
         ]

########
# Data #
########

eneaxis = np.loadtxt(fnames[0], usecols=0)
lsp = []
for i in range(len(fnames)):
    p = np.loadtxt(fnames[i], usecols=1)
    lsp.append(p)
lsp = np.array(lsp)

for i in range(len(fnames)):

    print(sum(lsp[i]) / np.pi / constants.hbar * 1e-15 * constants.eV / 1000 * 0.1)
    lsp[i] = lsp[i] / auc(eneaxis,lsp[i]) * 1000

########
# Plot #
########

fig, ax = plt.subplots(2, 2, figsize=(12, 7))

colors = ['#4285F4', '#4285F4', '#DB4437', '#DB4437',
          '#4285F4', '#4285F4', '#DB4437', '#DB4437',]
linestyles = ['-', '--', '-', '--', '-', '--', '-', '--']
labels = [
          '$^3E \\to ^1A_1$, Total',
          '$^3E \\to ^1A_1$, $a_1$',
          '$^3E \\to ^1A_1$, Total, Scaled',
          '$^3E \\to ^1A_1$, $a_1$, Scaled',
          '$^3E \\to ^3A_2$, Total',
          '$^3E \\to ^3A_2$, $a_1$',
          '$^3E \\to ^3A_2$, Total, Scaled',
          '$^3E \\to ^3A_2$, $a_1$, Scaled',
         ]

# Experimental data
fname_exp = 'NV-3E-3A2-exp.csv'
r_exp_eneaxis = np.loadtxt(open(fname_exp, "rb"), delimiter=",",
                         usecols=1, skiprows=1) * 1000 - 1946.34
r_exp_lsp = np.loadtxt(open(fname_exp, "rb"), delimiter=",",
                         usecols=2, skiprows=1)

exp_eneaxis = r_exp_eneaxis[1544:14046]
exp_lsp = r_exp_lsp[1544:14046]

# normalization
exp_lsp = exp_lsp / (1945 + exp_eneaxis)**3
exp_lsp = exp_lsp / sum(exp_lsp) / (r_exp_eneaxis[1] - r_exp_eneaxis[0]) * 1000 * 0.9


for i in range(len(fnames)):
    if i in [4, 6]:
        ax[0][0].plot(eneaxis * 1e-3, lsp[i] / sum(lsp[i]) * 1000 / 0.1,
                color=colors[i], label=labels[i],
                linewidth=1.5, linestyle=linestyles[i])
    if i in [5, 7]:
        ax[1][0].plot(eneaxis * 1e-3, lsp[i] / sum(lsp[i]) * 1000 / 0.1,
                color=colors[i], label=labels[i],
                linewidth=1.5, linestyle=linestyles[i])
    if i in [0, 2]:
        ax[0][1].plot(eneaxis * 1e-3, lsp[i] / sum(lsp[i]) * 1000 / 0.1,
                color=colors[i], label=labels[i],
                linewidth=1.5, linestyle=linestyles[i])
    if i in [1, 3]:
        ax[1][1].plot(eneaxis * 1e-3, lsp[i] / sum(lsp[i]) * 1000 / 0.1,
                color=colors[i], label=labels[i],
                linewidth=1.5, linestyle=linestyles[i])

for i in range(1):
    ax[0][i].fill_between(- exp_eneaxis * 1e-3, exp_lsp, color='lightgray', alpha=1)
    ax[1][i].fill_between(- exp_eneaxis * 1e-3, exp_lsp, color='lightgray', alpha=1)

ax[0][0].text(x=-0.15, y=1.02, s='(a)', fontsize=14, color='k', transform=ax[0][0].transAxes)
ax[0][1].text(x=-0.15, y=1.02, s='(b)', fontsize=14, color='k', transform=ax[0][1].transAxes)
ax[1][0].text(x=-0.15, y=1.02, s='(c)', fontsize=14, color='k', transform=ax[1][0].transAxes)
ax[1][1].text(x=-0.15, y=1.02, s='(d)', fontsize=14, color='k', transform=ax[1][1].transAxes)



for i in range(2):
    for j in range(2):
        ax[i][j].grid(color='gray', linestyle='--', linewidth=0.5)
        if j==0: ax[i][j].set_xlim((-0.05, 0.48))
        if j==1: ax[i][j].set_xlim((-0.05, 0.48))
        ax[i][j].set_ylim((0., 10.0))
        ax[i][j].tick_params(direction='in')
        if j==0: ax[i][j].legend(fontsize=12,loc='upper right',edgecolor='black')
        if j==1: ax[i][j].legend(fontsize=12,loc='upper right',edgecolor='black')
        ax[i][j].xaxis.set_ticks_position('both')
        ax[i][j].yaxis.set_ticks_position('both')
        if j==0: ax[i][j].set_xlabel("$E_{\mathrm{ZPL}} - \Delta$ (eV)")
        if j==1: ax[i][j].set_xlabel("$\Delta$ (eV)")
        if j==0: ax[i][j].set_ylabel("$F_{\mathrm{PL}}(\Delta)$ (eV$^{-1}$)")
        if j==1: ax[i][j].set_ylabel("$F(\Delta)$ (eV$^{-1}$)")

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.2, hspace=0.3)

plt.savefig("figure_s6.pdf", bbox_inches='tight', dpi=300)
plt.show()
