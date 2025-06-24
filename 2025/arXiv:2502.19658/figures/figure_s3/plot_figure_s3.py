#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})
import json
from scipy import constants

act_sp = np.genfromtxt('results.txt', usecols=0, dtype='str')
e_ks = np.loadtxt('results.txt', usecols=(1, 2))
lambda_z = np.loadtxt('results.txt', usecols=3)
lambda_perp = np.loadtxt('results.txt', usecols=4) / np.sqrt(2)
vees = np.loadtxt('results.txt', usecols=(5, 6, 7, 8))

########
# main #
########

fig, ax = plt.subplots(2, 1, figsize=(8, 6))
colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58', '#AB47BC',
          '#FF6F61', '#46BDC6', '#FF9E80', '#7E57C2', '#9CCC65']
labels = ['$^3A_2$', '$^1E$', '$^1A_1$', '$^3E$']
markers = ['s', 'o', '^', 'v']

xaxis = np.asarray([0, 3, 5, 8, 12, 16, 24, 32, 40, 48, 56, 62])

# vee
for i in range(vees.shape[1]):
    ax[0].plot(xaxis[1:], vees[1:, i], color=colors[i], label=labels[i], marker=markers[i],
               markersize=6, linestyle='-', linewidth=0.5, markerfacecolor='none', markeredgecolor=colors[i])

# lambda
ax[1].plot(xaxis[1:], lambda_z[1:], color=colors[4], marker='s', label='$\lambda_z^{\mathrm{QDET}}$',
           markersize=6, linestyle='-', linewidth=0.)
ax[1].plot(xaxis[1:], lambda_perp[1:], color=colors[6], marker='o', label='$\lambda_\perp^{\mathrm{QDET}}$',
           markersize=6, linestyle='-', linewidth=0.)

# band for range
ax[1].fill_between([0, 70], y1=22.898730432785417, y2=33.35775985509086, color='#E6BDE4')#colors[4], alpha=0.3)
ax[1].fill_between([0, 70], y1=49.148115995000154 / np.sqrt(2), y2=59.006625217051685 / np.sqrt(2), color='#BDE7EA')#colors[6], alpha=0.3)

ax[1].axhline(y=lambda_z[0], xmin=0, xmax=1, color=colors[4],
              linestyle='--', linewidth=1.0, label='$\lambda_z^{\mathrm{DFT}}$')
ax[1].axhline(y=lambda_perp[0], xmin=0, xmax=1, color=colors[6],
              linestyle='--', linewidth=1.0, label='$\lambda_\perp^{\mathrm{DFT}}$')

for i in range(2):
    ax[i].legend(fontsize=10, loc='upper right', edgecolor='black', ncol=2)

    if i==1: ax[i].set_xlabel('Number of orbitals in active space')
    ax[i].set_xticks([0, 3, 5, 8, 12, 16, 24, 32, 40, 48, 56, 62])
    if i==0: ax[i].set_ylabel('Excitation Energy (eV)', color='black')
    if i==1: ax[i].set_ylabel('$\lambda$ (GHz)', color='black')
    ax[i].tick_params(axis='y', colors='black')
    ax[i].tick_params(axis='both', direction='in')
    ax[i].tick_params(which='minor', direction='in')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].yaxis.set_ticks_position('both')
    ax[i].set_xlim([0, 68])
    if i==0: ax[i].set_ylim([-0.2, 3.4])
    if i==1: ax[i].set_ylim([0, 100])

ax[0].text(x=-0.12, y=1.02, s='(a)', fontsize=14, color='k', transform=ax[0].transAxes)
ax[1].text(x=-0.12, y=1.02, s='(b)', fontsize=14, color='k', transform=ax[1].transAxes)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.0, hspace=0.2)

plt.savefig("figure_s3.pdf", bbox_inches='tight', dpi=300)
plt.show()
