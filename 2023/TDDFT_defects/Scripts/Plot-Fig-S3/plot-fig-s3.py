#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})

#
pbecells = np.array([64, 216, 512, 1000])
pbevee1 = np.array([
[0.        , 0.01395603, 0.01395621, 0.02834675],
[0.        , 0.01349335, 0.01349411, 0.02657385],
[0.        , 0.01450997, 0.01450997, 0.02840069],
[0.        , 0.01456018, 0.01456516, 0.02804203]
]) * 13.605662285137

#
ddhcells = np.array([64, 216, 512, 1000])
ddhvee1 = np.array([
[0.        , 0.02014853, 0.02014853, 0.04089616],
[0.        , 0.02219697, 0.02220394, 0.04377267],
[0.        , 0.02353267, 0.02353268, 0.04595365],
[0.        , 0.02489256, 0.02489259, 0.05060581],
]) * 13.605662285137

print('pbe vee')
print(pbevee1)
print('*' * 45)

print('ddh vee')
print(ddhvee1)
print('*' * 45)

########
# Plot #
########

fig, ax = plt.subplots(1, 2, figsize=(9,3.5))

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
linestyles = ['', '', '-', ':', '-', ':']
labels1 = ['$^3A_{2g}$', '$^1E_{g}$', '$^1E_{g}$', '$^1A_{1g}$']

for i in range(4):
    if i == 0 or  i == 2: continue
    ax[0].plot(1/pbecells, pbevee1[:,i], linewidth=1.5, linestyle=linestyles[0], marker='s',
                  markersize=6, color=colors[i], label=labels1[i])
    ###
    x = 1 / pbecells
    y = pbevee1[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print('pbe', labels1[i], c)
    xaxis = np.linspace(0,0.02,101)
    ax[0].plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i])
    ###

    ax[1].plot(1/ddhcells, ddhvee1[:,i], linewidth=1.5, linestyle=linestyles[0], marker='s',
                  markersize=6, color=colors[i], label=labels1[i])
    ###
    x = 1 / ddhcells
    y = ddhvee1[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print('ddh', labels1[i], c)
    xaxis = np.linspace(0,0.02,101)
    ax[1].plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i])
    ###

ax[0].text(x=-0.25, y=1.03, s='(a)', fontsize=15,
              transform=ax[0].transAxes)
ax[1].text(x=-0.25, y=1.03, s='(b)', fontsize=15,
              transform=ax[1].transAxes)

for i in range(2):
    for j in range(1):
        ax[i].legend(fontsize=12,loc='best',edgecolor='black')

        ax[i].set_xlabel('$1 / N_{\mathrm{atom}}$')

        if i != -1: ax[i].set_ylabel('$E$ (eV)', color = 'black')
        ax[i].spines['left'].set_color('black')
        ax[i].spines['right'].set_color('black')
        ax[i].tick_params(axis='y', colors='black')
        ax[i].tick_params(axis='both', direction='in')
        ax[i].tick_params(which='minor', direction='in')
        ax[i].xaxis.set_ticks_position('both')
        ax[i].yaxis.set_ticks_position('both')

        ax[i].set_xlim([0, 0.016])
        ax[i].set_ylim([0., 0.7])

fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.4, hspace=0.2)

plt.savefig("Fig-S3.pdf",bbox_inches = 'tight',dpi=300)
plt.show()
