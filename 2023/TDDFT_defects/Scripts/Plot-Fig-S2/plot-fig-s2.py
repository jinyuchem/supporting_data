#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})

#
pbecells = np.array([64, 216, 512, 1000])
pbevee1 = np.array([
[0.167660, 0.172234, 0.172234, 0.183848],
[0.122200, 0.129804, 0.129804, 0.147560],
[0.102586, 0.105868, 0.105868, 0.111546],
[0.088018, 0.088963, 0.088963, 0.090416],
]) * 13.605662285137

pbevee2 = np.array([
[0.120178, 0.120179],
[0.073590, 0.073590],
[0.062232, 0.062232],
[0.058042, 0.058043],
]) * 13.605662285137

pbevee3 = np.array([
[0.161742, 0.172457, 0.172457, 0.195814],
[0.090000, 0.092202, 0.092202, 0.102852],
[0.068892, 0.069922, 0.069922, 0.075747],
[0.061200, 0.061733, 0.061733, 0.064784]
]) * 13.605662285137

#
ddhcells = np.array([64, 216, 512, 1000])
ddhvee1 = np.array([
[0.180872, 0.185553, 0.185553, 0.197631],
[0.129695, 0.136306, 0.136306, 0.156029],
[0.119244, 0.124712, 0.124712, 0.139264],
[0.11404, 0.11890, 0.11898, 0.12892],
]) * 13.605662285137

ddhvee2 = np.array([
[0.136749, 0.136749],
[0.099014, 0.099014],
[0.096890, 0.096890],
[0.09956, 0.09956],
]) * 13.605662285137

ddhvee3 = np.array([
[0.166309, 0.174865, 0.174865, 0.195600],
[0.111920, 0.113750, 0.113750, 0.122764],
[0.102067, 0.102986, 0.102986, 0.107420],
[0.10302, 0.10375, 0.10375, 0.10613],
]) * 13.605662285137

########
# Plot #
########

fig, ax = plt.subplots(1, 2, figsize=(9,3.5))

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
linestyles = ['', '', '-', ':', '-', ':']
labels1 = ['$^3A_{2u}$', '$^3E_{u}$', '$^3E_{u}$', '$^3A_{1u}$']
labels2 = ['$^3E_{g}$', '$^3E_{g}$']
labels3 = ['$^3A_{1g}$', '$^3E_{g}^\prime$', '$^3E_{g}^\prime$', '$^3A_{2g}$']

for i in range(4):
    if i == 2: continue
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
        ax[i].legend(fontsize=12,loc='lower right',edgecolor='black')

        if i != -1: ax[i].set_xlabel('$1 / N_{\mathrm{atom}}$')

        if i != -1: ax[i].set_ylabel('$E$ (eV)', color = 'black')
        ax[i].spines['left'].set_color('black')
        ax[i].spines['right'].set_color('black')
        ax[i].tick_params(axis='y', colors='black')
        ax[i].tick_params(axis='both', direction='in')
        ax[i].tick_params(which='minor', direction='in')
        ax[i].xaxis.set_ticks_position('both')
        ax[i].yaxis.set_ticks_position('both')

        ax[i].set_xlim([0, 0.02])
        ax[i].set_ylim([0.75, 3])

fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.4, hspace=0.2)

plt.savefig("Fig-S2.pdf",bbox_inches = 'tight',dpi=300)
plt.show()
