#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})

pbecells = np.array([54, 64, 128, 216, 250, 512])
# KS orbitals levels
pbeelevels = np.array([
[8.1957,  10.2181,  13.1866,  14.8899],
[8.1728,  10.4276,  13.1672,  14.7810],
[8.2643,  10.6017,  12.9550,  14.5631],
[8.2893,  10.7002,  12.8764,  14.3310],
[8.2978,  10.7112,  12.8590,  14.3046],
[8.3179,  10.7470,  12.8046,  13.8947],
])
# singlet
pbevee1 = np.array([
[0.24345893,   0.38347330],
[0.23129586,   0.37169999],
[0.19070026,   0.31944233],
[0.17236905,   0.28345621],
[0.16832478,   0.27841971],
[0.15626965,   0.23473187],
]) * 13.605662285137
# triplet
pbevee2 = np.array([
[0.20981065, 0.30433015],
[0.19227222, 0.27842814],
[0.16822887, 0.25917961],
[0.15693438, 0.24174941],
[0.15530267, 0.24145352],
[0.15004643, 0.22379908],
]) * 13.605662285137

#
ddhcells = np.array([54, 64, 128, 216, 250, 512])
# KS orbital levels
ddhelevels = np.array([
[6.4325,   9.7066,  15.5028,  17.1154],
[6.4372,   9.9159,  15.4525,  16.9736],
[6.5418,  10.0616,  15.2515,  16.8016],
[6.5641,  10.1404,  15.1744,  16.6253],
[6.5972,  10.1392,  15.1500,  16.5858],
[6.5957,  10.1755,  15.1108,  16.2541],
])
# singlet
ddhvee1 = np.array([
[0.33034870,   0.45224022],
[0.32023619,   0.43116656],
[0.30539692,   0.40844187],
[0.30506974,   0.39273489],
[0.30484900,   0.39101599],
[0.31127465,   0.37858380],
]) * 13.605662285137
# triplet
ddhvee2 = np.array([
[0.30560467, 0.32710401],
[0.29297110, 0.29949725],
[0.29145185, 0.28881069],
[0.29563004, 0.27881490],
[0.29682947, 0.27999923],
[0.30655133, 0.28040133],
]) * 13.605662285137

print(ddhvee1[:,1] - ddhvee2[:,1])

########
# Plot #
########

fig, ax = plt.subplots(1, 2, figsize=(9,3.5))

colors = ['', '#4285F4', '#DB4437', '#F4B400', '#0F9D58']
linestyles = ['', '', '-', ':', '-', ':']
labels1 = ['$^1A_{1g}$', '$^1T_{1u}$']
labels2 = ['$^3A_{1g}$', '$^3T_{1u}$']
labels3 = ['$\\varepsilon_p - \\varepsilon_s$', '$\\varepsilon_{CBM} - \\varepsilon_s$']

for i in range(1,2,1):
    print("pbe vee singlet")
    ax[0].plot(1/pbecells, pbevee1[:,i], linewidth=1.5, linestyle=linestyles[0], marker='s',
                  markersize=6, color=colors[i], label=labels1[i])    
    ###
    x = 1 / pbecells
    y = pbevee1[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(m, c)
    xaxis = np.linspace(0,0.02,101)
    ax[0].plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i])
    ###

    print("pbe vee triplet")
    ax[0].plot(1/pbecells, pbevee2[:,i], linewidth=1.5, linestyle=linestyles[0], marker='o',
                  markersize=7, markerfacecolor='None', color=colors[i+1], label=labels2[i])
    ###
    x = 1 / pbecells
    y = pbevee2[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(m, c)
    xaxis = np.linspace(0,0.02,101)
    ax[0].plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i+1])
    ###



    print("ddh vee singlet")
    ax[1].plot(1/ddhcells, ddhvee1[:,i], linewidth=1.5, linestyle=linestyles[0], marker='s',
                  markersize=6, color=colors[i], label=labels1[i])
    ###
    x = 1 / ddhcells
    y = ddhvee1[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(m, c)
    xaxis = np.linspace(0,0.02,101)
    ax[1].plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i])
    ###

    print("ddh vee triplet")
    ax[1].plot(1/ddhcells, ddhvee2[:,i], linewidth=1.5, linestyle=linestyles[0], marker='o',
                  markersize=7, markerfacecolor='None', color=colors[i+1], label=labels2[i])
    ###
    x = 1 / ddhcells
    y = ddhvee2[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(m, c)
    xaxis = np.linspace(0,0.02,101)
    ax[1].plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i+1])
    ###

ax[0].text(x=-0.25, y=1.03, s='(a)', fontsize=15,
              transform=ax[0].transAxes)
ax[1].text(x=-0.25, y=1.03, s='(b)', fontsize=15,
              transform=ax[1].transAxes)

for i in range(2):
    for j in range(1):
        if i==0 or i==1: ax[i].legend(fontsize=12,loc='upper left',edgecolor='black')
    
        if i==0 or i==1: ax[i].set_xlabel('$1 / N_{\mathrm{atom}}$')
    
        ax[i].set_ylabel('$E$ (eV)', color = 'black')
        ax[i].spines['left'].set_color('black')
        ax[i].spines['right'].set_color('black')
        ax[i].tick_params(axis='both', colors='black')
        ax[i].tick_params(axis='both', direction='in')
        ax[i].tick_params(which='minor', direction='in')
        ax[i].xaxis.set_ticks_position('both')
        ax[i].yaxis.set_ticks_position('both')
    
        ax[i].set_xlim([0, 0.02])
        ax[i].set_ylim([1.8, 8])
    
        ax[i].set_xticks([0.000, 0.005, 0.010, 0.015, 0.020])

fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.4, hspace=0.2)

plt.savefig("Fig-S5.pdf",bbox_inches = 'tight',dpi=300)
plt.show()
