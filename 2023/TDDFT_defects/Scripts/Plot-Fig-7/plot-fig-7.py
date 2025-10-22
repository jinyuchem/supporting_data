#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.patches import ConnectionPatch
import numpy.polynomial.hermite as Herm
import math

import os
path = os.getcwd()
os.chdir(path)

########
# Plot #
########

fig = plt.figure(figsize=(11, 8))

gs = GridSpec(nrows=2, ncols=7, height_ratios=[1.1, 1],
                                width_ratios=[1, 1, 0.5, 0.5, 0.5, 1, 0.5],
                                hspace=0.1, wspace=0.6,
                                left=0.05, right=0.9,
                                bottom=0.02, top=0.98)

ax1 = fig.add_subplot(gs[0,0:3])
ax2 = fig.add_subplot(gs[0,4:6])
ax3 = fig.add_subplot(gs[1,0:2])
ax4 = fig.add_subplot(gs[1,3:7])

#######
# MgO #
#######

ax1.axis('off')
ax1.set_xlim((0,1))
ax1.set_ylim((0,1))
cwd = os.getcwd()
im1 = plt.imread(get_sample_data(path + '/MgO.png'))
newax = fig.add_axes([0.05, 0.64, 0.26, 0.26], anchor='NE')
newax.imshow(im1)
newax.axis('off')
ax1.text(x=-0.1, y=1.0, s='(a)', fontsize=15)

#####################
# SiV defect levels #
#####################

# DDH
hhPS = np.array([6.5641, 15.1744])

hhup = np.array([
10.1404, 15.1744, 16.6253, 16.6253, 16.6253,
])

hhup = hhup - hhPS[0]

my_ylim = np.array([-1.5, 11])

ax2.fill_between(np.linspace(0,1,10), [my_ylim[0] for i in range(10)], [hhPS[0]-hhPS[0] for i in range(10)], color='dodgerblue', alpha=0.3)
ax2.fill_between(np.linspace(0,1,10), [my_ylim[1] for i in range(10)], [hhPS[1]-hhPS[0] for i in range(10)], color='grey', alpha=0.3)

ax2.text(x=0.87, y=9, s='CB', ha='center')
ax2.text(x=0.87, y=-1.2, s='VB', ha='center')

ax2.axhline(y=hhup[0], xmin=0.43, xmax=0.57, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[1], xmin=0.43, xmax=0.57, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[2], xmin=0.21, xmax=0.35, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[3], xmin=0.43, xmax=0.57, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[4], xmin=0.65, xmax=0.79, color='black', linestyle='-', linewidth=1.5)

ax2.arrow(0.47, hhup[0]-0.33, 0., 0.6, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.53, hhup[0]+0.33, 0., -0.6, head_width=0.02, head_length=0.1, color='red')

ax2.text(x=0.5, y=hhup[0]+0.6, s='$a_{1g}$', va='center', ha='center', color='red', fontsize=12)
#ax2.text(x=0.5, y=hhup[1]+0.5, s='$a_{1g}^{\prime}$', va='center', ha='center', color='red', fontsize=12)
ax2.text(x=0.5, y=hhup[2]+0.5, s='$t_{1u}$', va='center', ha='center', color='red', fontsize=12)
#ax2.text(x=0.75, y=hhup[3]+0.2, s='$e_{gy}$', va='center', color='red', fontsize=12)

ax2.set_xlim([0,1])
ax2.set_ylim(my_ylim)
ax2.tick_params(direction='in')
ax2.yaxis.set_ticks_position('both')
ax2.set_xticks([0.5])
#ax2.set_xticklabels(['V$_{\mathrm{O}}^0$ in MgO'], fontsize=13)
ax2.set_xticklabels([])

ax2.set_ylabel("Energy (eV)", labelpad=0, fontsize=13)
#ax2.set_yticklabels([])

ax2.text(x=-0.5, y=1.0 * 12.5 - 1.5, s='(b)', fontsize=15)

#######################
# MgO manybody states #
#######################

Singlets = np.array([
0.0,
4.1506758558545505,
5.343418280930429,
])

Singlets_labels = [
'$^1A_{1g}$',
'$^1A_{1g}^{\prime}$',
'$^1T_{1u}$',
]

Triplets = np.array([
3.793461369464244,
4.022242485581542
])

Triplets_labels = ['$^3T_{1u}$', '$^3A_{1g}^{\prime}$']

ax3.hlines(y=Triplets[0], xmin=6, xmax=9, linestyle='-', linewidth=1.5, color='k')
#ax3.hlines(y=Triplets[1], xmin=6, xmax=9, linestyle='-', linewidth=1.5, color='k')

for i in range(Triplets.shape[0]):
    if i==0: ax3.text(x=10, y=Triplets[i], s=Triplets_labels[i], va='center', ha='center', fontsize=12)

ax3.hlines(y=Singlets[0], xmin=1, xmax=4, linestyle='-', linewidth=1.5, color='k')
ax3.hlines(y=Singlets[1], xmin=1, xmax=4, linestyle='-', linewidth=1.5, color='lightgray')
ax3.hlines(y=Singlets[2], xmin=1, xmax=4, linestyle='-', linewidth=1.5, color='k')

for i in range(Singlets.shape[0]):
    if i != 1: ax3.text(x=5, y=Singlets[i], s=Singlets_labels[i], va='center', ha='center', fontsize=12)

ax3.text(x=2.5, y=-0.4, s='$S=0$', va='center', ha='center', fontsize=13, fontweight="bold")
ax3.text(x=7.5, y=-0.4, s='$S=1$', va='center', ha='center', fontsize=13, fontweight="bold")
ax3.set_ylim([-0.3, 6.])
#ax3.set_xlim([0,12])

ax3.text(x=-0.6, y=1.0 * 6.3 - 0.3, s='(c)', fontsize=15)

# Hide the right and top spines
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
# Hide ticks
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.tick_params(axis='both', which='both', bottom=False,
    top=False, left=False, right=False, labelbottom=False)

#######
# VEE #
#######

from scipy.optimize import curve_fit

def f1(x, a, b, c):
    f = a / x + b / x**3 + c
    return f

#
#pbecells = np.array([16, 54, 64, 128, 216, 250])
pbecells = np.array([54, 64, 128, 216, 250])
pbeelevels = np.array([
#[7.9836,   8.6766,  13.5877,  15.6724],
[8.1957,  10.2181,  13.1866,  14.8899],
[8.1728,  10.4276,  13.1672,  14.7810],
[8.2643,  10.6017,  12.9550,  14.5631],
[8.2893,  10.7002,  12.8764,  14.3310],
[8.2978,  10.7112,  12.8590,  14.3046]
])
pbevee1 = np.array([
#[0.38474522,   0.41835070],
[0.24345893,   0.38347330],
[0.23129586,   0.37169999],
[0.19070026,   0.31944233],
[0.17236905,   0.28345621],
[0.16832478,   0.27841971],
]) * 13.605662285137
# triplet
pbevee2 = np.array([
[0.20981065, 0.30433015],
[0.19227222, 0.27842814],
[0.16822887, 0.25917961],
[0.15693438, 0.24174941],
[0.15530267, 0.24145352],
]) * 13.605662285137


#
#ddhcells = np.array([16, 54, 64, 128, 216])
ddhcells = np.array([54, 64, 128, 216, 250, 512])
ddhelevels = np.array([
#[6.0764,   7.9065,  16.0135,  18.0460],
[6.4325,   9.7066,  15.5028,  17.1154],
[6.4372,   9.9159,  15.4525,  16.9736],
[6.5418,  10.0616,  15.2515,  16.8016],
[6.5641,  10.1404,  15.1744,  16.6253],
[6.5972,  10.1392,  15.1500,  16.5858],
[6.5957,  10.1755,  15.1108,  16.2541],
])
ddhvee1 = np.array([
#[0.45561288,   0.54791990],
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


colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
linestyles = ['', '', '-', ':', '-', ':']
#labels1 = ['$^3A_{2u}$', '$^3E_{u}$', '$^3E_{u}$', '$^3A_{1u}$']
#labels2 = ['$^3E_{g}$', '$^3E_{g}$']
#labels3 = ['$^3A_{1g}$', '$^3E_{g}^\prime$', '$^3E_{g}^\prime$', '$^3A_{2g}$']
#labels1 = ['$^1A_{1g}^{\prime}$', '$^1T_{1u}$']
labels1 = ['Bound Exciton', '$^1T_{1u}$']
labels2 = ['$^3A_{1g}^{\prime}$', '$^3T_{1u}$']

for i in range(1,2):
    print("ddh vee singlet")
    ax4.plot(1/ddhcells, ddhvee1[:,i], linewidth=1.5, linestyle=linestyles[0], marker='s',
            markersize=6, color=colors[i], label=labels1[i])
    ###
    x = 1 / ddhcells
    y = ddhvee1[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(m, c)
    xaxis = np.linspace(0,0.02,101)
    if i==1: ax4.plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i])
    ###

    print("ddh vee triplet")
    ax4.plot(1/ddhcells, ddhvee2[:,i], linewidth=1.5, linestyle=linestyles[0], marker='s',
            markerfacecolor='None', markersize=6, color=colors[i], label=labels2[i])
    ###
    x = 1 / ddhcells
    y = ddhvee2[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(m, c)
    xaxis = np.linspace(0,0.02,101)
    if i==1: ax4.plot(xaxis, xaxis * m + c, linestyle=':', c=colors[i])
    ###

for i in range(1):
    print("ddh vee bound exciton")
    ax4.plot(1/ddhcells, ddhvee1[:,i], linewidth=1.5, linestyle=linestyles[0], marker='o',
            markersize=6, color=colors[i], label=labels1[i])
    ###
    x = 1 / ddhcells
    y = ddhvee1[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(m, c)
    xaxis = np.linspace(0,0.02,101)
    if i==1: ax4.plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i])
    ###


#####
x = np.linspace(1e-6, 0.3, 10001)
y = 50.27106700592537 * x**3 + 4.810269969110053 - (10.59686111 * 2 / 4.19 * x)

print(4.810269969110053)

def func1(x, m, c):
    return x * m * np.exp(- 1 / x / 21) + c

def func2(x, m, c):
    return x * m * np.exp(- 1 / x / 42) + c

nebm3 = 10.59686111
nebc3 = 0.40216043
nebm4 = 10.59686111
nebc4 = 0.23371615
#ax4.plot(x**3, 50.27106700592537 * x**3 + 4.810269969110053 - func1(x * 2 / 4.1, nebm3, nebc3),
#              linestyle='--', color=colors[-2], label='$D=21$ Å')
ax4.plot(x**3, 50.27106700592537 * x**3 + 4.810269969110053 - func2(x * 2 / 4.1, nebm4, nebc4),
            linestyle='--', color=colors[0], label='')
print(4.810269969110053 - nebc4)
#####


ax4.legend(fontsize=13,loc='best',edgecolor='black',ncol=1)

ax4.set_xlabel('$1 / N_{\mathrm{atom}}$')

ax4.set_ylabel('$E$ (eV)', color = 'black')
ax4.spines['left'].set_color('black')
ax4.spines['right'].set_color('black')
ax4.tick_params(axis='y', colors='black')
ax4.tick_params(axis='both', direction='in')
ax4.tick_params(which='minor', direction='in')
ax4.xaxis.set_ticks_position('both')
ax4.yaxis.set_ticks_position('both')

ax4.set_xlim([0, 0.02])
ax4.set_ylim([3.6, 6.5])

ax4.set_xticks([0.000, 0.005, 0.010, 0.015, 0.020])

ax4.text(x=-0.004, y=1.0 * 2.9 + 3.6, s='(d)', fontsize=15)

plt.savefig('Fig-7.pdf',dpi=300,bbox_inches='tight' )
plt.show()
