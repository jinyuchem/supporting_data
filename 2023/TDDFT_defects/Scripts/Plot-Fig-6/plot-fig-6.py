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
                                hspace=0.1, wspace=0.5,
                                left=0.05, right=0.9, 
                                bottom=0.02, top=0.98)

ax1 = fig.add_subplot(gs[0,0:3])
ax2 = fig.add_subplot(gs[0,4:6])
ax3 = fig.add_subplot(gs[1,0:2])
ax4 = fig.add_subplot(gs[1,3:7])

#######
# SiV #
#######

ax1.axis('off')
ax1.set_xlim((0,1))
ax1.set_ylim((0,1))
cwd = os.getcwd()
im1 = plt.imread(get_sample_data(path + '/SiV.png'))
newax = fig.add_axes([-0.09, 0.57, 0.54, 0.34], anchor='NE')
newax.imshow(im1)
newax.axis('off')
ax1.text(x=-0.099, y=1.0, s='(a)', fontsize=15)

#####################
# SiV defect levels #
#####################

# DDH
hhPS = np.array([12.7730, 18.3675]) + 12.8254 - 12.7730

hhup = np.array([
12.0342, # 1013
12.0342, # 1014
12.0411, # 1015
12.0411, # 1016
12.0987, # 1017
12.0987, # 1018
12.5897, # 1019
12.5897, # 1020
12.8213, # 1021
13.1447, # 1022
13.1447, # 1023
])
hhdown = np.array([
12.0371, # 1013
12.0371, # 1014
12.0496, # 1015
12.0496, # 1016
12.2614, # 1017
12.2614, # 1018
12.7467, # 1019
12.7467, # 1020
12.8254, # 1021
14.5828, # 1022
14.5828, # 1023
])

hhup = hhup - hhPS[0]
hhdown = hhdown - hhPS[0]

my_ylim = np.array([-1.3, 6.4])

ax2.fill_between(np.linspace(0,1,10), [my_ylim[0] for i in range(10)], [hhPS[0]-hhPS[0] for i in range(10)], color='dodgerblue', alpha=0.3)
ax2.fill_between(np.linspace(0,1,10), [my_ylim[1] for i in range(10)], [hhPS[1]-hhPS[0] for i in range(10)], color='grey', alpha=0.3)
ax2.axvline(x=0.5, ymin=0, ymax=1, color='black', linestyle='--', linewidth=0.5)

ax2.text(x=0.87, y=5.8, s='CB', ha='center')
ax2.text(x=0.87, y=-1.2, s='VB', ha='center')

#ax2.axhline(y=hhup[0], xmin=0.07, xmax=0.19, color='black', linestyle='-', linewidth=1.5)
#ax2.axhline(y=hhup[1], xmin=0.31, xmax=0.43, color='black', linestyle='-', linewidth=1.5)
#ax2.axhline(y=hhup[2], xmin=0.07, xmax=0.19, color='black', linestyle='-', linewidth=1.5)
#ax2.axhline(y=hhup[3], xmin=0.31, xmax=0.43, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[4], xmin=0.07, xmax=0.19, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[5], xmin=0.31, xmax=0.43, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[6], xmin=0.07, xmax=0.19, color='gray', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[7], xmin=0.31, xmax=0.43, color='gray', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[8], xmin=0.19, xmax=0.31, color='gray', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[9], xmin=0.07, xmax=0.19, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[10], xmin=0.31, xmax=0.43, color='black', linestyle='-', linewidth=1.5)

#ax2.arrow(0.13, hhup[0]-0.23, 0., 0.4, head_width=0.02, head_length=0.1, color='red')
#ax2.arrow(0.37, hhup[1]-0.23, 0., 0.4, head_width=0.02, head_length=0.1, color='red')
#ax2.arrow(0.13, hhup[2]-0.23, 0., 0.4, head_width=0.02, head_length=0.1, color='red')
#ax2.arrow(0.37, hhup[3]-0.23, 0., 0.4, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.13, hhup[4]-0.23, 0., 0.3, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.37, hhup[5]-0.23, 0., 0.3, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.13, hhup[6]-0.23, 0., 0.3, head_width=0.02, head_length=0.1, color='red', alpha=0.6)
ax2.arrow(0.37, hhup[7]-0.23, 0., 0.3, head_width=0.02, head_length=0.1, color='red', alpha=0.6)
ax2.arrow(0.25, hhup[8]-0.23, 0., 0.3, head_width=0.02, head_length=0.1, color='red', alpha=0.6)
ax2.arrow(0.13, hhup[9]-0.23, 0., 0.3, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.37, hhup[10]-0.23, 0., 0.3, head_width=0.02, head_length=0.1, color='red')

ax2.text(x=0.0, y=hhup[4]+0.2, s='$e_{ux}$', va='center', color='red', fontsize=12)
ax2.text(x=0.25, y=hhup[5]+0.2, s='$e_{uy}$', va='center', color='red', fontsize=12)
ax2.text(x=0.0, y=hhup[9]+0.2, s='$e_{gx}$', va='center', color='red', fontsize=12)
ax2.text(x=0.25, y=hhup[10]+0.2, s='$e_{gy}$', va='center', color='red', fontsize=12)

#ax2.axhline(y=hhdown[0], xmin=0.57, xmax=0.69, color='black', linestyle='-', linewidth=1.5)
#ax2.axhline(y=hhdown[1], xmin=0.81, xmax=0.93, color='black', linestyle='-', linewidth=1.5)
#ax2.axhline(y=hhdown[2], xmin=0.57, xmax=0.69, color='black', linestyle='-', linewidth=1.5)
#ax2.axhline(y=hhdown[3], xmin=0.81, xmax=0.93, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhdown[4], xmin=0.57, xmax=0.69, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhdown[5], xmin=0.81, xmax=0.93, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhdown[6], xmin=0.57, xmax=0.69, color='gray', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhdown[7], xmin=0.81, xmax=0.93, color='gray', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhdown[8], xmin=0.69, xmax=0.81, color='gray', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhdown[9], xmin=0.57, xmax=0.69, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhdown[10], xmin=0.81, xmax=0.93, color='black', linestyle='-', linewidth=1.5)

#ax2.arrow(0.63, hhdown[0]+0.23, 0., -0.3, head_width=0.02, head_length=0.1, color='red')
#ax2.arrow(0.87, hhdown[1]+0.23, 0., -0.3, head_width=0.02, head_length=0.1, color='red')
#ax2.arrow(0.63, hhdown[2]+0.23, 0., -0.3, head_width=0.02, head_length=0.1, color='red')
#ax2.arrow(0.57, hhdown[3]+0.23, 0., -0.3, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.63, hhdown[4]+0.23, 0., -0.3, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.87, hhdown[5]+0.23, 0., -0.3, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.63, hhdown[6]+0.23, 0., -0.3, head_width=0.02, head_length=0.1, color='red', alpha=0.6)
ax2.arrow(0.87, hhdown[7]+0.23, 0., -0.3, head_width=0.02, head_length=0.1, color='red', alpha=0.6)
ax2.arrow(0.75, hhdown[8]+0.23, 0., -0.3, head_width=0.02, head_length=0.1, color='red', alpha=0.6)

ax2.text(x=0.5, y=hhdown[4]+0.2, s='$\overline{e}_{ux}$', va='center', color='red', fontsize=12)
ax2.text(x=0.75, y=hhdown[5]+0.2, s='$\overline{e}_{uy}$', va='center', color='red', fontsize=12)
ax2.text(x=0.5, y=hhdown[9]+0.2, s='$\overline{e}_{gx}$', va='center', color='red', fontsize=12)
ax2.text(x=0.75, y=hhdown[10]+0.2, s='$\overline{e}_{gy}$', va='center', color='red', fontsize=12)

ax2.set_xlim([0,1])
ax2.set_ylim(my_ylim)
ax2.tick_params(direction='in')
ax2.yaxis.set_ticks_position('both')
ax2.set_xticks([0.5])
#ax2.set_xticklabels(['SiV$^0$ in diamond'], fontsize=13)
ax2.set_xticklabels([])

ax2.set_ylabel("Energy (eV)", labelpad=0, fontsize=13)
#ax2.set_yticklabels([])

ax2.text(x=-0.35, y=1.0 * 7.1 - 0.7, s='(b)', fontsize=15)

#######################
# SiV manybody states #
#######################

Triplets = np.array([
0.0,
0.09688959,# 0.09688961,
0.10206681,
0.10298595,# 0.10298595,
0.10742001,
0.11924369,
0.12471158,# 0.12471158,
0.13926442,
]) * 13.605662285137

Triplets_labels = [
'$^3A_{2g}$',
'$^3E_{g}$',
'$^3A_{1g}$',
'$^3E^\prime_{g}$',
'$^3A_{2g}$',
'$^3A_{2u}$',
'$^3E_{u}$',
'$^3A_{1u}$',
]

Singlets = np.array([0.308, 0.613])

Singlets_labels = ['$^1E_{g}$', '$^1A_{1g}$',]


ax3.hlines(y=Triplets[0], xmin=1, xmax=4, linestyle='-', linewidth=1.5, color='k')
ax3.hlines(y=Triplets[1], xmin=1, xmax=4, linestyle='-', linewidth=1.5, color='lightgray')
ax3.hlines(y=Triplets[2], xmin=1, xmax=4, linestyle='-', linewidth=1.5, color='lightgray')
ax3.hlines(y=Triplets[3], xmin=1, xmax=4, linestyle='-', linewidth=1.5, color='lightgray')
ax3.hlines(y=Triplets[4], xmin=1, xmax=4, linestyle='-', linewidth=1.5, color='lightgray')
ax3.hlines(y=Triplets[5], xmin=1, xmax=4, linestyle='-', linewidth=1.5, color='k')
ax3.hlines(y=Triplets[6], xmin=1, xmax=4, linestyle='-', linewidth=1.5, color='k')
ax3.hlines(y=Triplets[7], xmin=1, xmax=4, linestyle='-', linewidth=1.5, color='k')

for i in range(Triplets.shape[0]):
    if i in [0, 5, 6, 7]:
        ax3.text(x=5, y=Triplets[i], s=Triplets_labels[i], va='center', ha='center', fontsize=12)

ax3.hlines(y=Singlets[0], xmin=6, xmax=9, linestyle='-', linewidth=1.5, color='k')
ax3.hlines(y=Singlets[1], xmin=6, xmax=9, linestyle='-', linewidth=1.5, color='k')

for i in range(Singlets.shape[0]):
    ax3.text(x=5, y=Singlets[i], s=Singlets_labels[i], va='center', ha='center', fontsize=12)

ax3.text(x=2.5, y=-0.2, s='$S=1$', va='center', ha='center', fontsize=13, fontweight="bold")
ax3.text(x=7.5, y=-0.2, s='$S=0$', va='center', ha='center', fontsize=13, fontweight="bold")
ax3.set_ylim([-0.3, 2.])

ax3.text(x=-0.61, y=1.0 * 2.3 - 0.3, s='(c)', fontsize=15)

# Hide the right and top spines
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
# Hide ticks
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.tick_params( axis='both', which='both', bottom=False,
    top=False, left=False, right=False, labelbottom=False)

#####################
# VEE extrapolation #
#####################

from scipy.optimize import curve_fit

def f1(x, a, b, c):
    f = a / x + b / x**3 + c
    return f

#
ddhcells = np.array([64, 216, 512, 1000])
ddhvee1 = np.array([
[0.180872, 0.185553, 0.185553, 0.197631],
[0.129695, 0.136306, 0.136306, 0.156029],
[0.119244, 0.124712, 0.124712, 0.139264],
# 512 ace 0.01
#[0.11748, 0.12339, 0.12347, 0.13900],
# 512 ace 0.001
#[0.11837, 0.12424, 0.12432, 0.13956],
# 1000 ace 0.01
[0.11404, 0.11890, 0.11898, 0.12892],
]) * 13.605662285137

ddhvee2 = np.array([
[0.136749, 0.136749],
[0.099014, 0.099014],
[0.096890, 0.096890],
# 512 ace 0.01
#[0.09691, 0.09692],
# 512 ace 0.001
#[0.09688, 0.09688],
# 1000 ace 0.01
[0.09956, 0.09956],
]) * 13.605662285137

ddhvee3 = np.array([
[0.166309, 0.174865, 0.174865, 0.195600],
[0.111920, 0.113750, 0.113750, 0.122764],
[0.102067, 0.102986, 0.102986, 0.107420],
# 512 ace 0.01
#[0.10317, 0.10457, 0.10457, 0.10936],
# 512 ace 0.001
#[0.10311, 0.10451, 0.10451, 0.10928],
# 1000 ace 0.01
[0.10302, 0.10375, 0.10375, 0.10613],
]) * 13.605662285137

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58', '#F4B400']
linestyles = ['', '', '-', ':', '-', ':']
labels1 = ['$^3A_{2u}$', '$^3E_{u}$', '$^3E_{u}$', '$^3A_{1u}$',
           'Bound Exciton']
labels2 = ['$^3E_{g}$', '$^3E_{g}$']
labels3 = ['$^3A_{1g}$', '$^3E_{g}^\prime$', '$^3E_{g}^\prime$', '$^3A_{2g}$']

for i in range(4):
    if i == 2: continue
    ax4.plot(1/ddhcells, ddhvee1[:,i], linewidth=1.5, linestyle=linestyles[0], marker='s',
                  markersize=6, color=colors[i], label=labels1[i])
    ###
    x = 1 / ddhcells
    y = ddhvee1[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print('ddh', labels1[i], c)
    xaxis = np.linspace(0,0.02,101)
    ax4.plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i])
    ###
for i in range(2):
    if i == 1: continue
    ax4.plot(1/ddhcells, ddhvee2[:,i], linewidth=1.5, linestyle=linestyles[0], marker='s',
                  markersize=6, color=colors[-1], label=labels1[-1])
    ###
    xdata = np.power(ddhcells/8, 1/3) * 3.55
    ydata = ddhvee2[:,0]
    popt, pcov = curve_fit(f1, xdata, ydata)
    print(popt)
    residuals = ydata - f1(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata - np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('r_squared', r_squared)

    xaxis = np.linspace(8, 64000, 10001)
    my_xaxis = np.power((xaxis/8), 1/3) * 3.55
    ax4.plot(1/xaxis, f1(my_xaxis, *popt), linestyle='--', c=colors[-1])
    ###

ax4.legend(fontsize=13,loc='best',edgecolor='black')

ax4.set_xlabel('$1 / N_{\mathrm{atom}}$',fontsize=13)

ax4.set_xticks([0, 0.005, 0.01, 0.015, 0.02])

ax4.set_ylabel('$E$ (eV)', color='black', fontsize=13)
ax4.spines['left'].set_color('black')
ax4.spines['right'].set_color('black')
ax4.tick_params(axis='y', colors='black')
ax4.tick_params(axis='both', direction='in')
ax4.tick_params(which='minor', direction='in')
ax4.xaxis.set_ticks_position('both')
ax4.yaxis.set_ticks_position('both')

ax4.set_xlim([0, 0.02])
ax4.set_ylim([1.1, 3])

ax4.text(x=-0.004, y= 1.0 * 1.9 + 1.1, s='(d)', fontsize=15)

plt.savefig('Fig-6.pdf',dpi=300,bbox_inches='tight' )
plt.show()
