#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'font.family': 'Helvetica'})
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.patches import ConnectionPatch
import numpy.polynomial.hermite as Herm
import math
import json


import os
path = os.getcwd()
os.chdir(path)

########
# Plot #
########

fig = plt.figure(figsize=(3.25, 1.75))

gs = GridSpec(nrows=1, ncols=2, height_ratios=[1],
                                width_ratios=[1, 0.6], 
                                hspace=0, wspace=0.25,
                                left=0.0, right=1.0, 
                                bottom=0.0, top=1.0)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])



########
# Fxns #
########

def read_results(fileName):
    """
    Read eigenvalues from JSON file.
    """

    with open(fileName, "r") as f:
        raw_ = json.load(f)

    allnodes = list(raw_.keys())
    alltime = []
    alletime = []
    allftime = []

    for key in list(raw_.keys()):
        alltime.append(raw_[key]["wbse_time"])
        alletime.append(raw_[key]["e_time"])
        allftime.append(raw_[key]["f_time"])

    return allnodes, alltime, alletime, allftime

########
# LOGO #
########

ax0.text(s='WEST', x=0.72, y=0.75, c='k', ha='center',
         transform=ax0.transAxes, fontweight='bold', fontsize=12)
ax0.text(s='TDDFT', x=0.72, y=0.65, c='k', ha='center',
         transform=ax0.transAxes, fontweight='bold', fontsize=9)

########
# Main #
########

fname = ['gpu_999_new.json']
allnodes = []
alltime = []
alletime = []
allftime = []
for f in fname:
    nodes, time, etime, ftime = read_results(f)
    allnodes.append(nodes)
    alltime.append(time)
    alletime.append(etime)
    allftime.append(ftime)

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
linestyles = ['-', '-', '-', ':', '-', ':']
markers = ['s', 'v', 'o']

# nv total wall time
for i in range(len(fname)):
    ax0.plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
            alltime[i], c='gray', marker=markers[i],
            linestyle=linestyles[i], lw=0.6, markersize=3)

# ideal scaling
for i in range(len(fname)):
    ax0.plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
             alltime[i][0] / (np.array(allnodes[i], dtype=np.float64) / 4), c='k',
             linestyle='--', lw=0.3, markersize=0,)

for i in range(1):
    for j in range(1):
        #ax.set_xscale("log")
        ax0.set_yscale("log")

        ax0.set_xlabel('# of GPU nodes')
        if j==0: ax0.set_ylabel('Wall time (s)')

        ax0.spines['left'].set_color('black')
        ax0.spines['right'].set_color('black')
        ax0.tick_params(axis='y', colors='black')
        ax0.tick_params(axis='both', direction='in')
        ax0.tick_params(which='minor', direction='in')
        ax0.xaxis.set_ticks_position('both')
        ax0.yaxis.set_ticks_position('both')

        ax0.set_xlim([0.5, 9.5])
        ax0.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax0.set_xticklabels(['2', '', '8', '', '32', '', '128', '', '512'])
        ax0.set_ylim([100, 100000])

######
# NV #
######

cwd = os.getcwd()
im1 = plt.imread(get_sample_data(path + '/NV-1000.tif'))
newax = fig.add_axes([-0.1, 0.02, 0.35, 0.35], anchor='NE')
ax0.text(s='NV$^-$ in diamond\n999 atoms', x=0.25, y=0.4,
         transform=ax0.transAxes, ha='center', fontsize=7, c='gray')
newax.imshow(im1)
newax.axis('off')

######
# CC #
######

# Functions #
def QUA(x,a,b,c):
    return a * (x - b)**2 + c

def QUA_1(x):
    return QUA(x, 1, 0, 0)

def QUA_2(x):
    return QUA(x, 1, 1, 9)

def QUA_3(x):
    return QUA(x, 1.2, 0.2, 6.8)

def QUA_4(x):
    return QUA(x, 0.8, 0.8, 2.2)

def BACK_L_1(x):
    return -np.sqrt(x)

def BACK_R_1(x):
    return np.sqrt(x)

def BACK_L_2(x):
    return (1.4 - np.sqrt(1.4**2 - 2.8*(7.5-x))) / (1.4)

def BACK_R_2(x):
    return (1.4 + np.sqrt(1.4**2 - 2.8*(7.5-x))) / (1.4)

# Parameters #

# range of the main plot
xlimit = [-2,3.5]
ylimit = [-1,12]

# Paraballas
XAXIS_1 = np.linspace(-1.4,1.4,101,endpoint = True)
XAXIS_2 = np.linspace(-0.4,2.4,101,endpoint = True)
XAXIS_3 = np.linspace(-1.2,1.6,101,endpoint = True)
XAXIS_4 = np.linspace(-0.7,2.3,101,endpoint = True)
PRA_1 = np.zeros(101)
PRA_2 = np.zeros(101)
PRA_3 = np.zeros(101)
PRA_4 = np.zeros(101)
for i in range(101):
    PRA_1[i] = QUA_1(XAXIS_1[i])
    PRA_2[i] = QUA_2(XAXIS_2[i])
    PRA_3[i] = QUA_3(XAXIS_3[i])
    PRA_4[i] = QUA_4(XAXIS_4[i])

# Plot #

# Hide the right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# Hide ticks
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.tick_params( axis='both', which='both', bottom=False,
    top=False, left=False, right=False, labelbottom=False)

ax1.annotate(text='', xy=(xlimit[0]+0.1, ylimit[0]+0.15), xytext=(2.95, ylimit[0]+0.15),
            arrowprops=dict(arrowstyle='<-', shrinkA=0, shrinkB=0, mutation_scale=15,
            color='black'))
ax1.annotate(text='', xy=(xlimit[0]+0.1, ylimit[0]+0.15), xytext=(xlimit[0]+0.1, ylimit[1]),
            arrowprops=dict(arrowstyle='<-', shrinkA=0, shrinkB=0, mutation_scale=15,
            color='black'))

# Paraballes
# 3A2
ax1.plot(XAXIS_1, PRA_1, linewidth=1., linestyle='-', color='#4285F4')
ax1.text(x=-1.8, y=2.3, s='$^3A_2$', fontsize=8, weight='bold', color='#4285F4')
# 3E
ax1.plot(XAXIS_2, PRA_2, linewidth=1., linestyle='-', color='#0F9D58')
ax1.text(x=-1.1, y=11.1, s='$^3E$', fontsize=8, weight='bold', color='#0F9D58')
# 1A1
ax1.plot(XAXIS_3, PRA_3, linewidth=1., linestyle='-', color='#F4B400')
ax1.text(x=-1.8, y=9.5, s='$^1A_1$', fontsize=8, weight='bold', color='#F4B400')
# 1E
ax1.plot(XAXIS_4, PRA_4, linewidth=1., linestyle='-', color='#DB4437')
ax1.text(x=2., y=4.3, s='$^1E$', fontsize=8, weight='bold', color='#DB4437')


# Vertical lines for minimum
ax1.axvline(x=0, ymin=0.13/(ylimit[1] - ylimit[0]), ymax=1,
           linewidth=0.6, linestyle='--', dashes=(5, 5), color='#4285F4')
ax1.axvline(x=1, ymin=0.13/(ylimit[1] - ylimit[0]), ymax=1,
           linewidth=0.6, linestyle='--', dashes=(5, 5), color='#0F9D58')
ax1.axvline(x=0.2, ymin=0.13/(ylimit[1] - ylimit[0]), ymax=1,
           linewidth=0.6, linestyle='--', dashes=(5, 5), color='#F4B400')
ax1.axvline(x=0.8, ymin=0.13/(ylimit[1] - ylimit[0]), ymax=1,
           linewidth=0.6, linestyle='--', dashes=(5, 5), color='#DB4437')

# 3A2
ax1.plot(0, 0, marker='o', markersize=3, markerfacecolor='white',
        markeredgecolor='#4285F4', markeredgewidth=0.5)
# 3E
ax1.plot(1, 9, marker='o', markersize=3, markerfacecolor='white',
        markeredgecolor='#0F9D58', markeredgewidth=0.5)
# 1A1
ax1.plot(0.2, 6.8, marker='o', markersize=3, markerfacecolor='white',
        markeredgecolor='#F4B400', markeredgewidth=0.5)
# 1E
ax1.plot(0.8, 2.2, marker='o', markersize=3, markerfacecolor='white',
        markeredgecolor='#DB4437', markeredgewidth=0.5)

ax1.set_ylim((ylimit))
ax1.set_xlim((xlimit))
ax1.set_xlabel('CC')
ax1.xaxis.set_label_coords(0.45, -0.03)
ax1.set_ylabel('Energy', fontsize=8)
ax1.yaxis.set_label_coords(-0.03, 0.5)
ax1.tick_params(direction='in')

plt.savefig('TOC.tif',dpi=300,bbox_inches='tight')
plt.show()
