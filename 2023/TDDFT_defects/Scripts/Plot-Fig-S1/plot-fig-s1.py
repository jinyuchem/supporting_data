#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})
import json

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
    alliotime = []

    for key in list(raw_.keys()):
        alltime.append(raw_[key]["wbse_time"])
        alletime.append(raw_[key]["e_time"])
        allftime.append(raw_[key]["f_time"])
        alliotime.append(raw_[key]["io_time"])

    return allnodes, alltime, alletime, allftime, alliotime

########
# Main #
########

fname = ['511_cpu.json', '511_gpu.json', '999_gpu.json']
allnodes = []
alltime = []
alletime = []
allftime = []
alliotime = []
for f in fname:
    nodes, time, etime, ftime, iotime = read_results(f)
    allnodes.append(nodes)
    alltime.append(time)
    alletime.append(etime)
    allftime.append(ftime)
    alliotime.append(iotime)

########
# Plot #
########

fig, ax = plt.subplots(1, 3, figsize=(13,4))

colors = ['#DB4437', '#4285F4', '#F4B400', '#0F9D58']
linestyles = ['-', '-', '-', ':', '-', ':']
labels = ['Total', 'Energy', 'Forces', 'I/O']
markers = ['o', 'v', '^', 's']

# nv total wall time
for i in range(len(fname)):
    ax[i].plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
            alltime[i], c=colors[2], marker=markers[0],
            linestyle=linestyles[0], lw=1.5, markersize=7, label=labels[0])
    ax[i].plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
            alletime[i], c=colors[0], marker=markers[1],
            linestyle=linestyles[0], lw=1.5, markersize=7, label=labels[1])
    ax[i].plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
            allftime[i], c=colors[1], marker=markers[2],
            linestyle=linestyles[0], lw=1.5, markersize=7, label=labels[2])

# ideal scaling
for i in range(len(fname)):
    if i==0:
        ax[i].plot(np.log2(np.array(allnodes[i][:], dtype=np.float64)),
                alltime[i][0] / np.array(allnodes[i][:], dtype=np.float64), c='k',
                linestyle='--', lw=1, markersize=0, label='Ideal\nscaling')
        ax[i].plot(np.log2(np.array(allnodes[i][:], dtype=np.float64)),
                alletime[i][0] / np.array(allnodes[i][:], dtype=np.float64), c='k',
                linestyle='--', lw=1, markersize=0,)
        ax[i].plot(np.log2(np.array(allnodes[i][:], dtype=np.float64)),
                allftime[i][0] / (np.array(allnodes[i][:], dtype=np.float64)), c='k',
                linestyle='--', lw=1, markersize=0,)
    elif i==1:
        ax[i].plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
                alltime[i][0] / np.array(allnodes[i], dtype=np.float64), c='k',
                linestyle='--', lw=1, markersize=0, label='Ideal\nscaling')
        ax[i].plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
                alletime[i][0] / np.array(allnodes[i], dtype=np.float64), c='k',
                linestyle='--', lw=1, markersize=0,)
        ax[i].plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
                allftime[i][0] / (np.array(allnodes[i], dtype=np.float64)), c='k',
                linestyle='--', lw=1, markersize=0,)
    elif i==2:
        ax[i].plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
                alltime[i][0] * 4 / np.array(allnodes[i], dtype=np.float64), c='k',
                linestyle='--', lw=1, markersize=0, label='Ideal\nscaling')
        ax[i].plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
                alletime[i][0] * 4 / np.array(allnodes[i], dtype=np.float64), c='k',
                linestyle='--', lw=1, markersize=0,)
        ax[i].plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
                allftime[i][0] * 4 / (np.array(allnodes[i], dtype=np.float64)), c='k',
                linestyle='--', lw=1, markersize=0,)

ax[0].set_title('(a) CPU, $4 \\times 4 \\times 4$')
ax[1].set_title('(b) GPU, $4 \\times 4 \\times 4$')
ax[2].set_title('(c) GPU, $5 \\times 5 \\times 5$')

for i in range(3):
    for j in range(1):
        #ax.set_xscale("log")
        ax[i].set_yscale("log")

        if i==1: ax[i].legend(fontsize=10.5,loc='upper right',
                              edgecolor='black',ncol=1,columnspacing=1)

        if i==0: ax[i].set_xlabel('Number of CPU nodes')
        elif i > 0: ax[i].set_xlabel('Number of GPU nodes')

        if i==0: ax[i].set_ylabel('Wall time (s)')

        ax[i].spines['left'].set_color('black')
        ax[i].spines['right'].set_color('black')
        ax[i].tick_params(axis='y', colors='black')
        ax[i].tick_params(axis='both', direction='in')
        ax[i].tick_params(which='minor', direction='in')
        ax[i].xaxis.set_ticks_position('both')
        ax[i].yaxis.set_ticks_position('both')

        if i==0:
            ax[i].set_xlim([-0.5, 8.5])
            ax[i].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
            ax[i].set_xticklabels(['$2^0$', '$2^1$', '$2^{2}$',
                            '$2^{3}$', '$2^{4}$', '$2^{5}$',
                            '$2^{6}$', '$2^{7}$', '$2^{8}$'],
              rotation=0)
        elif i==1:
            ax[i].set_xlim([-0.5, 7.5])
            ax[i].set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
            ax[i].set_xticklabels(['$2^0$', '$2^1$', '$2^{2}$',
                            '$2^{3}$', '$2^{4}$', '$2^{5}$',
                            '$2^{6}$', '$2^{7}$'],
              rotation=0)
        elif i==2:
            ax[i].set_xlim([1.5, 9.5])
            ax[i].set_xticks([2, 3, 4, 5, 6, 7, 8, 9])
            ax[i].set_xticklabels(['$2^{2}$',
                            '$2^{3}$', '$2^{4}$', '$2^{5}$',
                            '$2^{6}$', '$2^{7}$', '$2^{8}$', '$2^{9}$'],
              rotation=0)

        ax[i].set_ylim([10, 100000])

fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.25, hspace=0.2)

plt.savefig("Fig-S1.pdf", bbox_inches='tight', dpi=300)
plt.show()
