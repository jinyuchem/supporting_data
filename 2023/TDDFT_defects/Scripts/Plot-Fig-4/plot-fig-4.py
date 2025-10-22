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

    for key in list(raw_.keys()):
        alltime.append(raw_[key]["wbse_time"])
        alletime.append(raw_[key]["e_time"])
        allftime.append(raw_[key]["f_time"])

    return allnodes, alltime, alletime, allftime

########
# Main #
########

fname = ['cpu_511.json', 'gpu_511_new.json', 'gpu_999_new.json']
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

########
# Plot #
########

fig, ax = plt.subplots(1, 1, figsize=(4,4))

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
linestyles = ['-', '-', '-', ':', '-', ':']
labels = ['CPU, $(4 \\times 4 \\times 4)$',
           'GPU, $(4 \\times 4 \\times 4)$',
           'GPU, $(5 \\times 5 \\times 5)$',
]
markers = ['^', 'v', 'o']

# nv total wall time
for i in range(len(fname)):
    ax.plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
            alltime[i], c=colors[i], marker=markers[i],
            linestyle=linestyles[i], lw=1.5, markersize=7, label=labels[i])

# ideal scaling
for i in range(len(fname)):
    if i==0:
        ax.plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
                alltime[i][0] / np.array(allnodes[i], dtype=np.float64), c='k',
                linestyle='--', lw=1, markersize=0, label='Ideal scaling')
    elif i==1:
        ax.plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
                alltime[i][0] / np.array(allnodes[i], dtype=np.float64), c='k',
                linestyle='--', lw=1, markersize=0,)
    elif i==2:
        ax.plot(np.log2(np.array(allnodes[i], dtype=np.float64)),
                alltime[i][0] / (np.array(allnodes[i], dtype=np.float64) / 4), c='k',
                linestyle='--', lw=1, markersize=0,)

for i in range(1):
    for j in range(1):
        #ax.set_xscale("log")
        ax.set_yscale("log")

        if j==0: ax.legend(fontsize=10.5,loc='best',edgecolor='black')

        ax.set_xlabel('Number of nodes')
        if j==0: ax.set_ylabel('Wall time (s)')

        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.tick_params(axis='y', colors='black')
        ax.tick_params(axis='both', direction='in')
        ax.tick_params(which='minor', direction='in')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        ax.set_xlim([-0.5, 9.5])
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_xticklabels(['$2^{0}$', '$2^{1}$', '$2^{2}$',
                            '$2^{3}$', '$2^{4}$', '$2^{5}$',
                            '$2^{6}$', '$2^{7}$', '$2^{8}$', '$2^{9}$'],
              rotation=0)

        ax.set_ylim([100, 100000])

plt.savefig("Fig-4.pdf", bbox_inches='tight', dpi=300)
plt.show()
