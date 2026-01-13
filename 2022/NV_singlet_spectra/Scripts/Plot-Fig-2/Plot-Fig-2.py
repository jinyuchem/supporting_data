#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import os
from matplotlib.cbook import get_sample_data

# energy of 3A2, 1E, 1A1, 3E
VEE = np.array([
[0.0, 0.510, 1.334, 2.076], # TDDFT PBE
[0.0, 0.680, 1.978, 2.375], # TDDFT DDH
[0.0, 0.40, 0.99, 2.32], # GW-BSE
[0.0, 0.463, 1.270, 2.152], # New QDET
[0.0, 0.49, 1.41, 2.02], # CI-CRPA
[0.0, 0.25, 1.60, 2.14], # CASSCF
])

VEE_exp = np.array([0.0, 0.34, 0.43, 1.51, 1.60, 2.18])

########
# Plot #
########

fig = plt.figure(figsize=(7.5, 7.5))

gs = GridSpec(nrows=5, ncols=1, height_ratios=[3.5, 0.1, 1, 1, 1],
                                width_ratios=[1], 
                                hspace=0.1, wspace=0.2,
                                left=0.05, right=0.9, 
                                bottom=0.02, top=0.98)

ax3 = fig.add_subplot(gs[0])
ax40 = fig.add_subplot(gs[2])
ax41 = fig.add_subplot(gs[3])
ax42 = fig.add_subplot(gs[4])

#######
# VEE #
#######

for i in range(VEE.shape[0]):
    ax3.hlines(y=VEE[i,0], xmin=((i+1)*2+0.5), xmax=((i+1)*2+1.5),
               linewidth=2, linestyle='-', color='#4285F4')
    ax3.hlines(y=VEE[i,1], xmin=((i+1)*2+0.5), xmax=((i+1)*2+1.5),
               linewidth=2, linestyle='-', color='#DB4437')
    ax3.hlines(y=VEE[i,2], xmin=((i+1)*2+0.5), xmax=((i+1)*2+1.5),
               linewidth=2, linestyle='-', color='#F4B400')
    ax3.hlines(y=VEE[i,3], xmin=((i+1)*2+0.5), xmax=((i+1)*2+1.5),
               linewidth=2, linestyle='-', color='#0F9D58')

# exp 3A2
ax3.hlines(y=VEE_exp[0], xmin=0.5, xmax=1.5,
           linewidth=2, linestyle='-', color='#4285F4')
# exp 1E
ax3.hlines(y=VEE_exp[1], xmin=0.5, xmax=1.5,
           linewidth=2, linestyle='-', color='#DB4437')
ax3.hlines(y=VEE_exp[2], xmin=0.5, xmax=1.5,
           linewidth=2, linestyle='-', color='#DB4437')
ax3.fill_between(x=np.linspace(0.5,1.5,10), y1=VEE_exp[1], y2=VEE_exp[2],
                color='#DB4437', alpha=0.5)
# exp 1A1
ax3.hlines(y=VEE_exp[3], xmin=0.5, xmax=1.5,
           linewidth=2, linestyle='-', color='#F4B400')
ax3.hlines(y=VEE_exp[4], xmin=0.5, xmax=1.5,
           linewidth=2, linestyle='-', color='#F4B400')
ax3.fill_between(x=np.linspace(0.5,1.5,10), y1=VEE_exp[3], y2=VEE_exp[4],
                color='#F4B400', alpha=0.5)
# exp 3E
ax3.hlines(y=VEE_exp[5], xmin=0.5, xmax=1.5,
           linewidth=2, linestyle='-', color='#0F9D58')

# text
ax3.text(x=1, y=2.62, s='Expt.', ha='center', va='center', fontsize=13)
ax3.text(x=3, y=2.62, s='TDDFT\n(PBE)', ha='center', va='center', fontsize=13)
ax3.text(x=5, y=2.62, s='TDDFT\n(DDH)', ha='center', va='center', fontsize=13)
ax3.text(x=7, y=2.62, s='GW-BSE', ha='center', va='center', fontsize=13)
ax3.text(x=9, y=2.62, s='QDET', ha='center', va='center', fontsize=13)
ax3.text(x=11, y=2.62, s='CI-CRPA', ha='center', va='center', fontsize=13)
ax3.text(x=13, y=2.62, s='CASSCF', ha='center', va='center', fontsize=13)


ax3.text(x=-0.2, y=0.0, s='$^3A_2$', ha='center', va='center',
        fontsize=13, color='#4285F4')
ax3.text(x=-0.2, y=(VEE_exp[1]+VEE_exp[2])/2, s='$^1E$', ha='center', va='center',
        fontsize=13, color='#DB4437')
ax3.text(x=-0.2, y=(VEE_exp[3]+VEE_exp[4])/2, s='$^1A_1$', ha='center', va='center',
        fontsize=13, color='#F4B400')
ax3.text(x=-0.2, y=VEE_exp[5], s='$^3E$', ha='center', va='center',
        fontsize=13, color='#0F9D58')

# box
# Create a Rectangle patch
rect = patches.Rectangle((2.2, -0.07), 3.6, 2.5, linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
# Add the patch to the Axes
ax3.add_patch(rect)

ax3.set_xlim(-1,14)
ax3.set_ylim(-0.2, 2.8)
ax3.tick_params(axis='both', direction='in')
ax3.set_ylabel('Vertical Excitation (eV)')
ax3.yaxis.set_ticks_position('both')
ax3.set_xticks([])
ax3.text(x=-2.65, y=1.0 * 3 - 0.2, s='a', weight='bold', fontsize=15)

#########
# Wfxns #
#########

# exex, exey, eyex, eyey, a1ex, exa1, a1ey, eya1, a1a1
TDDFT_PBE = np.array([
[-0.0183, 0.6936, 0.6936, 0.0183, 0, 0, 0, 0, 0],
[-0.67122662,  0, 0., 0.67122662, 0.19884326, 0, -0.10587132, 0, 0],
[0, -0.67122662, -0.67122662,  0, -0.10587132, 0, -0.19884326, 0, 0],
[-0.7069, -0.0186, -0.0186, 0.7069, 0, 0, 0, 0, 0]
])

TDDFT_DDH = np.array([
[0.0131, 0.6803, 0.6811, -0.0130, 0, 0, 0, 0, 0],
[0.6475, -0.0084, -0.0084, -0.6476, -0.2424, 0, -0.0010, 0, 0],
[0.0084, 0.6480, 0.6472, -0.0084, -0.0010, 0, 0.2423, 0, 0],
[0.7054, -0.0135, -0.0135, -0.7053, 0, 0, 0, 0, 0]
])


QDET_PBE = np.array([
[0, -0.704, 0.704, 0, 0, 0, 0, 0, 0],
[-0.67044537, 0, 0, 0.67044537, 0.18079922, 0.18079922, 0.0568443, 0.0568443, 0],
[0, -0.67044537, -0.67044537, 0, -0.05643264, -0.05643264, 0.18171055, 0.18171055, 0],
[0.685, 0, 0, 0.685, 0, 0, 0, 0, -0.162],
[0, 0, 0, 0, 0.478, -0.478, -0.458, 0.458, 0],
[0, 0, 0, 0, 0.458, -0.458, 0.478, -0.478, 0]
])

labels_1 = ['TDDFT (PBE)', 'TDDFT (DDH)', 'QDET$^b$']

s = ['$^1A_1$', '$^1E_y$', '$^1E_x$', '$^3A_2$']

colors = ['#4285F4',
          '#DB4437',
          '#F4B400']

labels = [
'$|e_x\overline{e}_x\\rangle$',
'$|e_x\overline{e}_y\\rangle$',
'$|e_y\overline{e}_x\\rangle$',
'$|e_y\overline{e}_y\\rangle$',
'$|a_1\overline{e}_x\\rangle$',
'$|e_x\overline{a}_1\\rangle$',
'$|a_1\overline{e}_y\\rangle$',
'$|e_y\overline{a}_1\\rangle$',
'$|a_1\overline{a}_1\\rangle$',
]

patterns = [None, "x"*2, "-"*2,
            None, "x"*2, "-"*2,
            None, "x"*2, "-"*2]

width = 0.2
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

axs = [ax40, ax41, ax42]

for i in range(3):
    axs[i].bar(x - 1*width, TDDFT_PBE[3-i,:]**2, width, color=colors[0],
                   edgecolor='black', hatch=patterns[0], label='TDDFT (PBE)')
    axs[i].bar(x - 0*width, TDDFT_DDH[3-i,:]**2, width, color=colors[1],
                   edgecolor='black', hatch=patterns[0], label='TDDFT (DDH)')
    axs[i].bar(x + 1*width, QDET_PBE[3-i,:]**2, width, color=colors[2],
                   edgecolor='black', hatch=patterns[0], label='QDET')

    if i==2:
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(labels, rotation=0, fontsize=13)
    else:
        axs[i].set_xticks(x)
        axs[i].set_xticklabels([])
    axs[i].set_xlim([-0.3, 9.7])
    axs[i].set_ylim([0, 0.58])

    axs[i].set_yticks([0, 0.2, 0.4])

    axs[i].yaxis.set_ticks_position('both')
    axs[i].xaxis.set_ticks_position('both')

    if i==0:
        axs[i].legend(loc='upper right',
                      ncol=1, edgecolor='black', fontsize=13, framealpha=1,
                      labelspacing=0.2, handlelength=1.0, handleheight=0.5,
                      handletextpad=0.5, columnspacing=1.0, borderaxespad=0.3)

    axs[i].tick_params(axis='both', direction='in')
    axs[i].yaxis.set_ticks_position('both')

    axs[i].text(x=-0.16, y=0.45, s=s[i])

axs[0].text(x=-1.65/15 * 10 - 0.3, y=1.0 * 0.58, s='b', weight='bold', fontsize=15)
axs[1].set_ylabel('$|c|^2$')

plt.savefig('Fig-2.pdf',dpi=300,bbox_inches='tight' )
plt.show()
