#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.cbook import get_sample_data
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

current_working_directory = os.getcwd()

PBE_DIS = np.array( [ [ 1.472, 1.472, 1.472, 2.460, 2.460, 2.460, 2.671, 2.671, 2.671 ],
                      [ 1.466, 1.477, 1.477, 2.461, 2.459, 2.459, 2.765, 2.634, 2.634 ],
                      [ 1.472, 1.472, 1.472, 2.458, 2.458, 2.458, 2.692, 2.692, 2.692 ],
                      [ 1.492, 1.487, 1.487, 2.474, 2.479, 2.479, 2.707, 2.783, 2.783 ] ] )

DDH_DIS = np.array( [ [ 1.463, 1.463, 1.463, 2.446, 2.446, 2.446, 2.645, 2.645, 2.645 ],
                      [ 1.457, 1.469, 1.469, 2.445, 2.447, 2.447, 2.750, 2.618, 2.618 ],
                      [ 1.463, 1.463, 1.463, 2.445, 2.445, 2.445, 2.654, 2.654, 2.654 ],
                      [ 1.481, 1.480, 1.480, 2.453, 2.459, 2.459, 2.716, 2.774, 2.774 ] ] )

PBE_DDIS = np.zeros((3, 9))
DDH_DDIS = np.zeros((3, 9))
for i in range(3):
    PBE_DDIS[i] = PBE_DIS[i+1] - PBE_DIS[0]
    DDH_DDIS[i] = DDH_DIS[i+1] - DDH_DIS[0]

########
# Plot #
########

fig, axs = plt.subplots(1, 1, figsize=(6, 7))

labels_1 = [ '$^1E$\n(PBE)', '$^1E$\n(DDH)',
             '$^1A_1$\n(PBE)', '$^1A_1$\n(DDH)',
             '$^3E$\n(PBE)', '$^3E$\n(DDH)' ]

width = 0.3

colors = ['#4285F4', '#4285F4', '#4285F4',
          '#DB4437', '#DB4437', '#DB4437',
          '#4285F4', '#DB4437', '#F4B400']

labels = ['N-C1', 'N-C2', 'N-C3',
          'C2-C3', 'C3-C1', 'C1-C2',
          'C$_2-$C$_3$', 'C$_3-$C$_1$', 'C$_1-$C$_2$']

patterns = [None, "x"*2, "-"*2,
            None, "x"*2, "-"*2,
            None, None, None]

####
# Schematic
####

def get_circles_dashed(cc4, cc5, cc6, cvo):
    circle0 = plt.Circle((cc4[0], cc4[1]), 0.03, edgecolor='black', facecolor='#6E2C00',
                         alpha=0.2, linestyle='--', linewidth=1)
    circle1 = plt.Circle((cc5[0], cc5[1]), 0.03, edgecolor='black', facecolor='#6E2C00',
                         alpha=0.2, linestyle='--', linewidth=1)
    circle2 = plt.Circle((cc6[0], cc6[1]), 0.03, edgecolor='black', facecolor='#6E2C00',
                         alpha=0.2, linestyle='--', linewidth=1)
    circle3 = plt.Circle((cvo[0], cvo[1]), 0.04, edgecolor='black', facecolor='white',
                         alpha=0.2, linestyle='--', linewidth=1)
    return [circle0, circle1, circle2, circle3]

def get_circles(cc4, cc5, cc6, cvo):
    circle0 = plt.Circle((cc4[0], cc4[1]), 0.03, edgecolor='black', facecolor='#6E2C00',
                         alpha=0.7, linestyle='-', linewidth=1)
    circle1 = plt.Circle((cc5[0], cc5[1]), 0.03, edgecolor='black', facecolor='#6E2C00',
                         alpha=0.7, linestyle='-', linewidth=1)
    circle2 = plt.Circle((cc6[0], cc6[1]), 0.03, edgecolor='black', facecolor='#6E2C00',
                         alpha=0.7, linestyle='-', linewidth=1)
    circle3 = plt.Circle((cvo[0], cvo[1]), 0.04, edgecolor='black', facecolor='white',
                         alpha=0.7, linestyle='--', linewidth=1)
    return [circle0, circle1, circle2, circle3]

# 3a2
cc40 = np.array([0.5, 0.7])
cc50 = np.array([0.5-0.1*np.sqrt(3), 0.4])
cc60 = np.array([0.5+0.1*np.sqrt(3), 0.4])
cvo0 = np.array([0.5, 0.513])

# 1e
change = np.array([0.03669, -0.01235, -0.01235])
change = change * 5

cc4 = cc40 + [0.7, 0.0] 
cc5 = cc50 + [0.7, 0.0] 
cc6 = cc60 + [0.7, 0.0] 
cvo = cvo0 + [0.7, 0.0]
l_circles_1e00 = get_circles_dashed(cc4, cc5, cc6, cvo)
l_circles_1e01 = get_circles_dashed(cc4 + [0, 0.6],
                            cc5 + [0, 0.6],
                            cc6 + [0, 0.6],
                            cvo + [0, 0.6])
l_circles_1e02 = get_circles_dashed(cc4 + [0, -0.6],
                            cc5 + [0, -0.6],
                            cc6 + [0, -0.6],
                            cvo + [0, -0.6])

t = 0.3 - np.sqrt((0.2 * np.sqrt(3) * (1 + change[1]))**2 - (0.1 * np.sqrt(3) * (1 + change[0]))**2)\
  - change[0]/2 * 0.2
cc4 = cc4 + [0, -t]
cc5 = cc5 + [-change[0]/2 * 0.2 * np.sqrt(3), change[0]/2 * 0.2]
cc6 = cc6 + [change[0]/2 * 0.2 * np.sqrt(3), change[0]/2 * 0.2]
cvo = cvo 
l_circles_1e1 = get_circles(cc4, cc5, cc6, cvo)

########
# Plot #
########

x = np.array([-0.2, 3, 6.2])
y = np.array([1.4, 4.6, 7.8])
for i in range(9):

    if i>=6:
        axs.bar(x+(i-4-3)*width, PBE_DDIS[:,i], width, color=colors[i],
                   label=labels[i], edgecolor='black', hatch=patterns[i])
        axs.bar(y+(i-4-3)*width, DDH_DDIS[:,i], width, color=colors[i],
                   edgecolor='black', hatch=patterns[i])

axs.set_ylabel('$\Delta d = d_{\mathrm{ES}} - d_{\mathrm{GS}}$ (Å)', labelpad=0, fontsize=13)
axs.set_xticks([x[0], y[0], x[1], y[1], x[2], y[2]])
axs.set_xticklabels(labels_1, rotation=0, fontsize=12)
axs.set_xlim([-1., 8.6])
axs.set_ylim([-0.04, 0.24])

axs.set_yticks([-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24])
axs.set_yticklabels(['-0.04','','0','','0.04','','0.08','','0.12','','0.16', '', '0.20', '', '0.24'])
axs.axhline(y=0, xmin=0, xmax=1, linewidth=0.5, color='black')

axs.legend(loc='upper left',
              ncol=3, edgecolor='black', fontsize=12, framealpha=1,
              labelspacing=0.2, handlelength=1.0, handleheight=0.5,
              handletextpad=0.5, columnspacing=1.0, borderaxespad=0.3)

axs.tick_params(axis='both', direction='in')
axs.yaxis.set_ticks_position('both')



axins = inset_axes(axs, width="100%", height="100%",
                   bbox_to_anchor=(1.6, 0.12, 4.5, 0.1),
                   bbox_transform=axs.transData, loc=2, borderpad=0)


for circle in l_circles_1e00:
    axins.add_patch(circle)
for circle in l_circles_1e1:
    axins.add_patch(circle)

axins.set_xlim([0,1])
axins.set_ylim([0,1])

axins.annotate("",
            xy=(cc50[0] + 0.7, cc50[1]), xycoords='data',
            xytext=(cc60[0] + 0.7, cc60[1]), textcoords='data',
            arrowprops=dict(arrowstyle="|-|", color='black',
                            shrinkA=0, shrinkB=0,
                            patchA=None, patchB=None, lw=1,
                            mutation_scale=6, ls='--'
                            ),
            annotation_clip=False
            )
axins.annotate("",
            xy=(cc60[0] + 0.69, cc50[1]), xycoords='data',
            xytext=(cc60[0] + 0.7, cc50[1]), textcoords='data',
            arrowprops=dict(arrowstyle="<-", color='black',
                            shrinkA=0, shrinkB=0,
                            patchA=None, patchB=None, lw=1,
                            mutation_scale=20, ls='--'
                            ),
            annotation_clip=False
            )
axins.annotate("",
            xy=(cc50[0] + 0.70, cc50[1]), xycoords='data',
            xytext=(cc50[0] + 0.71, cc50[1]), textcoords='data',
            arrowprops=dict(arrowstyle="->", color='black',
                            shrinkA=0, shrinkB=0,
                            patchA=None, patchB=None, lw=1,
                            mutation_scale=20, ls='--'
                            ),
            annotation_clip=False
            )





axins.annotate("",
            xy=(cc5[0], cc5[1]), xycoords='data',
            xytext=(cc6[0], cc6[1]), textcoords='data',
            arrowprops=dict(arrowstyle="|-|", color='black',
                            shrinkA=0, shrinkB=0,
                            patchA=None, patchB=None, lw=1,
                            mutation_scale=6, ls='-'
                            ),
            annotation_clip=False
            )
axins.annotate("",
            xy=(cc5[0], cc5[1]), xycoords='data',
            xytext=(cc6[0], cc6[1]), textcoords='data',
            arrowprops=dict(arrowstyle="<->", color='black',
                            shrinkA=0, shrinkB=0,
                            patchA=None, patchB=None, lw=1,
                            mutation_scale=20, ls='-'
                            ),
            annotation_clip=False
            )


xx = (cc5[0] + cc6[0]) / 2
yy = cc5[1] + 0.03

axins.text(x=xx, y=yy, s='$d_{\mathrm{ES}}$(C2-C3)',
        va='center', ha='center')
yy = cc50[1] - 0.03
axins.text(x=xx, y=yy, s='$d_{\mathrm{GS}}$(C2-C3)',
        va='center', ha='center')


axins.text(x=cvo0[0] + 0.7, y=cvo0[1], s='V$_{\mathrm{C}}$',
        va='center', ha='center')
axins.text(x=cc4[0]-0.06, y=cc4[1], s='C$_1$',
        va='center', ha='center')
axins.text(x=cc5[0]-0.06, y=cc5[1], s='C$_2$',
        va='center', ha='center')
axins.text(x=cc6[0]+0.06, y=cc6[1], s='C$_3$',
        va='center', ha='center')


axins.tick_params(axis='both', direction='in')
axins.yaxis.set_ticks_position('both')

# Hide the right and top spines
axins.spines['right'].set_visible(False)
axins.spines['top'].set_visible(False)
axins.spines['left'].set_visible(False)
axins.spines['bottom'].set_visible(False)

axins.set_xticks([])
axins.set_yticks([])

axins.axis('equal')


plt.savefig('Fig-3.pdf',dpi=300,bbox_inches='tight' )
plt.show()
