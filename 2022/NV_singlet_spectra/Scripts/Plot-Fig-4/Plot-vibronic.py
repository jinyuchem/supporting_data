#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})
from scipy.optimize import curve_fit
from scipy import constants

E_ene = np.loadtxt('1E-levels.dat', usecols=1)
E_text = np.genfromtxt('1E-levels.dat', usecols=2, dtype='str')
E_ene = E_ene - min(E_ene)

A_ene = np.loadtxt('1A1-levels.dat', usecols=1)
A_text = np.genfromtxt('1A1-levels.dat', usecols=2, dtype='str')
A_ene = A_ene - min(A_ene)

########
# Plot #
########

fig, ax = plt.subplots(1, 2, figsize=(4,6))

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
labels = ['$^3A_2$', '$^1E$', '$^1E^{\prime}$', '$^1A_1$']
markers = ['s', 'o', '^', 'v']

######
# 1E #

for i in range(E_ene.shape[0]-11):
    ax[0].hlines(y=E_ene[i], xmin=0.35, xmax=0.6, linewidth=0.5, color='black')
    if i == 0:
        ax[0].text(y=E_ene[0], x=0.26, s='$^1\widetilde{E}$', va='center', ha='center', fontsize=12)
    elif i == 1:
        ax[0].text(y=E_ene[i], x=0.28, s=E_text[i], va='center', ha='center', fontsize=12)
    elif i == 2:
        ax[0].text(y=E_ene[i] - 3, x=0.28, s=E_text[i], va='center', ha='center', fontsize=12)
    elif i == 3:
        ax[0].text(y=E_ene[i] + 7, x=0.68, s=E_text[i], va='center', ha='center', fontsize=12)
    elif i > 0 and i % 2 == 1:
        ax[0].text(y=E_ene[i], x=0.68, s=E_text[i], va='center', ha='center', fontsize=12)
    elif i > 0 and i % 2 == 0:
        ax[0].text(y=E_ene[i], x=0.28, s=E_text[i], va='center', ha='center', fontsize=12)

ax[0].set_xlim(0.1,1)
ax[0].set_ylim(-10,150)
ax[0].set_xticks([])
ax[0].tick_params(axis='both', direction='in')
ax[0].tick_params(which='minor', direction='in')
ax[0].xaxis.set_ticks_position('both')

ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)

ax[0].set_xticklabels([])
ax[0].set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
ax[0].set_yticklabels(['0', '', '', '', '40', '', '', '', '80', '', '', '', '120'])
ax[0].arrow(0.1, y=0, dy=120, dx=0,
            color='black', width=0.0005, length_includes_head=True,
            head_width=0., head_length=0, overhang=1)
ax[0].set_ylabel('Energy (meV)')

#######
# 1A1 #

for i in range(A_ene.shape[0]):
    ax[1].hlines(y=A_ene[i], xmin=0.2, xmax=0.5, linewidth=0.5, color='black')
ax[1].text(y=A_ene[0], x=0.74, s='$^1\widetilde{A}_1$', va='center', ha='center', fontsize=12)
ax[1].text(y=A_ene[1], x=0.74, s=A_text[1], va='center', ha='center', fontsize=12)
ax[1].text(y=A_ene[3], x=0.74, s=A_text[2] + '+' + A_text[3], va='center', ha='center', fontsize=12)
ax[1].text(y=A_ene[6], x=0.74, s=A_text[4] + '+' + A_text[5] + '+' + A_text[6], va='center', ha='center', fontsize=12)
ax[1].text(y=A_ene[9], x=0.74, s=A_text[7] + '+' + A_text[8] + '+' + A_text[9], va='center', ha='center', fontsize=12)

ax[1].set_xlim(0,1)
ax[1].set_ylim(-300,360)
ax[1].set_xticks([])
ax[1].tick_params(axis='both', direction='in')
ax[1].tick_params(which='minor', direction='in')
ax[1].xaxis.set_ticks_position('both')
ax[1].yaxis.set_ticks_position('right')

ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)

# Hide ticks
ax[1].set_xticklabels([])
ax[1].set_yticks([0, 40, 80, 120, 160, 200, 240, 280, 320])
ax[1].set_yticklabels(['0', '', '', '', '160', '', '', '', '320'])
ax[1].arrow(1, y=0, dy=320, dx=0,
            color='black', width=0.0005, length_includes_head=True,
            head_width=0., head_length=0, overhang=1)
ax[1].set_ylabel('Energy (meV)')
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.set_label_coords(1.25,0.7)






ax[1].annotate("",
               xy=(0.2, 0), xycoords='data',
               xytext=(-0.28, -260), textcoords='data',
               arrowprops=dict(arrowstyle="<-", color=colors[1],
                               shrinkA=0, shrinkB=0,
                               patchA=None, patchB=None, lw=0.5,
                               ),
               annotation_clip=False
               )


ax[1].annotate("",
               xy=(0.2, 0), xycoords='data',
               xytext=(-0.28, -220), textcoords='data',
               arrowprops=dict(arrowstyle="<-", color=colors[3],
                               shrinkA=0, shrinkB=0, ls='-',
                               patchA=None, patchB=None, lw=0.5,
                               ),
               annotation_clip=False
               )

ax[1].annotate("",
               xy=(0.2, 0), xycoords='data',
               xytext=(-0.28, -58), textcoords='data',
               arrowprops=dict(arrowstyle="<-", color=colors[1],
                               shrinkA=0, shrinkB=0,
                               patchA=None, patchB=None, lw=0.5,
                               ),
               annotation_clip=False
               )




fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=-0.15, hspace=0.05)

plt.savefig("vibronic-levels.png",bbox_inches = 'tight',dpi=600)
plt.show()
