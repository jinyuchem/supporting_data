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

fig = plt.figure(figsize=(12, 6))

gs = GridSpec(nrows=2, ncols=4, height_ratios=[1, 1],
                                width_ratios=[1.65, 1, 0.0, 2], 
                                hspace=0.25, wspace=0.3,
                                left=0.05, right=0.9, 
                                bottom=0.02, top=0.98)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])

ax = fig.add_subplot(gs[:,3])

######
# NV #
######

ax1.axis('off')
ax1.set_xlim((0,1))
ax1.set_ylim((0,1))
cwd = os.getcwd()
im1 = plt.imread(get_sample_data(path + '/NV-.png'))
newax = fig.add_axes([-0.08, 0.58, 0.32, 0.32], anchor='NE')
newax.imshow(im1)
newax.axis('off')
ax1.text(x=-0.29, y=1.0, s='(a)', fontsize=15)

####################
# NV defect levels #
####################

# DDH
hhPS = np.array([12.7730, 18.3675])
hhup = np.array([13.4049, 14.5855, 14.5855])
hhdown = np.array([14.3934, 17.5945, 17.5945])
hhup = hhup - hhPS[0]
hhdown = hhdown - hhPS[0]

my_ylim = np.array([-0.7, 6.4])

ax2.fill_between(np.linspace(0,1,10), [my_ylim[0] for i in range(10)], [hhPS[0]-hhPS[0] for i in range(10)], color='dodgerblue', alpha=0.3)
ax2.fill_between(np.linspace(0,1,10), [my_ylim[1] for i in range(10)], [hhPS[1]-hhPS[0] for i in range(10)], color='grey', alpha=0.3)
ax2.axvline(x=0.5, ymin=0, ymax=1, color='black', linestyle='--', linewidth=0.5)

ax2.text(x=0.87, y=5.8, s='CB', ha='center')
ax2.text(x=0.87, y=-0.5, s='VB', ha='center')

ax2.axhline(y=hhup[0], xmin=0.19, xmax=0.31, color='black', linestyle='-', linewidth=1.5)
ax2.text(x=0.1, y=hhup[0]+0.45, s='$a_1$', va='center', color='red') 
ax2.axhline(y=hhup[1], xmin=0.31, xmax=0.43, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[2], xmin=0.07, xmax=0.19, color='black', linestyle='-', linewidth=1.5)
ax2.text(x=0.05, y=hhup[1]+0.55, s='$e_x$', va='center', color='red')
ax2.text(x=0.3, y=hhup[1]+0.55, s='$e_y$', va='center', color='red')

ax2.axhline(y=hhdown[0], xmin=0.69, xmax=0.81, color='black', linestyle='-', linewidth=1.5)
ax2.text(x=0.6, y=hhdown[0]+0.35, s='$\overline{a}_1$', va='center', color='red')
ax2.axhline(y=hhdown[1], xmin=0.81, xmax=0.93, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhdown[2], xmin=0.57, xmax=0.69, color='black', linestyle='-', linewidth=1.5)
ax2.text(x=0.55, y=hhdown[1]+0.35, s='$\overline{e}_x$', va='center', color='red')
ax2.text(x=0.8, y=hhdown[1]+0.35, s='$\overline{e}_y$', va='center', color='red')

ax2.arrow(0.25, hhup[0]-0.23, 0., 0.4, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.13, hhup[1]-0.23, 0., 0.4, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.37, hhup[2]-0.23, 0., 0.4, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.75, hhdown[0]+0.23, 0., -0.4, head_width=0.02, head_length=0.1, color='red')

ax2.set_xlim([0,1])
ax2.set_ylim(my_ylim)
ax2.tick_params(direction='in')
ax2.yaxis.set_ticks_position('both')
ax2.set_xticks([0.5])
ax2.set_xticklabels(['NV$^-$ in diamond'], fontsize=13)

ax2.set_ylabel("Energy (eV)", labelpad=10, fontsize=12)

ax2.text(x=-0.4, y=1.0 * 7.1 - 0.7, s='(b)', fontsize=15)

######
# VV #
######

ax3.axis('off')
ax3.set_xlim((0,1))
ax3.set_ylim((0,1))
cwd = os.getcwd()
im3 = plt.imread(get_sample_data(path + '/kk-VV.png'))
newax = fig.add_axes([-0.19, -0.01, 0.44, 0.44], anchor='NE')
newax.imshow(im3)
newax.axis('off')
ax3.text(x=-0.29, y=1, s='(c)', fontsize=15)

####################
# VV defect levels #
####################

kkPS = np.array([8.7818, 12.0318])
kkup = np.array([8.3116, 9.1354, 9.1356, 11.8446, 11.8446])
kkdown = np.array([9.3088, 11.2411, 11.2411, 11.9109, 11.9109])
kkup = kkup - kkPS[0]
kkdown = kkdown - kkPS[0]

my_ylim = np.array([-1, 4.])

ax4.fill_between(np.linspace(0,1,10), [my_ylim[0] for i in range(10)], [kkPS[0]-kkPS[0] for i in range(10)], color='dodgerblue', alpha=0.3)
ax4.fill_between(np.linspace(0,1,10), [my_ylim[1] for i in range(10)], [kkPS[1]-kkPS[0] for i in range(10)], color='grey', alpha=0.3)
ax4.axvline(x=0.5, ymin=0, ymax=1, color='black', linestyle='--', linewidth=0.5)

ax4.text(x=0.87, y=3.45, s='CB', ha='center')
ax4.text(x=0.87, y=-0.6, s='VB', ha='center')

ax4.axhline(y=kkup[0], xmin=0.19, xmax=0.31, color='black', linestyle='-', linewidth=1.5)
ax4.text(x=0.05, y=kkup[0]+0.2, s='$a_1$', va='center', color='red')
ax4.axhline(y=kkup[1], xmin=0.31, xmax=0.43, color='black', linestyle='-', linewidth=1.5)
ax4.axhline(y=kkup[2], xmin=0.07, xmax=0.19, color='black', linestyle='-', linewidth=1.5)
ax4.text(x=0.05, y=kkup[1]+0.4, s='$e_x$', va='center', color='red')
ax4.text(x=0.35, y=kkup[1]+0.4, s='$e_y$', va='center', color='red')
ax4.axhline(y=kkup[3], xmin=0.31, xmax=0.43, color='black', linestyle='-', linewidth=1.5)
ax4.axhline(y=kkup[4], xmin=0.07, xmax=0.19, color='black', linestyle='-', linewidth=1.5)

ax4.axhline(y=kkdown[0], xmin=0.69, xmax=0.81, color='black', linestyle='-', linewidth=1.5)
ax4.text(x=0.55, y=kkdown[0]+0.2, s='$\overline{a}_1$', va='center', color='red')
ax4.axhline(y=kkdown[1], xmin=0.81, xmax=0.93, color='black', linestyle='-', linewidth=1.5)
ax4.axhline(y=kkdown[2], xmin=0.57, xmax=0.69, color='black', linestyle='-', linewidth=1.5)
ax4.text(x=0.55, y=kkdown[1]+0.25, s='$\overline{e}_x$', va='center', color='red')
ax4.axhline(y=kkdown[3], xmin=0.81, xmax=0.93, color='black', linestyle='-', linewidth=1.5)
ax4.text(x=0.85, y=kkdown[1]+0.25, s='$\overline{e}_y$', va='center', color='red')
ax4.axhline(y=kkdown[4], xmin=0.57, xmax=0.69, color='black', linestyle='-', linewidth=1.5)

ax4.arrow(0.25, kkup[0]-0.13, 0., 0.2, head_width=0.02, head_length=0.1, color='red')
ax4.arrow(0.13, kkup[1]-0.13, 0., 0.2, head_width=0.02, head_length=0.1, color='red')
ax4.arrow(0.37, kkup[2]-0.13, 0., 0.2, head_width=0.02, head_length=0.1, color='red')
ax4.arrow(0.75, kkdown[0]+0.13, 0., -0.2, head_width=0.02, head_length=0.1, color='red')

ax4.set_xlim([0,1])
ax4.set_ylim(my_ylim)
ax4.set_xticks([0.5])
ax4.set_xticklabels(['$kk$-VV$^0$ in 4H-SiC'], fontsize=13)
ax4.tick_params(direction='in')
ax4.yaxis.set_ticks_position('both')

ax4.set_ylabel("Energy (eV)", labelpad=0, fontsize=12)
ax4.text(x=-0.4, y=1.0 * 5 - 1, s='(d)', fontsize=15)

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
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Hide ticks
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params( axis='both', which='both', bottom=False,
    top=False, left=False, right=False, labelbottom=False)

# Axes
ax.arrow(x=xlimit[0]+0.1, y=ylimit[0]+0.15, dx=3.-xlimit[0]-0.15, dy=0,
          color='black', width=0.01, length_includes_head=True,
          head_width=0.2, head_length=0.25, overhang=0.5)
ax.arrow(x=xlimit[0]+0.1, y=ylimit[0]+0.15, dy=ylimit[1]-ylimit[0]-0.15, dx=0,
          color='black', width=0.01, length_includes_head=True,
          head_width=0.15, head_length=0.35, overhang=0.5)

# Paraballes
# 3A2
ax.plot(XAXIS_1, PRA_1, linewidth=3., linestyle='-', color='#4285F4')
ax.text(x=-1.8, y=2.3, s='$^3A_2$', fontsize=15, weight='bold', color='#4285F4')
# 3E
ax.plot(XAXIS_2, PRA_2, linewidth=3., linestyle='-', color='#0F9D58')
ax.text(x=-0.9, y=11.1, s='$^3E$', fontsize=15, weight='bold', color='#0F9D58')
# 1A1
ax.plot(XAXIS_3, PRA_3, linewidth=3., linestyle='-', color='#F4B400')
ax.text(x=-1.3, y=9.5, s='$^1A_1$', fontsize=15, weight='bold', color='#F4B400')
# 1E
ax.plot(XAXIS_4, PRA_4, linewidth=3., linestyle='-', color='#DB4437')
ax.text(x=2.3, y=4.3, s='$^1E$', fontsize=15, weight='bold', color='#DB4437')


# Vertical lines for minimum
ax.axvline(x=0, ymin=0.13/(ylimit[1] - ylimit[0]), ymax=1,
           linewidth=0.8, linestyle='--', color='#4285F4')
ax.axvline(x=1, ymin=0.13/(ylimit[1] - ylimit[0]), ymax=1,
           linewidth=0.8, linestyle='--', dashes=(5, 5), color='#0F9D58')
ax.axvline(x=0.2, ymin=0.13/(ylimit[1] - ylimit[0]), ymax=1,
           linewidth=0.8, linestyle='--', dashes=(5, 5), color='#F4B400')
ax.axvline(x=0.8, ymin=0.13/(ylimit[1] - ylimit[0]), ymax=1,
           linewidth=0.8, linestyle='--', dashes=(5, 5), color='#DB4437')


# Names of PECs
ax.text(x=-0.55, y=-0.7, s='$Q_{^3A_2}$', fontsize=12, color='#4285F4', weight='bold')
ax.text(x=1.05, y=-0.7, s='$Q_{^3E}$', fontsize=12, color='#0F9D58', weight='bold')
ax.text(x=0.1, y=-1.2, s='$Q_{^1A_1}$', fontsize=12, color='#F4B400', weight='bold')
ax.text(x=0.7, y=-1.2, s='$Q_{^1E}$', fontsize=12, color='#DB4437', weight='bold')


# circles on PECs
# 3A2
ax.plot(0, 0, marker='o', markersize=8, markerfacecolor='white',
        markeredgecolor='#4285F4', markeredgewidth=1.5)
ax.plot(1, 1, marker='o', markersize=8, markerfacecolor='white',
        markeredgecolor='#0F9D58', markeredgewidth=1.5)
ax.plot(0.2, 0.04, marker='o', markersize=8, markerfacecolor='white',
        markeredgecolor='#F4B400', markeredgewidth=1.5)
ax.plot(0.8, 0.64, marker='o', markersize=8, markerfacecolor='white',
        markeredgecolor='#DB4437', markeredgewidth=1.5)

# 3E
ax.plot(0, 10, marker='o', markersize=8, markerfacecolor='white',
        markeredgecolor='#0F9D58', markeredgewidth=1.5)
ax.plot(1, 9, marker='o', markersize=8, markerfacecolor='white',
        markeredgecolor='#0F9D58', markeredgewidth=1.5)
# 1A1
ax.plot(0, 6.848, marker='o', markersize=8, markerfacecolor='white',
        markeredgecolor='#F4B400', markeredgewidth=1.5)
ax.plot(0.2, 6.8, marker='o', markersize=8, markerfacecolor='white',
        markeredgecolor='#F4B400', markeredgewidth=1.5)
# 1E
ax.plot(0, 2.712, marker='o', markersize=8, markerfacecolor='white',
        markeredgecolor='#DB4437', markeredgewidth=1.5)
ax.plot(0.8, 2.2, marker='o', markersize=8, markerfacecolor='white',
        markeredgecolor='#DB4437', markeredgewidth=1.5)


# Franck-Condon shift
# 3E ES
ax.hlines(y=10, xmin=-0.8, xmax=2.4, linestyle='--', linewidth=1, color='k')
ax.hlines(y=9, xmin=1, xmax=2.4, linestyle='--', linewidth=1, color='k')
ax.annotate(text='', xy=(2.2,10), xytext=(2.2,9),
            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, mutation_scale=15,
            color='black'))
ax.text(x=2.9, y=9.5, s='$E_{\mathrm{FC, ES}, ^3E}$', ha='center', va='center')
# 3E GS
ax.hlines(y=0, xmin=-0.8, xmax=2.4, linestyle='--', linewidth=1, color='k')
ax.hlines(y=1, xmin=1, xmax=2.4, linestyle='--', linewidth=1, color='k')
ax.annotate(text='', xy=(2.2,1), xytext=(2.2,0),
            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, mutation_scale=15,
            color='black'))
ax.text(x=2.9, y=0.5, s='$E_{\mathrm{FC, GS}, ^3E}$', ha='center', va='center')

# ZPL
ax.annotate(text='', xy=(1.6,9), xytext=(1.6,0),
            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, mutation_scale=15,
            color='black'))
#ax.text(x=2.2, y=5.5, s='$E_{\mathrm{ZPL}, ^3E}$', ha='center', va='center')
ax.text(x=2.2, y=5.5, s='$E_{\mathrm{AE}, ^3E}$', ha='center', va='center')

# VEE
ax.annotate(text='', xy=(-0.6,10), xytext=(-0.6,0),
            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, mutation_scale=15,
            color='black'))
ax.text(x=-1.1, y=5.5, s='$E_{\mathrm{VE}, ^3E}$', ha='center', va='center')


ax.set_ylim((ylimit))
ax.set_xlim((xlimit))
ax.set_xlabel('Configuration Coordinate')
ax.xaxis.set_label_coords(0.5, -0.03)
ax.set_ylabel('Energy', fontsize=15)
ax.tick_params(direction='in')

ax.text(x=-3, y=12, s='(e)', fontsize=15)

plt.savefig('Fig-5.pdf',dpi=300,bbox_inches='tight' )
plt.show()
