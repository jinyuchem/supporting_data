#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import os
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.patches import ConnectionPatch
import numpy.polynomial.hermite as Herm
import math
import os

current_working_directory = os.getcwd()

########
# Plot #
########

fig = plt.figure(figsize=(10, 4))

gs = GridSpec(nrows=3, ncols=3, height_ratios=[0.1, 1, 0.1],
                                width_ratios=[0.1, 0.3, 1], 
                                hspace=0.2, wspace=0.2,
                                left=0.05, right=0.9, 
                                bottom=0.02, top=0.98)

ax1 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[:, 2])

#############
# KS levels #
#############

# list the important KS orbitals
DDH_orbitals = dict()
DDH_orbitals["label"] = [
    ["a1pp", "a1p", "a1", "ex", "ey"],
]
DDH_orbitals["index"] = [
    [1009, 1019, 1022, 1023, 1024],
]
DDH_orbitals["energies"] = [
    [12.5172, 13.2243, 14.3056, 15.8398, 15.8399],
]

DDH_hhPS = np.array([13.2452, 17.4679])

hhup = DDH_orbitals["energies"][0] - DDH_hhPS[0]

my_ylim = np.array([-1.5, 5])

ax1.fill_between(np.linspace(-0.1,1.1,10),
                 [my_ylim[0] for i in range(10)], [DDH_hhPS[0]-DDH_hhPS[0] for i in range(10)],
                 color='#AFDCEC')
ax1.fill_between(np.linspace(-0.1,1.1,10),
                 [my_ylim[1] for i in range(10)], [DDH_hhPS[1]-DDH_hhPS[0] for i in range(10)],
                 color='#D3D3D3')

ax1.text(x=0.96, y=4.4, s='CB', ha='center')
ax1.text(x=0.96, y=-0.8, s='VB', ha='center')

# spin up
ax1.hlines(y=hhup[0], xmin=0.42, xmax=0.58, color='black', linestyle='-', linewidth=1.5)
ax1.hlines(y=hhup[1], xmin=0.42, xmax=0.58, color='black', linestyle='-', linewidth=1.5)
ax1.hlines(y=hhup[2], xmin=0.42, xmax=0.58, color='black', linestyle='-', linewidth=1.5)
ax1.hlines(y=hhup[3], xmin=0.18, xmax=0.34, color='black', linestyle='-', linewidth=1.5)
ax1.hlines(y=hhup[4], xmin=0.66, xmax=0.82, color='black', linestyle='-', linewidth=1.5)

ax1.arrow(0.47, hhup[0]-0.2, 0., 0.3, head_width=0.03, head_length=0.1, color='red')
ax1.arrow(0.53, hhup[0]-0.2 + 0.4, 0., -0.3, head_width=0.03, head_length=0.1, color='red')
ax1.arrow(0.47, hhup[1]-0.2, 0., 0.3, head_width=0.03, head_length=0.1, color='red')
ax1.arrow(0.53, hhup[1]-0.2 + 0.4, 0., -0.3, head_width=0.03, head_length=0.1, color='red')
ax1.arrow(0.47, hhup[2]-0.2, 0., 0.3, head_width=0.03, head_length=0.1, color='red')
ax1.arrow(0.53, hhup[2]-0.2 + 0.4, 0., -0.3, head_width=0.03, head_length=0.1, color='red')
ax1.arrow(0.26, hhup[3]-0.2, 0., 0.3, head_width=0.03, head_length=0.1, color='red', linestyle='-')
ax1.arrow(0.74, hhup[4]-0.2, 0., 0.3, head_width=0.03, head_length=0.1, color='red', linestyle='-')

ax1.text(x=0.28, y=hhup[0], s='$a_1^{\prime\prime}$', va='center', color='red', fontsize=12)
ax1.text(x=0.28, y=hhup[1], s='$a_1^{\prime}$', va='center', color='red', fontsize=12)
ax1.text(x=0.28, y=hhup[2], s='$a_1$', va='center', color='red', fontsize=12)
ax1.text(x=0.04, y=hhup[3], s='$e_x$', va='center', color='red', fontsize=12)
ax1.text(x=0.86, y=hhup[4], s='$e_y$', va='center', color='red', fontsize=12)

ax1.set_xlim([-0.05, 1.05])
ax1.set_ylim(my_ylim)
ax1.tick_params(direction='in')
ax1.yaxis.set_ticks_position('both')
ax1.set_xticks([0.5])
ax1.set_xticklabels([])
ax1.set_ylabel("Energy (eV)", labelpad=0, fontsize=13)

ax1.text(x=-0.35, y=1.15, s='(a)', fontsize=14, transform=ax1.transAxes)
ax2.text(x=0.0, y=0.98, s='(b)', fontsize=14, transform=ax2.transAxes)

###############
# KS orbitals #
###############

ax2.axis('off')
ax2.set_xlim((0, 1))
ax2.set_ylim((0, 1))
cwd = os.getcwd()
im1 = plt.imread(get_sample_data(current_working_directory + '/a1pp.jpg'))
newax = fig.add_axes([0.4, -0.1, 0.15, 1], anchor='NE')
newax.imshow(im1)
newax.axis('off')
im2 = plt.imread(get_sample_data(current_working_directory + '/a1p.jpg'))
newax = fig.add_axes([0.6, -0.1, 0.15, 1], anchor='NE')
newax.imshow(im2)
newax.axis('off')
im3 = plt.imread(get_sample_data(current_working_directory + '/a1.jpg'))
newax = fig.add_axes([0.8, -0.1, 0.15, 1], anchor='NE')
newax.imshow(im3)
newax.axis('off')

im4 = plt.imread(get_sample_data(current_working_directory + '/ex.jpg'))
newax = fig.add_axes([0.5, -0.55, 0.15, 1], anchor='NE')
newax.imshow(im4)
newax.axis('off')
im5 = plt.imread(get_sample_data(current_working_directory + '/ey.jpg'))
newax = fig.add_axes([0.7, -0.55, 0.15, 1], anchor='NE')
newax.imshow(im5)
newax.axis('off')

ax2.text(x=0.20, y=0.94, s='$a_1^{\prime\prime}$', fontsize=12, ha='center', transform=ax2.transAxes)
ax2.text(x=0.58, y=0.94, s='$a_1^{\prime}$', fontsize=12, ha='center', transform=ax2.transAxes)
ax2.text(x=0.96, y=0.94, s='$a_1$', fontsize=12, ha='center', transform=ax2.transAxes)

ax2.text(x=0.39, y=0.47, s='$e_x$', fontsize=12, ha='center', transform=ax2.transAxes)
ax2.text(x=0.77, y=0.47, s='$e_y$', fontsize=12, ha='center', transform=ax2.transAxes)

plt.savefig('figure_s2.pdf', dpi=500, bbox_inches='tight')