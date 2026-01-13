#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 13.5})
from scipy.optimize import curve_fit
from scipy import constants
from scipy.linalg import eigh
from math import nan
from matplotlib.gridspec import GridSpec
from matplotlib.cbook import get_sample_data
import os
import matplotlib.patches as patches

current_working_directory = os.getcwd()

############
# Function #
############

def pes_numerical_solver(x, y, eph, Ft, F):
    # Note: transform x coordinates from amu^{0.5} Å into unit less
    x = x * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
    t_eph = eph * 1e-3 * constants.eV / constants.hbar
    x = x * np.sqrt(t_eph / constants.hbar)

    y = y * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
    t_eph = eph * 1e-3 * constants.eV / constants.hbar
    y = y * np.sqrt(t_eph / constants.hbar)

    Gt = 0
    G = 0

    elow = np.zeros((x.shape[0], y.shape[0]))
    ehigh = np.zeros((x.shape[0], y.shape[0]))
    a = np.zeros((x.shape[0], y.shape[0]))

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            mat = np.array([
                  [
                      Le,
                      Ft * x[i],
                      Ft * y[j]
                  ],
                  [
                      Ft * x[i],
                      F * x[i],
                      - F * y[j]
                  ],
                  [
                      Ft * y[j],
                      - F * y[j],
                      - F * x[i]
                  ]
                  ])

            w, v = eigh(mat)
            elow[i,j] = float(w[0]) + 0.5 * eph * (x[i]**2 + y[j]**2)
            ehigh[i,j] = float(w[1]) + 0.5 * eph * (x[i]**2 + y[j]**2)
            a[i,j] = float(w[2]) + 0.5 * eph * (x[i]**2 + y[j]**2) #- Le

    return elow, ehigh, a

##############
# Parameters #
##############

Le = 821

eph = 62.9506828
Ft = 133.22436286
F = 62.37653058

# here x and y are all unitless
x = np.linspace(0.6, -0.6, 201)
y = np.linspace(-0.6, 0.6, 201)

lb = pes_numerical_solver(x, y, eph, Ft, F)[0]

ref_min = np.min(lb)

ref_shift = abs(ref_min)

#lb = lb - np.min(lb)

for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        if lb[i,j] > 80 + ref_min:
            lb[i,j] = nan


aux = pes_numerical_solver(x, y, eph, Ft, F)



_min = np.min(aux)

aaaa = pes_numerical_solver(x, y, eph, Ft, F)[2]
ref_min = np.min(aaaa)

_max = ref_min + 250



########
# Plot #
########

fig = plt.figure(figsize=(12, 9))
gs = GridSpec(nrows=11, ncols=2, height_ratios=[1, 1, 1, 0.5, 1, 1, 1, 0.5, 1, 1, 1],
                                width_ratios=[1.5, 1],
                                hspace=0.02, wspace=0.5,
                                left=0.05, right=0.9,
                                bottom=0.02, top=0.98)

ax1 = fig.add_subplot(gs[0:3,1])
ax05 = fig.add_subplot(gs[4:7,1])
ax0 = fig.add_subplot(gs[8:,1])

ax10 = fig.add_subplot(gs[0:5,0])

ax30 = fig.add_subplot(gs[5:,0])



##############
# contour 1e #
##############

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
labels = ['$^3A_2$', '$^1E$', '$^1E^{\prime}$', '$^1A_1$']
markers = ['s', 'o', '^', 'v']

im = ax0.contour(x, y, lb, 10, colors='white', linestyles='-', linewidths=0.01)
im = ax0.contourf(x, y, lb, 10, cmap="turbo", vmin=_min, vmax=_max)

divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[-64, -48, -32, -16, 0, 16])
cbar.ax.set_yticklabels(['$-64$', '', '$-32$', '', '$0$', ''])
cbar.set_label('Energy (meV)')

ax0.set_xlabel('$Q_{\\beta}$ (amu$^{0.5}$ Å)', labelpad=5)
ax0.set_ylabel('$Q_{\\alpha}$ (amu$^{0.5}$ Å)')

ax0.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax0.set_yticklabels(['', '$0.4$', '', '$0.0$', '', '$-0.4$', ''])

ax0.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax0.set_xticklabels(['$-0.4$', '', '$0.0$', '', '$0.4$', ''])

ax0.tick_params(axis='both', direction='in')
ax0.tick_params(which='minor', direction='in')
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')

ax0.set_aspect('equal')

ax0.text(x=-1.0, y=0.65, s='e', fontsize=15, weight='bold')

###############
# contour 1ep #
###############

lb = pes_numerical_solver(x, y, eph, Ft, F)[1]

ref_min = np.min(lb)

for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        if lb[i,j] > 290:
            lb[i,j] = nan

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
labels = ['$^3A_2$', '$^1E$', '$^1E^{\prime}$', '$^1A_1$']
markers = ['s', 'o', '^', 'v']

im = ax05.contour(x, y, lb, 10, colors='white', linestyles='-', linewidths=0.01)
im = ax05.contourf(x, y, lb, 10, cmap="turbo", vmin=_min, vmax=_max)

divider = make_axes_locatable(ax05)
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im, cax=cax, orientation='vertical',ticks=[0, 60, 120, 180, 240, 300])
cbar.ax.set_yticklabels(['0', '', '120', '', '240', ''])

cbar.set_label('Energy (meV)')

ax05.set_ylabel('$Q_{\\alpha}$ (amu$^{0.5}$ Å)')

ax05.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax05.set_yticklabels(['', '$0.4$', '', '$0.0$', '', '$-0.4$', ''])

ax05.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax05.set_xticklabels([])

ax05.tick_params(axis='both', direction='in')
ax05.tick_params(which='minor', direction='in')
ax05.xaxis.set_ticks_position('both')
ax05.yaxis.set_ticks_position('both')

ax05.set_aspect('equal')

ax05.text(x=-1, y=0.65, s='d', fontsize=15, weight='bold')

###############
# contour 1a1 #
###############

lb = pes_numerical_solver(x, y, eph, Ft, F)[2]

ref_min = np.min(lb)

for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        if lb[i,j] > 250 + ref_min:
            lb[i,j] = nan

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
labels = ['$^3A_2$', '$^1E$', '$^1E^{\prime}$', '$^1A_1$']
markers = ['s', 'o', '^', 'v']

im = ax1.contour(x, y, lb, 10, colors='white', linestyles='-', linewidths=0.01)
im = ax1.contourf(x, y, lb, 10, cmap="turbo", vmin=_min, vmax=_max)

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[800, 850, 900, 950, 1000, 1050])
cbar.ax.set_yticklabels(['800', '', '900', '', '1000', ''])

cbar.set_label('Energy (meV)')

ax1.set_ylabel('$Q_{\\alpha}$ (amu$^{0.5}$ Å)')

ax1.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax1.set_yticklabels(['', '$0.4$', '', '$0.0$', '', '$-0.4$', ''])

ax1.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax1.set_xticklabels([])

ax1.tick_params(axis='both', direction='in')
ax1.tick_params(which='minor', direction='in')
ax1.xaxis.set_ticks_position('both')
ax1.yaxis.set_ticks_position('both')

ax1.set_aspect('equal')

ax1.text(x=-1, y=0.65, s='c', fontsize=15, weight='bold')

############
# surfaces #
############

ax10.axis('off')
ax10.set_xlim((0,1))
ax10.set_ylim((0,1))
cwd = os.getcwd()
im1 = plt.imread(get_sample_data(current_working_directory + '/pes-3D.png'))
newax = fig.add_axes([-0.02, 0.53, 0.5, 0.5], anchor='NE')
newax.imshow(im1)
newax.axis('off')
ax10.text(x=-0.05, y=1.03, s='a', weight='bold', fontsize=15)

############
# vibronic #
############

# Create a Rectangle patch
rect = patches.Rectangle((0.038, -0.12), 0.96, 1.01, linewidth=1, edgecolor='black', facecolor='none', clip_on=False)

# Add the patch to the Axes
ax30.add_patch(rect)

ax30.axis('off')
ax30.set_xlim((0,1))
ax30.set_ylim((0,1))
cwd = os.getcwd()
im1 = plt.imread(get_sample_data(current_working_directory + '/vibronic-levels.png'))
newax = fig.add_axes([-0.05, -0.03, 0.5, 0.5], anchor='NE')
newax.imshow(im1)
newax.axis('off')
ax30.text(x=-0.05, y=0.9, s='b', weight='bold', fontsize=15)

##
plt.savefig("Fig-4.pdf",bbox_inches = 'tight',dpi=300)
plt.show()
