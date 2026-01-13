#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 14})
from scipy.optimize import curve_fit
from scipy import constants
from scipy.linalg import eigh
from math import nan
from matplotlib.gridspec import GridSpec
from matplotlib.cbook import get_sample_data
import os
from mpl_toolkits.mplot3d import axes3d

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
                      Ft * x[i,j],
                      Ft * y[i,j]
                  ],
                  [
                      Ft * x[i,j],
                      F * x[i,j],
                      - F * y[i,j]
                  ],
                  [
                      Ft * y[i,j],
                      - F * y[i,j],
                      - F * x[i,j]
                  ]
                  ])

            w, v = eigh(mat)
            elow[i,j] = float(w[0]) + 0.5 * eph * (x[i,j]**2 + y[i,j]**2)
            ehigh[i,j] = float(w[1]) + 0.5 * eph * (x[i,j]**2 + y[i,j]**2)
            a[i,j] = float(w[2]) + 0.5 * eph * (x[i,j]**2 + y[i,j]**2) #- Le

    return elow, ehigh, a

##############
# Parameters #
##############

Le = 821

eph = 62.9506828
Ft = 133.22436286
F = 62.37653058

########
# Plot #
########

R = np.linspace(0, 0.75, 101)
theta = np.linspace(0, 2*np.pi, 101)

x = np.outer(R, np.cos(theta))
y = np.outer(R, np.sin(theta))

# 1e_lower
lb = pes_numerical_solver(x, y, eph, Ft, F)

X1, Y1 = x, y
Z1 = lb[0]

R = np.linspace(0, 0.6, 101)
theta = np.linspace(0, 2*np.pi, 101)

x = np.outer(R, np.cos(theta))
y = np.outer(R, np.sin(theta))

lb = pes_numerical_solver(x, y, eph, Ft, F)

X2, Y2 = x, y
Z2 = lb[1]
Z3 = lb[2]

_min = np.min(Z1)
_max = np.max(Z3)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection="3d")

im = ax.plot_surface(X2, Y2, Z2, cmap="turbo", lw=1, rstride=1, cstride=1, vmin=_min, vmax=_max)
ax.plot_surface(X2, Y2, Z3, cmap="turbo", lw=1, rstride=1, cstride=1, vmin=_min, vmax=_max)
ax.plot_surface(X1, Y1, Z1, cmap="turbo", lw=1, rstride=1, cstride=1, vmin=_min, vmax=_max)

ax.set_ylabel('$Q_{\\beta}$ (amu$^{0.5}$ Å)', labelpad=0)
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel("Energy (meV)", labelpad=10, rotation=90)

ax.set_zticks([0,200,400,600,800,1000])
ax.set_zticklabels(['0', '', '400', '', '800', ''])
ax.set_yticks([-0.6,-0.3,0.0,0.3,0.6])
ax.set_yticklabels(['$-0.6$', '', '0.0', '', '0.6'], fontsize=14)
ax.set_xticks([-0.6,-0.3,0.0,0.3,0.6])
ax.set_xticklabels(['$-0.6$', '', '0.0', '', '0.6'], fontsize=14)

ax.tick_params(axis='x', which='major', labelsize=14, pad=-4)
ax.tick_params(axis='y', which='major', labelsize=14, pad=-4)
ax.set_xlabel('$Q_{\\alpha}$ (amu$^{0.5}$ Å)', labelpad=0)

ax.text(x=0.6, y=-0.7, z=1000, s='$^1A_1$')
ax.text(x=0.8, y=-0.78, z=350, s='$^1E_{\mathrm{higher}}$')
ax.text(x=0.8, y=-0.78, z=-80, s='$^1E_{\mathrm{lower}}$')

ax.set_box_aspect((np.ptp(X1), np.ptp(Y2), np.ptp(Z1)/85))

ax.view_init(azim=45, elev=15)


cbar = fig.colorbar(im, shrink=0.55, aspect=15, pad=0.0, ticks=[0, 200, 400, 600, 800, 1000])
cbar.ax.set_yticklabels(['0', '', '400', '', '800', ''])

plt.savefig("pes-3D.png",bbox_inches = 'tight',dpi=300)
plt.show()
