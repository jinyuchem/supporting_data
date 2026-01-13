#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 12})
from scipy.optimize import curve_fit
from scipy import constants
from scipy.linalg import eigh
from math import nan
from matplotlib.gridspec import GridSpec
from matplotlib.cbook import get_sample_data
import os

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
            a[i,j] = float(w[2]) + 0.5 * eph * (x[i]**2 + y[j]**2)

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

fig = plt.figure(figsize=(12, 4))
gs = GridSpec(nrows=3, ncols=11, height_ratios=[1, 1.5, 1],
                                width_ratios=[1, 1, 2, 1.5, 1, 1, 2, 0.5, 1, 1, 2],
                                hspace=0.02, wspace=1,
                                left=0.05, right=0.9,
                                bottom=0.02, top=0.98)

ax0 = fig.add_subplot(gs[:,0:3])

ax10 = fig.add_subplot(gs[0,4:7])
ax11 = fig.add_subplot(gs[1,4:7])
ax12 = fig.add_subplot(gs[2,4:7])

ax20 = fig.add_subplot(gs[0,8:])
ax21 = fig.add_subplot(gs[1,8:])
ax22 = fig.add_subplot(gs[2,8:])

##############
# contour 1e #
##############

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
labels = ['$^3A_2$', '$^1E$', '$^1E^{\prime}$', '$^1A_1$']
markers = ['s', 'o', '^', 'v']

im = ax0.contourf(x, y, lb, 10, cmap="turbo", vmin=_min, vmax=_max)

divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('Energy (meV)')

# path 1
coords_y = np.linspace(-0.1,1.1,13)
coords_y = coords_y - 0.5
coords_y = coords_y * 0.825090
coords_x = np.zeros(13)

ax0.plot(coords_x, coords_y, marker='o', markersize=5, color='white', linewidth=0, label='path 1')

# path 3
coords_x = np.linspace(-0.1,1.1,13)
coords_x = coords_x - 0.5
coords_x = coords_x * 0.825090
coords_y = np.zeros(13)

ax0.plot(coords_x, coords_y, marker='^', markersize=5, color='white', linewidth=0, label='path 2')
legend = ax0.legend(fontsize=11, loc='best', edgecolor='black')
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((1, 1, 1, 0.25))

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

ax0.text(x=-1.0, y=0.65, s='a', fontsize=15, weight='bold')

##########
# Path 1 #
##########

# colors
google_g = np.array([15, 157, 88]) / 255

google_r = np.array([219, 68, 55]) / 255

google_b = np.array([66, 133, 244]) / 255

google_y = np.array([244, 180, 0]) / 255

# 1a1: green, ex: red, ey: blue

def curve_20(x, a):
    return a * x**2

def curve_30(x, a, d):
    return a * x**2 + d * x**3

def curve_200(x, a):
    return a * x**2

def curve_300(x, a, d):
    return a * x**2 + d * x**3

##############
# Parameters #
##############

cod = np.loadtxt('path-1-gs-ene.dat', usecols=0) * 0.825090 - 0.825090/2
gs_ene = np.loadtxt('path-1-gs-ene.dat', usecols=1)

es_ene = np.loadtxt('path-1-es-ene.dat', usecols=(1,2,3,4))
es_ene = es_ene.T

o_ene = np.copy(es_ene)
o_ene[0] = 0.0
for i in range(4):
    o_ene[i,:] = o_ene[i,:] + gs_ene[:]

o_ene[0,:] = o_ene[0,:] - min(o_ene[0,:])
o_ene[2,:] = o_ene[2,:] - o_ene[1,6]
o_ene[1,:] = o_ene[1,:] - o_ene[1,6]
o_ene[3,:] = o_ene[3,:] - min(o_ene[3,:])

o_ene = o_ene * 13.6056980659

ene = np.zeros(o_ene.shape)
ene[0] = o_ene[0]
ene[3] = o_ene[3]
ene[1] = o_ene[1]
ene[2] = o_ene[2]

#######
# fit #
#######

# 1A1 curve_20
p_1a1_20, pcov = curve_fit(curve_20, cod[:], ene[3,:])

# 1A1 curve_30
p_1a1_30, pcov = curve_fit(curve_30, cod[:], ene[3,:])

# 3A2 curve_20
p_3a2_20, pcov = curve_fit(curve_20, cod[:], ene[0,:])

# 3A2 curve_30
p_3a2_30, pcov = curve_fit(curve_30, cod[:], ene[0,:])

########
# Plot #
########

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
labels = ['$^3A_2$', '$^1E$', '$^1E^{\prime}$', '$^1A_1$']
markers = ['s', 'o', '^', 'v']

# 3a2
ax12.scatter(cod, 1e3*ene[0], color=colors[2], marker='o', s=30)
ax12.axhline(y=0.0, xmin=0, xmax=1, color='black', linestyle='-', linewidth=0.5)
ax12.set_xlim((-0.6,0.6))
ax12.set_ylim((-25, 250))
ax12.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax12.set_xticklabels([])
ax12.tick_params(axis='both', direction='in')
ax12.tick_params(which='minor', direction='in')
ax12.xaxis.set_ticks_position('both')
ax12.yaxis.set_ticks_position('both')

ax12.text(x=-0.56, y=250*0.8, s='$^3A_2$')

# 1e
colors_1e = np.loadtxt('path-1-wfxn.dat', usecols=(4,5,6))**2
for i in range(cod.shape[0]):
    colors_1e[i] = colors_1e[i] / sum(colors_1e[i])
    colors_1e[i] = np.dot(colors_1e[i], np.vstack((google_g, google_r, google_b)))

ax11.scatter(cod, 1e3*ene[1], color=colors_1e, marker='o', s=30)

ax11.axhline(y=0.0, xmin=0, xmax=1, color='black', linestyle='-', linewidth=0.5)
ax11.set_xlim((-0.6,0.6))
ax11.set_ylim((0, 80))
ax11.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax11.set_xticklabels([])
ax11.tick_params(axis='both', direction='in')
ax11.tick_params(which='minor', direction='in')
ax11.xaxis.set_ticks_position('both')
ax11.yaxis.set_ticks_position('both')

# 1ep
colors_1ep = np.loadtxt('path-1-wfxn.dat', usecols=(7,8,9))**2
for i in range(cod.shape[0]):
    colors_1ep[i] = colors_1ep[i] / sum(colors_1ep[i])
    colors_1ep[i] = np.dot(colors_1ep[i], np.vstack((google_g, google_r, google_b)))

ax11.scatter(cod, 1e3*ene[2], color=colors_1ep, marker='o', s=30)
ax11.axhline(y=0.0, xmin=0, xmax=1, color='black', linestyle='-', linewidth=0.5)
ax11.set_xlim((-0.6,0.6))
ax11.set_ylim((-87, 270))
ax11.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax11.set_xticklabels([])
ax11.tick_params(axis='both', direction='in')
ax11.tick_params(which='minor', direction='in')
ax11.xaxis.set_ticks_position('both')
ax11.yaxis.set_ticks_position('both')

ax11.text(x=-0.56, y=230*0.8, s='$^1E$')

ax11.set_ylabel('$\Delta E$ (meV)')

# 1a1
colors_1a1 = np.loadtxt('path-1-wfxn.dat', usecols=(1,2,3))**2
for i in range(cod.shape[0]):
    colors_1a1[i] = colors_1a1[i] / sum(colors_1a1[i])
    colors_1a1[i] = np.dot(colors_1a1[i], np.vstack((google_g, google_r, google_b)))

ax10.scatter(cod, 1e3*ene[3], color=colors_1a1, marker='o', s=30)

ax10.axhline(y=0.0, xmin=0, xmax=1, color='black', linestyle='-', linewidth=0.5)
ax10.set_xlim((-0.6,0.6))
ax10.set_ylim((-34, 340))
ax10.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax10.set_xticklabels([])
ax10.tick_params(axis='both', direction='in')
ax10.tick_params(which='minor', direction='in')
ax10.xaxis.set_ticks_position('both')
ax10.yaxis.set_ticks_position('both')

ax10.text(x=-0.56, y=340*0.8, s='$^1A_1$')

ax10.text(x=-0.865/1.1*1.2, y=340*1.06-34, s='b', fontsize=15, weight='bold')

# fitting 1a1
cod_ene = np.linspace(-0.6,0.6,121)
ene_1a1_20 = curve_20(cod_ene, *p_1a1_20)
ene_1a1_30 = curve_30(cod_ene, *p_1a1_30)
ax10.plot(cod_ene, 1e3*ene_1a1_20, color='red', linestyle='--', linewidth=0.5, label='quadratic')
ax10.plot(cod_ene, 1e3*ene_1a1_30, color='blue', linestyle='--', linewidth=0.5, label='cubic')
ax10.legend(fontsize=10,loc='upper center',edgecolor='black')

# fitting 3a2
cod_ene = np.linspace(-0.6,0.6,121)
ene_3a2_20 = curve_20(cod_ene, *p_3a2_20)
ene_3a2_30 = curve_30(cod_ene, *p_3a2_30)
ax12.plot(cod_ene, 1e3*ene_3a2_20, color='red', linestyle='--', linewidth=0.5, label='2nd')
ax12.plot(cod_ene, 1e3*ene_3a2_30, color='blue', linestyle='--', linewidth=0.5, label='3rd')

ax12.set_xlabel('$Q_{\\alpha}$ (amu$^{0.5}$ Å)')
ax12.set_xticklabels(['', '$-0.4$', '', '0.0', '', '0.4', ''])

##########
# Path 2 #
##########

def curve_20(x, a):
    return a * x**2

def curve_30(x, a, d):
    return a * x**2 + d * x**3

cod = np.loadtxt('path-v-gs-ene.dat', usecols=0) * 0.825090 - 0.825090/2
gs_ene = np.loadtxt('path-v-gs-ene.dat', usecols=1)

es_ene = np.loadtxt('path-v-es-ene.dat', usecols=(1,2,3,4))
es_ene = es_ene.T

ene = np.copy(es_ene)
ene[0] = 0.0
for i in range(4):
    ene[i,:] = ene[i,:] + gs_ene[:]

shift = 13.690597617956922 / 13.6056980659 / 1000
ene[0,:] = ene[0,:] - min(ene[0,:])
ene[2,:] = ene[2,:] - ene[1,6]
ene[1,:] = ene[1,:] - ene[1,6]
ene[3,:] = ene[3,:] - min(ene[3,:])

ene = ene * 13.6056980659

#######
# fit #
#######

# 1A1 curve_20
p_1a1_20, pcov = curve_fit(curve_20, cod[:], ene[3,:])

# 1A1 curve_30
p_1a1_30, pcov = curve_fit(curve_30, cod[:], ene[3,:])

# 3A2 curve_20
p_3a2_20, pcov = curve_fit(curve_20, cod[:], ene[0,:])

# 3A2 curve_30
p_3a2_30, pcov = curve_fit(curve_30, cod[:], ene[0,:])

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
labels = ['$^3A_2$', '$^1E$', '$^1E^{\prime}$', '$^1A_1$']
markers = ['s', 'o', '^', 'v']

# 3a2
ax22.scatter(cod, 1e3*ene[0], color=colors[2], marker='^', s=30)
ax22.axhline(y=0.0, xmin=0, xmax=1, color='black', linestyle='-', linewidth=0.5)
ddd = 0.4 - cod[6]
ax22.set_xlim((-0.6,0.6))
ax22.set_ylim((-25, 250))
ax22.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax22.set_xticklabels([])
ax22.tick_params(axis='both', direction='in')
ax22.tick_params(which='minor', direction='in')
ax22.xaxis.set_ticks_position('both')
ax22.yaxis.set_ticks_position('both')

ax22.text(x=-0.58/1.2*1.1, y=250*0.8, s='$^3A_2$')

# 1e
colors_1e = np.loadtxt('path-v-wfxn.dat', usecols=(4,5,6))**2
for i in range(cod.shape[0]):
    colors_1e[i] = colors_1e[i] / sum(colors_1e[i])
    colors_1e[i] = np.dot(colors_1e[i], np.vstack((google_g, google_r, google_b)))

ax21.scatter(cod, 1e3*ene[1], color=colors_1e, marker='^', s=30)
ax21.axhline(y=0.0, xmin=0, xmax=1, color='black', linestyle='-', linewidth=0.5)
ax21.set_xlim((-0.6,0.6))
ax21.set_ylim((-0.0, 37))
ax21.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
ax21.set_xticklabels([])
ax21.tick_params(axis='both', direction='in')
ax21.tick_params(which='minor', direction='in')
ax21.xaxis.set_ticks_position('both')
ax21.yaxis.set_ticks_position('both')

# 1ep
colors_1ep = np.loadtxt('path-v-wfxn.dat', usecols=(7,8,9))**2
for i in range(cod.shape[0]):
    colors_1ep[i] = colors_1ep[i] / sum(colors_1ep[i])
    colors_1ep[i] = np.dot(colors_1ep[i], np.vstack((google_g, google_r, google_b)))

ax21.scatter(cod, 1e3*ene[2], color=colors_1ep, marker='^', s=30)

ax21.axhline(y=0.0, xmin=0, xmax=1, color='black', linestyle='-', linewidth=0.5)
ddd = 0.4 - cod[6]
ax21.set_xlim((-0.6,0.6))
ax21.set_ylim((-87,270))
ax21.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax21.set_xticklabels([])
ax21.tick_params(axis='both', direction='in')
ax21.tick_params(which='minor', direction='in')
ax21.xaxis.set_ticks_position('both')
ax21.yaxis.set_ticks_position('both')

ax21.text(x=-0.58*1.1/1.2-0.02, y=270*0.8, s='$^1E$')

ax21.set_ylabel('$\Delta E$ (meV) ')


###
# colorbar 
###

axins = inset_axes(ax21, width="30%", height="50%", loc='upper left',
                   bbox_to_anchor=(0.4,0,1,1), bbox_transform=ax21.transAxes)
axins.axis('off')
axins.set_xlim(0,1)
axins.set_ylim(0,1)
y = np.linspace(0.1,0.9,101)

cy0 = google_b[0] * y + google_r[0] * (1-y)
cy1 = google_b[1] * y + google_r[1] * (1-y)
cy2 = google_b[2] * y + google_r[2] * (1-y)

cy = np.array([cy0, cy1, cy2]).T

for i in range(101):
   axins.hlines(y=y[i], xmin=0.1, xmax=0.17, color=cy[i])

axins.hlines(y=0.1, xmin=0.1, xmax=0.17, color='black', linewidth=1)
axins.hlines(y=0.9, xmin=0.1, xmax=0.17, color='black', linewidth=1)
axins.vlines(x=0.1, ymin=0.1, ymax=0.9, color='black', linewidth=1)
axins.vlines(x=0.17, ymin=0.1, ymax=0.9, color='black', linewidth=1)
axins.text(x=0.2, y=0.06, s='$^1E_x^{(0)}$', fontsize=9)
axins.text(x=0.2, y=0.86, s='$^1E_y^{(0)}$', fontsize=9)




# 1a1
colors_1a1 = np.loadtxt('path-v-wfxn.dat', usecols=(1,2,3))**2
for i in range(cod.shape[0]):
    colors_1a1[i] = colors_1a1[i] / sum(colors_1a1[i])
    colors_1a1[i] = np.dot(colors_1a1[i], np.vstack((google_g, google_r, google_b)))

ax20.scatter(cod, 1e3*ene[3], color=colors_1a1, marker='^', s=30)

ax20.axhline(y=0.0, xmin=0, xmax=1, color='black', linestyle='-', linewidth=0.5)
ddd = 0.4 - cod[6]
ax20.set_xlim((-0.6,0.6))
ax20.set_ylim((-34, 340))
ax20.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
ax20.set_xticklabels([])
ax20.tick_params(axis='both', direction='in')
ax20.tick_params(which='minor', direction='in')
ax20.xaxis.set_ticks_position('both')
ax20.yaxis.set_ticks_position('both')

ax20.text(x=-0.58*1.1/1.2, y=340*0.8, s='$^1A_1$')

ax20.text(x=-0.9, y=340*1.06-34, s='c', fontsize=15, weight='bold')

# fit 1a1
cod_ene = np.linspace(-0.6,0.6,121)
ene_1a1_20 = curve_20(cod_ene, *p_1a1_20)
ene_1a1_30 = curve_30(cod_ene, *p_1a1_30)

ax20.plot(cod_ene, 1e3*ene_1a1_20, color='red', linestyle='--', linewidth=0.5, label='2nd')
ax20.plot(cod_ene, 1e3*ene_1a1_30, color='blue', linestyle='--', linewidth=0.5, label='3rd')

# fit 3a2
cod_ene = np.linspace(-0.6,0.6,121)
ene_3a2_20 = curve_20(cod_ene, *p_3a2_20)
ene_3a2_30 = curve_30(cod_ene, *p_3a2_30)

ax22.plot(cod_ene, 1e3*ene_3a2_20, color='red', linestyle='--', linewidth=0.5, label='2nd')
ax22.plot(cod_ene, 1e3*ene_3a2_30, color='blue', linestyle='--', linewidth=0.5, label='3rd')

ax22.set_xlabel('$Q_{\\beta}$ (amu$^{0.5}$ Å)')
ax22.set_xticklabels(['', '$-0.4$', '', '0.0', '', '0.4', ''])

plt.savefig("Fig-S1.pdf",bbox_inches = 'tight',dpi=300)
plt.show()
