#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})
from scipy.optimize import curve_fit
from scipy import constants
from scipy.linalg import eigh
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

#######
# fxn #
#######

def harmonic(x, y, eph):
    return 0.5 * eph * (x**2 + y**2)

def elow_x_numerical_solver(x, eph, Ft, F):
    # Note: transform x coordinates from amu^{0.5} Å into unit less
    x = x * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
    t_eph = eph * 1e-3 * constants.eV / constants.hbar
    x = x * np.sqrt(t_eph / constants.hbar)

    y = np.zeros(x.shape[0])

    Gt = 0
    G = 0

    elow = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        mat = np.array([
              [
                  Le,
                  Ft * x[i],
                  Ft * y[i]
              ],
              [
                  Ft * x[i],
                  F * x[i],
                  - F * y[i]
              ],
              [
                  Ft * y[i],
                  - F * y[i],
                  - F * x[i]
              ]
              ])

        w, v = eigh(mat)
        elow[i] = float(w[0]) + 0.5 * eph * (x[i]**2 + y[i]**2)

    return elow

def ehigh_x_numerical_solver(x, eph, Ft, F):
    # Note: transform x coordinates from amu^{0.5} Å into unit less
    x = x * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
    t_eph = eph * 1e-3 * constants.eV / constants.hbar
    x = x * np.sqrt(t_eph / constants.hbar)

    y = np.zeros(x.shape[0])

    Gt = 0
    G = 0

    ehigh = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        mat = np.array([
              [
                  Le,
                  Ft * x[i],
                  Ft * y[i]
              ],
              [
                  Ft * x[i],
                  F * x[i],
                  - F * y[i]
              ],
              [
                  Ft * y[i],
                  - F * y[i],
                  - F * x[i]
              ]
              ])

        w, v = eigh(mat)
        ehigh[i] = float(w[1]) + 0.5 * eph * (x[i]**2 + y[i]**2)

    return ehigh

def a_x_numerical_solver(x, eph, Ft, F):
    # Note: transform x coordinates from amu^{0.5} Å into unit less
    x = x * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
    t_eph = eph * 1e-3 * constants.eV / constants.hbar
    x = x * np.sqrt(t_eph / constants.hbar)

    y = np.zeros(x.shape[0])

    Gt = 0
    G = 0

    a = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        mat = np.array([
              [
                  Le,
                  Ft * x[i],
                  Ft * y[i]
              ],
              [
                  Ft * x[i],
                  F * x[i],
                  - F * y[i]
              ],
              [
                  Ft * y[i],
                  - F * y[i],
                  - F * x[i]
              ]
              ])

        w, v = eigh(mat)
        a[i] = float(w[2]) + 0.5 * eph * (x[i]**2 + y[i]**2) - Le

    return a




def elow_y_numerical_solver(x, eph, Ft, F):
    # Note: transform x coordinates from amu^{0.5} Å into unit less
    x = x * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
    t_eph = eph * 1e-3 * constants.eV / constants.hbar
    x = x * np.sqrt(t_eph / constants.hbar)

    y = x
    x = np.zeros(y.shape[0])

    Gt = 0
    G = 0

    elow = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        mat = np.array([
              [
                  Le,
                  Ft * x[i],
                  Ft * y[i]
              ],
              [
                  Ft * x[i],
                  F * x[i],
                  - F * y[i]
              ],
              [
                  Ft * y[i],
                  - F * y[i],
                  - F * x[i]
              ]
              ])

        w, v = eigh(mat)
        elow[i] = float(w[0]) + 0.5 * eph * (x[i]**2 + y[i]**2)

    return elow

def ehigh_y_numerical_solver(x, eph, Ft, F):
    # Note: transform x coordinates from amu^{0.5} Å into unit less
    x = x * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
    t_eph = eph * 1e-3 * constants.eV / constants.hbar
    x = x * np.sqrt(t_eph / constants.hbar)

    y = x
    x = np.zeros(y.shape[0])

    Gt = 0
    G = 0

    ehigh = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        mat = np.array([
              [
                  Le,
                  Ft * x[i],
                  Ft * y[i]
              ],
              [
                  Ft * x[i],
                  F * x[i],
                  - F * y[i]
              ],
              [
                  Ft * y[i],
                  - F * y[i],
                  - F * x[i]
              ]
              ])

        w, v = eigh(mat)
        ehigh[i] = float(w[1]) + 0.5 * eph * (x[i]**2 + y[i]**2)

    return ehigh

def a_y_numerical_solver(x, eph, Ft, F):
    # Note: transform x coordinates from amu^{0.5} Å into unit less
    x = x * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
    t_eph = eph * 1e-3 * constants.eV / constants.hbar
    x = x * np.sqrt(t_eph / constants.hbar)

    y = x
    x = np.zeros(y.shape[0])

    Gt = 0
    G = 0

    a = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        mat = np.array([
              [
                  Le,
                  Ft * x[i],
                  Ft * y[i]
              ],
              [
                  Ft * x[i],
                  F * x[i],
                  - F * y[i]
              ],
              [
                  Ft * y[i],
                  - F * y[i],
                  - F * x[i]
              ]
              ])

        w, v = eigh(mat)
        a[i] = float(w[2]) + 0.5 * eph * (x[i]**2 + y[i]**2) - Le

    return a





#############
# load data #
#############

cod_x = np.loadtxt('path-1-gs-ene.dat', usecols=0) * 0.825090 - 0.825090*0.5
gs_ene = np.loadtxt('path-1-gs-ene.dat', usecols=1)
es_ene = np.loadtxt('path-1-es-ene.dat', usecols=(2,3,4))

elow_x = es_ene[:,0] + gs_ene[:]
elow_x = elow_x - elow_x[6]
elow_x = elow_x * 13.6056980659 * 1000

ehigh_x = es_ene[:,1] + gs_ene[:]
ehigh_x = ehigh_x - ehigh_x[6]
ehigh_x = ehigh_x * 13.6056980659 * 1000

a_x = es_ene[:,2] + gs_ene[:]
a_x = a_x - a_x[6]
a_x = a_x * 13.6056980659 * 1000



cod_y = np.loadtxt('path-v-gs-ene.dat', usecols=0) * 0.825090 - 0.825090*0.5
gs_ene = np.loadtxt('path-v-gs-ene.dat', usecols=1)
es_ene = np.loadtxt('path-v-es-ene.dat', usecols=(2,3,4))

elow_y = es_ene[:,0] + gs_ene[:]
elow_y = elow_y - elow_y[6]
elow_y = elow_y * 13.6056980659 * 1000

ehigh_y = es_ene[:,1] + gs_ene[:]
ehigh_y = ehigh_y - ehigh_y[6]
ehigh_y = ehigh_y * 13.6056980659 * 1000

a_y = es_ene[:,2] + gs_ene[:]
a_y = a_y - a_y[6]
a_y = a_y * 13.6056980659 * 1000


##############
# parameters #
##############

# path along y=0
Le = 821


###############################
# Final: fit x and y together #
###############################

print('=' * 60)
print('Fit together X and Y')

ini_para = np.array([65, 140, 600])

def combine_funct(x, eph, Ft, F):
    elow = elow_x_numerical_solver(x[:13], eph, Ft, F)
    ehigh = ehigh_x_numerical_solver(x[13:26], eph, Ft, F)
    a = a_x_numerical_solver(x[26:], eph, Ft, F)
    return np.append(np.append(elow, ehigh), a)

def combine_funct_y(x, eph, Ft, F):
    elow = elow_y_numerical_solver(x[:13], eph, Ft, F)
    ehigh = ehigh_y_numerical_solver(x[13:26], eph, Ft, F)
    a = a_y_numerical_solver(x[26:], eph, Ft, F)
    return np.append(np.append(elow, ehigh), a)

def combine_funct_tot(x, eph, Ft, F):
    e_x = combine_funct(x[:39], eph, Ft, F)
    e_y = combine_funct_y(x[39:], eph, Ft, F)
    return np.append(e_x, e_y)


c_cod_x = np.append(np.append(cod_x, cod_x), cod_x)
c_e_x = np.append(np.append(elow_x, ehigh_x), a_x)

c_cod_y = np.append(np.append(cod_y, cod_y), cod_y)
c_e_y = np.append(np.append(elow_y, ehigh_y), a_y)

c_cod = np.append(c_cod_x, c_cod_y)
c_e = np.append(c_e_x, c_e_y)

all_para, pcov = curve_fit(combine_funct_tot, c_cod, c_e, ini_para)
print(all_para)

########
# Plot #
########

fig, ax = plt.subplots(1, 2, figsize=(12,6))

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
labels = ['$^3A_2$', '$^1E$', '$^1E^{\prime}$', '$^1A_1$']
markers = ['s', 'o', '^', 'v']

ax[0].scatter(cod_x, elow_x, color=colors[0], marker='o', s=30, label='$^1E_{low}$')
ax[0].plot(cod_x, elow_x_numerical_solver(cod_x, *all_para), color=colors[0],
           linestyle='--', label='$^1E_{low}$ fitted')

ax[0].scatter(cod_x, ehigh_x, color=colors[1], marker='o', s=30, label='$^1E_{high}$')
ax[0].plot(cod_x, ehigh_x_numerical_solver(cod_x, *all_para), color=colors[1],
           linestyle='--', label='$^1E_{high}$ fitted')

ax[0].scatter(cod_x, a_x, color=colors[3], marker='o', s=30, label='$^1A_1$')
ax[0].plot(cod_x, a_x_numerical_solver(cod_x, *all_para), color=colors[3],
           linestyle='--', label='$^1A_{1}$ fitted')


ax[0].axvline(x=cod_x[1], ymin=0, ymax=1, color='gray', linestyle='--')
ax[0].axvline(x=cod_x[-2], ymin=0, ymax=1, color='gray', linestyle='--')
ax[0].axhline(y=0.0, xmin=0, xmax=1, color='black', linestyle='-', linewidth=0.5)
ax[0].legend(fontsize=12,loc='best', edgecolor='black')
ax[0].set_xlim((-0.6,0.6))
ax[0].set_ylim((-80, 250))
ax[0].tick_params(axis='both', direction='in')
ax[0].tick_params(which='minor', direction='in')
ax[0].xaxis.set_ticks_position('both')
ax[0].yaxis.set_ticks_position('both')
ax[0].set_xlabel('$Q_{\\alpha}$ (amu$^{0.5}$ Å)')
ax[0].set_ylabel('$E$ (meV)')


ax[1].scatter(cod_y, elow_y, color=colors[0], marker='o', s=30, label='$^1E_{low}$')
ax[1].plot(cod_y, elow_y_numerical_solver(cod_y, *all_para), color=colors[0],
           linestyle='--', label='$^1E_{low}$ fitted')

ax[1].scatter(cod_y, ehigh_y, color=colors[1], marker='o', s=30, label='$^1E_{high}$')
ax[1].plot(cod_y, ehigh_y_numerical_solver(cod_y, *all_para), color=colors[1],
           linestyle='--', label='$^1E_{high}$ fitted')

ax[1].scatter(cod_y, a_y, color=colors[3], marker='o', s=30, label='$^1A_1$')
ax[1].plot(cod_y, a_y_numerical_solver(cod_y, *all_para), color=colors[3],
           linestyle='--', label='$^1A_{1}$ fitted')


ax[1].axhline(y=0.0, xmin=0, xmax=1, color='black', linestyle='-', linewidth=0.5)
ax[1].set_xlim((-0.6,0.6))
ax[1].set_ylim((-80, 250))
ax[1].tick_params(axis='both', direction='in')
ax[1].tick_params(which='minor', direction='in')
ax[1].xaxis.set_ticks_position('both')
ax[1].yaxis.set_ticks_position('both')
ax[1].set_xlabel('$Q_{\\beta}$ (amu$^{0.5}$ Å)')


plt.savefig("Fig-S2.pdf",bbox_inches = 'tight',dpi=300)
plt.show()
