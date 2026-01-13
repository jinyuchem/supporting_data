#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})
from scipy.optimize import curve_fit
from scipy import constants

#############
# Constants #
#############

eV_J = constants.eV
Ang_M = 1e-10
N_A = constants.N_A
AMU_kg = 1e-3/N_A
hplanck = constants.h

############
# Function #
############

def curve_20(x, a):
    return a * x**2

def curve_21(x, a, b, c):
    return a * (x-b)**2 + c

def curve_30(x, a, d):
    return a * x**2 + d * x**3

def curve_31(x, a, b, c, d):
    return a * (x-b)**2 + c + d * (x-b)**3 

#######
# PBE #
#######

cod = np.loadtxt('pbe-gs-ene.dat', usecols=0) * 0.421736
gs_ene = np.loadtxt('pbe-gs-ene.dat', usecols=1)

es_ene = np.loadtxt('pbe-es-ene.dat', usecols=(1,2,3,4))
es_ene = es_ene.T

ene = np.copy(es_ene)
ene[0] = 0.0
for i in range(4):
    ene[i,:] = ene[i,:] + gs_ene[:]

for i in range(4):
    ene[i,:] = ene[i,:] - ene[i,1]

ene = ene * 13.6056980659

# 1A1 curve_20
p_1a1_20, pcov = curve_fit(curve_20, cod[:], ene[3,:])
print('=' * 60)
print('Parameters for 1A1 curve: ', p_1a1_20)
residuals = ene[3,:] - curve_20(cod, *p_1a1_20)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ene[3,:] - np.mean(ene[3,:]))**2)
r_squared = 1 - (ss_res / ss_tot)
print('$R^2$ is: ', r_squared)

phonon_1a1 = 2 * p_1a1_20[0] * eV_J/(Ang_M**2 * AMU_kg)
phonon_1a1 = phonon_1a1 * (hplanck/2/np.pi)**2 / (eV_J**2)
phonon_1a1 = np.sqrt(phonon_1a1) * 1000
print('1A1 phonon is %.5f meV'%phonon_1a1)

# 1A1 curve_30
p_1a1_30, pcov = curve_fit(curve_30, cod[:], ene[3,:])
print('=' * 60)
print('Parameters for 1A1 curve: ', p_1a1_30)
residuals = ene[3,:] - curve_30(cod, p_1a1_30[0], p_1a1_30[1])
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ene[3,:] - np.mean(ene[3,:]))**2)
r_squared = 1 - (ss_res / ss_tot)
print('$R^2$ is: ', r_squared)

phonon_1a1 = 2 * p_1a1_30[0] * eV_J/(Ang_M**2 * AMU_kg)
phonon_1a1 = phonon_1a1 * (hplanck/2/np.pi)**2 / (eV_J**2)
phonon_1a1 = np.sqrt(phonon_1a1) * 1000
print('1A1 phonon is %.5f meV'%phonon_1a1)


#######
# DDH #
#######

print('=' * 60)
print('DDH')

ddh_cod = np.loadtxt('ddh-gs-ene.dat', usecols=0) * 0.417577
ddh_gs_ene = np.loadtxt('ddh-gs-ene.dat', usecols=1)

ddh_es_ene = np.loadtxt('ddh-es-ene.dat', usecols=(1,2,3,4))
ddh_es_ene = ddh_es_ene.T

ddh_ene = np.copy(ddh_es_ene)
ddh_ene[0] = 0.0
for i in range(4):
    ddh_ene[i,:] = ddh_ene[i,:] + ddh_gs_ene[:]

for i in range(4):
    ddh_ene[i,:] = ddh_ene[i,:] - ddh_ene[i,1]

ddh_ene = ddh_ene * 13.6056980659

# 1A1 curve_20
ddh_p_1a1_20, pcov = curve_fit(curve_20, ddh_cod[:], ddh_ene[3,:])
print('=' * 60)
print('Parameters for 1A1 curve: ', ddh_p_1a1_20)
residuals = ddh_ene[3,:] - curve_20(ddh_cod, *ddh_p_1a1_20)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ddh_ene[3,:] - np.mean(ddh_ene[3,:]))**2)
r_squared = 1 - (ss_res / ss_tot)
print('$R^2$ is: ', r_squared)

ddh_phonon_1a1 = 2 * ddh_p_1a1_20[0] * eV_J/(Ang_M**2 * AMU_kg)
ddh_phonon_1a1 = ddh_phonon_1a1 * (hplanck/2/np.pi)**2 / (eV_J**2)
ddh_phonon_1a1 = np.sqrt(ddh_phonon_1a1) * 1000
print('1A1 phonon is %.5f meV'%ddh_phonon_1a1)

# 1A1 curve_30
ddh_p_1a1_30, pcov = curve_fit(curve_30, ddh_cod[:], ddh_ene[3,:])
print('=' * 60)
print('Parameters for 1A1 curve: ', ddh_p_1a1_30)
residuals = ddh_ene[3,:] - curve_30(ddh_cod, ddh_p_1a1_30[0], ddh_p_1a1_30[1])
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ddh_ene[3,:] - np.mean(ddh_ene[3,:]))**2)
r_squared = 1 - (ss_res / ss_tot)
print('$R^2$ is: ', r_squared)

ddh_phonon_1a1 = 2 * ddh_p_1a1_30[0] * eV_J/(Ang_M**2 * AMU_kg)
ddh_phonon_1a1 = ddh_phonon_1a1 * (hplanck/2/np.pi)**2 / (eV_J**2)
ddh_phonon_1a1 = np.sqrt(ddh_phonon_1a1) * 1000
print('1A1 phonon is %.5f meV'%ddh_phonon_1a1)


########
# Plot #
########

fig, ax = plt.subplots(2, 1, figsize=(6, 6))

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
labels = ['$^3A_2$', '$^1E$', '$^1E^{\prime}$', '$^1A_1$']
markers = ['s', 'o', '^', 'v']

# PBE
ax[0].plot(cod, ene[3], color=colors[3], label='', linestyle='',
           marker=markers[3], markersize=4)
ax[0].axvline(x=cod[1], ymin=0, ymax=1, color='gray', linestyle='--')
ax[0].axvline(x=cod[-2], ymin=0, ymax=1, color='gray', linestyle='--')
ax[0].axhline(y=0.0, xmin=0, xmax=1, color='black', linestyle='-', linewidth=0.5)

ax[0].set_xlim((-0.1,0.5))
ax[0].set_xticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax[0].set_xticklabels([])

ax[0].tick_params(axis='both', direction='in')
ax[0].tick_params(which='minor', direction='in')
ax[0].xaxis.set_ticks_position('both')
ax[0].yaxis.set_ticks_position('both')

cod_ene = np.linspace(-0.1,0.5,121)
ene_1a1_20 = curve_20(cod_ene, p_1a1_30[0])
ene_1a1_30 = curve_30(cod_ene, *p_1a1_30)

ax[0].plot(cod_ene, ene_1a1_20, color='red', linestyle='-', linewidth=0.5, label='w/o $Q^3$')
ax[0].plot(cod_ene, ene_1a1_30, color='blue', linestyle='-', linewidth=0.5, label='w/ $Q^3$')
ax[0].legend(fontsize=12,loc='upper left',edgecolor='black')

ax[0].text(x=0.1, y=0.15, s='$a_2 = % .3f$ (eV amu$^{-1}$ Å$^{-2}$)\n$a_3 = % .3f$ (eV amu$^{-1.5}$ Å$^{-3}$)'%(p_1a1_30[0], p_1a1_30[1]), fontsize=10)

ax[0].set_xticklabels([])
ax[0].set_ylim(-0.01, 0.22)

ax[0].text(x=-0.07, y=0.12, s='PBE')

# DDH
ax[1].plot(ddh_cod, ddh_ene[3], color=colors[3], label='', linestyle='',
           marker=markers[3], markersize=4)
ax[1].axvline(x=ddh_cod[1], ymin=0, ymax=1, color='gray', linestyle='--')
ax[1].axvline(x=ddh_cod[-1], ymin=0, ymax=1, color='gray', linestyle='--')
ax[1].axhline(y=0.0, xmin=0, xmax=1, color='black', linestyle='-', linewidth=0.5)

ax[1].set_xlim((-0.1,0.5))
ax[1].set_xticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax[1].set_xticklabels([])

ax[1].tick_params(axis='both', direction='in')
ax[1].tick_params(which='minor', direction='in')
ax[1].xaxis.set_ticks_position('both')
ax[1].yaxis.set_ticks_position('both')

cod_ene = np.linspace(-0.1,0.5,121)
ene_1a1_20 = curve_20(cod_ene, ddh_p_1a1_30[0])
ene_1a1_30 = curve_30(cod_ene, *ddh_p_1a1_30)

ax[1].plot(cod_ene, ene_1a1_20, color='red', linestyle='-', linewidth=0.5, label='w/o $Q^3$')
ax[1].plot(cod_ene, ene_1a1_30, color='blue', linestyle='-', linewidth=0.5, label='w/ $Q^3$')
ax[1].legend(fontsize=12,loc='upper left',edgecolor='black')

ax[1].text(x=0.1, y=0.15, s='$a_2 = % .3f$ (eV amu$^{-1}$ Å$^{-2}$)\n$a_3 = % .3f$ (eV amu$^{-1.5}$ Å$^{-3}$)'%(ddh_p_1a1_30[0], ddh_p_1a1_30[1]), fontsize=10)

ax[1].set_xticklabels([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax[1].set_xlabel('$Q$ (amu$^{0.5}$ Å)')
ax[1].set_ylim(-0.01, 0.22)

ax[1].text(x=-0.07, y=0.12, s='DDH')

fig.add_subplot( 111, frameon = False )
plt.tick_params( labelcolor = 'none', top = False, bottom = False, left = False, right = False )
plt.ylabel( "$\Delta E$ (eV)", labelpad = 10, fontsize=12)
plt.subplots_adjust( wspace = 0.2, hspace = 0.1)

plt.savefig("Fig-S8.pdf",bbox_inches = 'tight',dpi=300)
plt.show()
