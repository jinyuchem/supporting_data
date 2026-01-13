#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})
from matplotlib.gridspec import GridSpec
import os
from matplotlib.cbook import get_sample_data

#############
# Functions #
#############

def Gaussian(x,mu,sigma):
    prefix=1/np.sqrt(2*np.pi*sigma**2)
    eeee=np.exp( -(x-mu)**2/(2*sigma**2) )
    return prefix*eeee

def sigma(freq):
    f = 6 - (6 - 2) * freq / 160
    return f

#############
# Constants #
#############

NA=6.0221409e23
AMU_KG=1e-3/NA
ANG=1e-10
HBAR=1.0545718e-34
THZ_SN=1e12
C0=299792458
CMN_SN=C0*100
eV=1.60218e-19

########
# Main #
########

fnames = [
         'PBE-dq-PBE-ph-All-B-3A2-3E-13823-rc2-5-cph-5.dat',
         'DDH-dq-PBE-ph-All-B-3A2-3E-13823-rc2-5-cph-5.dat',
         'DDH-dq-DDH-ph-All-B-3A2-3E-13823-rc2-5-cph-5.dat',
         'PBE-dq-PBE-ph-All-B-1A1-1E-13823-rc2-5-cph-5.dat',
         'DDH-dq-PBE-ph-All-B-1A1-1E-13823-rc2-5-cph-5.dat',
         'DDH-dq-DDH-ph-All-B-1A1-1E-13823-rc2-5-cph-5.dat'
         ]

bf = []
freq = []
for i in range(6):
    bf.append(np.loadtxt(fnames[i], usecols=3))
    freq.append(np.loadtxt(fnames[i], usecols=6) * 1000/8065.544)

# deg?
wd = [
      np.zeros(freq[0].shape[0]),
      np.zeros(freq[1].shape[0]),
      np.zeros(freq[2].shape[0]),
      np.zeros(freq[3].shape[0]),
      np.zeros(freq[4].shape[0]),
      np.zeros(freq[5].shape[0])
     ]
for i in range(6):
    for j in range(1,freq[i].shape[0]):
        if abs(freq[i][j] - freq[i][j-1]) < 1e-5:
            wd[i][j] = 1
            wd[i][j-1] = 1

bf_a = [
        np.zeros(freq[0].shape[0]),
        np.zeros(freq[1].shape[0]),
        np.zeros(freq[2].shape[0]),
        np.zeros(freq[3].shape[0]),
        np.zeros(freq[4].shape[0]),
        np.zeros(freq[5].shape[0])
       ]
bf_e = [
        np.zeros(freq[0].shape[0]),
        np.zeros(freq[1].shape[0]),
        np.zeros(freq[2].shape[0]),
        np.zeros(freq[3].shape[0]),
        np.zeros(freq[4].shape[0]),
        np.zeros(freq[5].shape[0])
       ]

for i in range(6):
    bf_e[i][:] = bf[i][:] * wd[i][:]
    bf_a[i][:] = bf[i][:] - bf_e[i][:]

reso = 501

#######
# print ave freq
for i in range(6):
   ave_freq = sum(bf[i][:]**2) / sum(bf[i][:]**2 / freq[i][:])

   ave_freq = sum(bf_a[i][:]**2) / sum(bf_a[i][:]**2 / freq[i][:])

   ave_freq = sum(bf_e[i][:]**2) / sum(bf_e[i][:]**2 / freq[i][:])
#######

DATA = np.zeros((6,reso))
DATA_A = np.zeros((6,reso))
DATA_E = np.zeros((6,reso))

ENEAXIS = np.linspace(0, 200, reso, endpoint = True )

for i in range(reso):
    DATA[0,i] = sum(bf[0][:]**2 * Gaussian(ENEAXIS[i], freq[0][:], sigma(freq[0][:])))
    DATA[1,i] = sum(bf[1][:]**2 * Gaussian(ENEAXIS[i], freq[1][:], sigma(freq[1][:])))
    DATA[2,i] = sum(bf[2][:]**2 * Gaussian(ENEAXIS[i], freq[2][:], sigma(freq[2][:])))
    DATA[3,i] = sum(bf[3][:]**2 * Gaussian(ENEAXIS[i], freq[3][:], sigma(freq[3][:])))
    DATA[4,i] = sum(bf[4][:]**2 * Gaussian(ENEAXIS[i], freq[4][:], sigma(freq[4][:])))
    DATA[5,i] = sum(bf[5][:]**2 * Gaussian(ENEAXIS[i], freq[5][:], sigma(freq[5][:])))

    DATA_A[0,i] = sum(bf_a[0][:]**2 * Gaussian(ENEAXIS[i], freq[0][:], sigma(freq[0][:])))
    DATA_A[1,i] = sum(bf_a[1][:]**2 * Gaussian(ENEAXIS[i], freq[1][:], sigma(freq[1][:])))
    DATA_A[2,i] = sum(bf_a[2][:]**2 * Gaussian(ENEAXIS[i], freq[2][:], sigma(freq[2][:])))
    DATA_A[3,i] = sum(bf_a[3][:]**2 * Gaussian(ENEAXIS[i], freq[3][:], sigma(freq[3][:])))
    DATA_A[4,i] = sum(bf_a[4][:]**2 * Gaussian(ENEAXIS[i], freq[4][:], sigma(freq[4][:])))
    DATA_A[5,i] = sum(bf_a[5][:]**2 * Gaussian(ENEAXIS[i], freq[5][:], sigma(freq[5][:])))

    DATA_E[0,i] = sum(bf_e[0][:]**2 * Gaussian(ENEAXIS[i], freq[0][:], sigma(freq[0][:])))
    DATA_E[1,i] = sum(bf_e[1][:]**2 * Gaussian(ENEAXIS[i], freq[1][:], sigma(freq[1][:])))
    DATA_E[2,i] = sum(bf_e[2][:]**2 * Gaussian(ENEAXIS[i], freq[2][:], sigma(freq[2][:])))
    DATA_E[3,i] = sum(bf_e[3][:]**2 * Gaussian(ENEAXIS[i], freq[3][:], sigma(freq[3][:])))
    DATA_E[4,i] = sum(bf_e[4][:]**2 * Gaussian(ENEAXIS[i], freq[4][:], sigma(freq[4][:])))
    DATA_E[5,i] = sum(bf_e[5][:]**2 * Gaussian(ENEAXIS[i], freq[5][:], sigma(freq[5][:])))

########
# Plot #
########

fig, ax = plt.subplots(3, 2, figsize=(9,6))

ax00 = ax[0][0]
ax10 = ax[1][0]
ax20 = ax[2][0]
ax01 = ax[0][1]
ax11 = ax[1][1]
ax21 = ax[2][1]

# plot a
ylim = 0.77

colors = ['#DB4437', '#4285F4', '#F4B400', '#0F9D58']
linestyles = ['-', '-', '-', '--']
labels = [
         'PBE$-\Delta Q$, PBE$-ph$',
         'PBE$-\Delta Q$, PBE$-ph$, $a_1$ modes',
         'PBE$-\Delta Q$, PBE$-ph$, $e$ modes',
         'DDH$-\Delta Q$, PBE$-ph$',
         'DDH$-\Delta Q$, PBE$-ph$, $a_1$ modes',
         'DDH$-\Delta Q$, PBE$-ph$, $e$ modes',
         'DDH$-\Delta Q$, DDH$-ph$',
         'DDH$-\Delta Q$, DDH$-ph$, $a_1$ modes',
         'DDH$-\Delta Q$, DDH$-ph$, $e$ modes'
         ]

##########################
# fig-11
ax00.plot(ENEAXIS[:], DATA[0], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax00.plot(ENEAXIS[:], DATA[1], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[3])
ax00.plot(ENEAXIS[:], DATA[2], linewidth=1.5, linestyle=linestyles[2],
           color=colors[2], label=labels[6])

ax00.set_ylim((0,ylim/10))
ax00.set_xlim((0,200))
ax00.set_xticklabels([])

ax00.tick_params(axis='both', direction='in')
ax00.tick_params(which='minor', direction='in')
ax00.xaxis.set_ticks_position('both')
ax00.yaxis.set_ticks_position('both')

ax00.text(x=-45, y=ylim/10*1.07, s='a', fontsize=15, weight='bold')
ax00.text(x=10, y=ylim/10*0.88, s='$a_1 + e$ modes')

ax00.text(x=145, y=ylim/10*0.83, s='$^3E \\to ^3A_2$', fontsize=14)

# fig-12
ax10.plot(ENEAXIS[:], DATA_A[0], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax10.plot(ENEAXIS[:], DATA_A[1], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[3])
ax10.plot(ENEAXIS[:], DATA_A[2], linewidth=1.5, linestyle=linestyles[2],
           color=colors[2], label=labels[6])

ax10.set_ylim((0,ylim/10))
ax10.set_xlim((0,200))
ax10.set_xticklabels([])

ax10.tick_params(axis='both', direction='in')
ax10.tick_params(which='minor', direction='in')
ax10.xaxis.set_ticks_position('both')
ax10.yaxis.set_ticks_position('both')

ax10.text(x=10, y=ylim/10*0.88, s='$a_1$ modes')

# fig-13
ax20.plot(ENEAXIS[:], DATA_E[0], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax20.plot(ENEAXIS[:], DATA_E[1], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[3])
ax20.plot(ENEAXIS[:], DATA_E[2], linewidth=1.5, linestyle=linestyles[2],
           color=colors[2], label=labels[6])

ax20.legend(fontsize=11,loc='upper right',edgecolor='black',
              labelspacing=0.2, handlelength=1.0, handleheight=0.5,
              handletextpad=0.5, columnspacing=1.0, borderaxespad=0.3)

ax20.set_ylim((0,ylim/10))
ax20.set_xlim((0,200))
ax20.set_xlabel('$\hbar \omega$ (meV)')

ax20.tick_params(axis='both', direction='in')
ax20.tick_params(which='minor', direction='in')
ax20.xaxis.set_ticks_position('both')
ax20.yaxis.set_ticks_position('both')

ax20.text(x=10, y=ylim/10*0.88, s='$e$ modes')

##########################
ylim = 0.35

# fig-21
ax01.plot(ENEAXIS[:], DATA[3], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax01.plot(ENEAXIS[:], DATA[4], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[3])
ax01.plot(ENEAXIS[:], DATA[5], linewidth=1.5, linestyle=linestyles[2],
           color=colors[2], label=labels[6])

ax01.set_ylim((0,ylim/10))
ax01.set_xlim((0,200))
ax01.set_xticklabels([])

ax01.tick_params(axis='both', direction='in')
ax01.tick_params(which='minor', direction='in')
ax01.xaxis.set_ticks_position('both')
ax01.yaxis.set_ticks_position('both')

ax01.text(x=-30, y=ylim/10*1.07, s='b', fontsize=15, weight='bold')

ax01.text(x=10, y=ylim/10*0.88, s='$a_1 + e$ modes')

ax01.text(x=145, y=ylim/10*0.83, s='$^1E \\to ^1A_1$', fontsize=14)

# fig-22
ax11.plot(ENEAXIS[:], DATA_A[3], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax11.plot(ENEAXIS[:], DATA_A[4], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[3])
ax11.plot(ENEAXIS[:], DATA_A[5], linewidth=1.5, linestyle=linestyles[2],
           color=colors[2], label=labels[6])

ax11.set_ylim((0,ylim/10))
ax11.set_xlim((0,200))
ax11.set_xticklabels([])

ax11.tick_params(axis='both', direction='in')
ax11.tick_params(which='minor', direction='in')
ax11.xaxis.set_ticks_position('both')
ax11.yaxis.set_ticks_position('both')

ax11.text(x=10, y=ylim/10*0.88, s='$a_1$ modes')

# fig-23
ax21.plot(ENEAXIS[:], DATA_E[3], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax21.plot(ENEAXIS[:], DATA_E[4], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[3])
ax21.plot(ENEAXIS[:], DATA_E[5], linewidth=1.5, linestyle=linestyles[2],
           color=colors[2], label=labels[6])

ax21.set_ylim((0,ylim/10))
ax21.set_xlim((0,200))
ax21.set_xlabel('$\hbar \omega$ (meV)')

ax21.tick_params(axis='both', direction='in')
ax21.tick_params(which='minor', direction='in')
ax21.xaxis.set_ticks_position('both')
ax21.yaxis.set_ticks_position('both')

ax21.text(x=10, y=ylim/10*0.88, s='$e$ modes')

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel('S($\hbar \omega$) (meV$^{-1}$)', labelpad=10)
plt.subplots_adjust(wspace=0.3, hspace=0.05)

plt.savefig("Fig-S5.pdf", bbox_inches='tight', dpi=300)
plt.show()
