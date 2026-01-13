#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})

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

########
# Main #
########

fnames = [
         'All-B-3A2-3E-Ph-3A2.dat',
         'All-B-3A2-3E-Ph-1A1.dat',
         'All-B-1A1-1E-Ph-3A2.dat',
         'All-B-1A1-1E-Ph-1A1.dat'
         ]

bf = []
freq = []
for i in range(4):
    bf.append(np.loadtxt(fnames[i], usecols=3))
    freq.append(np.loadtxt(fnames[i], usecols=6) * 1000/8065.544)

# deg?
wd = [
      np.zeros(freq[0].shape[0]),
      np.zeros(freq[1].shape[0]),
      np.zeros(freq[2].shape[0]),
      np.zeros(freq[3].shape[0])
     ]

for i in range(4):
    for j in range(1,freq[i].shape[0]):
        if abs(freq[i][j] - freq[i][j-1]) < 1e-5:
            wd[i][j] = 1
            wd[i][j-1] = 1

bf_a = [
        np.zeros(freq[0].shape[0]),
        np.zeros(freq[1].shape[0]),
        np.zeros(freq[2].shape[0]),
        np.zeros(freq[3].shape[0])
       ]

bf_e = [
        np.zeros(freq[0].shape[0]),
        np.zeros(freq[1].shape[0]),
        np.zeros(freq[2].shape[0]),
        np.zeros(freq[3].shape[0])
       ]

for i in range(4):
    bf_e[i][:] = bf[i][:] * wd[i][:]
    bf_a[i][:] = bf[i][:] - bf_e[i][:]


DATA = np.zeros((4,1000))
DATA_A = np.zeros((4,1000))
DATA_E = np.zeros((4,1000))

ENEAXIS = np.linspace(0, 200, 1000, endpoint = True )

for i in range(1000):
    DATA[0,i] = sum(bf[0][:]**2 * Gaussian(ENEAXIS[i], freq[0][:], sigma(freq[0][:])))
    DATA[1,i] = sum(bf[1][:]**2 * Gaussian(ENEAXIS[i], freq[1][:], sigma(freq[1][:])))
    DATA[2,i] = sum(bf[2][:]**2 * Gaussian(ENEAXIS[i], freq[2][:], sigma(freq[2][:])))
    DATA[3,i] = sum(bf[3][:]**2 * Gaussian(ENEAXIS[i], freq[3][:], sigma(freq[3][:])))

    DATA_A[0,i] = sum(bf_a[0][:]**2 * Gaussian(ENEAXIS[i], freq[0][:], sigma(freq[0][:])))
    DATA_A[1,i] = sum(bf_a[1][:]**2 * Gaussian(ENEAXIS[i], freq[1][:], sigma(freq[1][:])))
    DATA_A[2,i] = sum(bf_a[2][:]**2 * Gaussian(ENEAXIS[i], freq[2][:], sigma(freq[2][:])))
    DATA_A[3,i] = sum(bf_a[3][:]**2 * Gaussian(ENEAXIS[i], freq[3][:], sigma(freq[3][:])))

    DATA_E[0,i] = sum(bf_e[0][:]**2 * Gaussian(ENEAXIS[i], freq[0][:], sigma(freq[0][:])))
    DATA_E[1,i] = sum(bf_e[1][:]**2 * Gaussian(ENEAXIS[i], freq[1][:], sigma(freq[1][:])))
    DATA_E[2,i] = sum(bf_e[2][:]**2 * Gaussian(ENEAXIS[i], freq[2][:], sigma(freq[2][:])))
    DATA_E[3,i] = sum(bf_e[3][:]**2 * Gaussian(ENEAXIS[i], freq[3][:], sigma(freq[3][:])))

########
# Plot #
########

fig, ax = plt.subplots(3, 2, figsize=(11,6))

colors = ['#DB4437', '#4285F4', '#F4B400', '#0F9D58']
linestyles = ['-', '-', '--', '--']
labels = [
         'Using $^3A_2$ phonons',
         'Using $^1A_1$ phonons'
         ]

# fig-1
ax[0][0].plot(ENEAXIS[:], DATA[0], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax[0][0].plot(ENEAXIS[:], DATA[1], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[1])
ax[0][0].legend(fontsize=11,loc='upper right',edgecolor='black')

# fig-2
ax[1][0].plot(ENEAXIS[:], DATA_A[0], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax[1][0].plot(ENEAXIS[:], DATA_A[1], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[1])

# fig-3
ax[2][0].plot(ENEAXIS[:], DATA_E[0], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax[2][0].plot(ENEAXIS[:], DATA_E[1], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[1])

# fig-4
ax[0][1].plot(ENEAXIS[:], DATA[2], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax[0][1].plot(ENEAXIS[:], DATA[3], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[1])

# fig-5
ax[1][1].plot(ENEAXIS[:], DATA_A[2], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax[1][1].plot(ENEAXIS[:], DATA_A[3], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[1])

# fig-6
ax[2][1].plot(ENEAXIS[:], DATA_E[2], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax[2][1].plot(ENEAXIS[:], DATA_E[3], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[1])

for i in range(3):
    ylim = 0.7
    ax[i][0].set_ylim((0,ylim/10))
    ax[i][0].set_xlim((0,200))
    if i != 2:
        ax[i][0].set_xticklabels([])
    ax[i][0].tick_params(axis='both', direction='in')
    ax[i][0].tick_params(which='minor', direction='in')
    ax[i][0].xaxis.set_ticks_position('both')
    ax[i][0].yaxis.set_ticks_position('both')

    ylim = 0.7
    ax[i][1].set_yticklabels([])
    ax[i][1].set_ylim((0,ylim/10))
    ax[i][1].set_xlim((0,200))
    if i != 2:
        ax[i][1].set_xticklabels([])
    ax[i][1].tick_params(axis='both', direction='in')
    ax[i][1].tick_params(which='minor', direction='in')
    ax[i][1].xaxis.set_ticks_position('both')
    ax[i][1].yaxis.set_ticks_position('both')

ax[0][0].text(x=-40, y=0.073, s='a', weight='bold', fontsize=15)
ax[0][1].text(x=-20, y=0.073, s='b', weight='bold', fontsize=15)

ax[0][0].text(x=6, y=0.05, s='$a_1 + e$')
ax[1][0].text(x=6, y=0.05, s='$a_1$')
ax[2][0].text(x=6, y=0.05, s='$e$')
ax[0][1].text(x=6, y=0.05, s='$a_1 + e$')
ax[1][1].text(x=6, y=0.05, s='$a_1$')
ax[2][1].text(x=6, y=0.05, s='$e$')

ax[i][0].set_xlabel('$\hbar \omega$ (meV)')
ax[i][1].set_xlabel('$\hbar \omega$ (meV)')

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel('S($\hbar \omega$) (meV$^{-1}$)', labelpad=10)
plt.subplots_adjust(wspace=0.15, hspace=0.1)

plt.savefig("Fig-S7.pdf", bbox_inches='tight', dpi=300)
plt.show()
