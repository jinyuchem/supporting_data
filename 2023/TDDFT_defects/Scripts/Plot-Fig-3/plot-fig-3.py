#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 11})
import json

########
# Fxns #
########

def read_results(fileName):
    """
    Read eigenvalues from JSON file.
    """

    with open(fileName, "r") as f:
        raw_ = json.load(f)

    allthr = list(raw_.keys())
    alleigs = []
    allitertime = []
    allhpsitime = []
    allforces = []

    for key in list(raw_.keys()):
        alleigs.append(raw_[key]["eigs"])
        allitertime.append(raw_[key]["wbse_time"])
        allhpsitime.append(raw_[key]["hyb2_time"])

        tmp_forces = []
        for key2 in list(raw_[key]["forces"].keys()):
            tmp_forces.append(raw_[key]["forces"][key2])
        allforces.append(tmp_forces)

    alleigs = np.array(alleigs)

    return allthr, alleigs, allitertime, allhpsitime, allforces

# Note: forces are forces_drhox1, forces_drhox2, forces_drhoz, forces_total, forces_collect

########
# Main #
########

fname = 'results.json'
allthr, alleigs, allitertime, allhpsitime, allforces = read_results(fname)

##########################
# Calculate forces error #
##########################

ref_forces = allforces[0]

# ovlp: type
me = []
mse = []
mae = []

for i in range(len(allthr)):
    tme = []
    tmse = []
    tmae = []
    error1 = np.array(allforces[i][0]) - np.array(ref_forces[0])
    error2 = np.array(allforces[i][1]) - np.array(ref_forces[1])
    error = error1 + error2
    ttme = np.max(abs(error))
    ttmse = np.linalg.norm(error) / np.sqrt(error.shape[0])
    ttmae = sum(abs(error)) / error.shape[0]

    tme.append(ttme)
    tmse.append(ttmse)
    tmae.append(ttmae)

    for j in range(2, 5):
        error = np.array(allforces[i][j]) - np.array(ref_forces[j])
        ttme = np.max(abs(error))
        ttmse = np.linalg.norm(error) / np.sqrt(error.shape[0])
        ttmae = sum(abs(error)) / error.shape[0]

        tme.append(ttme)
        tmse.append(ttmse)
        tmae.append(ttmae)

    me.append(tme)
    mse.append(tmse)
    mae.append(tmae)

me = np.array(me) * 13.605662285137 / 0.529177249
mse = np.array(mse) * 13.605662285137 / 0.529177249
mae = np.array(mae) * 13.605662285137 / 0.529177249

########
# Plot #
########

allthr = [6, 5, 4, 3, 2, 1, 0]

fig, ax = plt.subplots(1, 3, figsize=(11.5,2.6))

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
linestyles = ['', '', '-', ':', '-', ':']
labels1 = ['', '']

# nv forces error
ax[0].plot(allthr, 1e3 * me[1:,3], c=colors[0], marker='s',
              lw=0, markersize=5, label='Max Abs Error')
ax[0].plot(allthr, 1e3 * mse[1:,3], c=colors[1], marker='o',
              lw=0, markersize=5, label='MSE')
ax[0].plot(allthr, 1e3 * mae[1:,3], c=colors[2], marker='^',
              lw=0, markersize=5, label='MAE')
# nv total wall time
ax[2].plot(allthr, 100 * np.array(allitertime)[1:]/allitertime[0], c=colors[-1], marker='s',
              lw=0, markersize=5,)

# nv k1d time
ax[1].plot(allthr, 100 * np.array(allhpsitime)[1:]/allhpsitime[0], c=colors[-1], marker='s',
              lw=0, markersize=5,)

ax[0].text(x=-0.3, y=1.05, s='(a)', fontsize=15,
              transform=ax[0].transAxes)
ax[1].text(x=-0.3, y=1.05, s='(b)', fontsize=15,
              transform=ax[1].transAxes)
ax[2].text(x=-0.3, y=1.05, s='(c)', fontsize=15,
              transform=ax[2].transAxes)

for i in range(1):
    for j in range(3):
        #ax[j].set_xscale("log")

        if j==0: ax[j].legend(fontsize=11,loc='best',edgecolor='black')

        ax[j].set_xlabel('$\lambda_\mathrm{thr}$')
        if j==0: ax[j].set_ylabel('Error in Forces (meV Å$^{-1}$)')
        if j==2: ax[j].set_ylabel('Total wall time (%)')
        if j==1: ax[j].set_ylabel('$\mathcal{K}^{2d}$ wall time (%)')

        ax[j].spines['left'].set_color('black')
        ax[j].spines['right'].set_color('black')
        ax[j].tick_params(axis='y', colors='black')
        ax[j].tick_params(axis='both', direction='in')
        ax[j].tick_params(which='minor', direction='in')
        ax[j].xaxis.set_ticks_position('both')
        ax[j].yaxis.set_ticks_position('both')

   
        ax[j].set_xlim([-0.5, 6.5])
        ax[j].set_xticks([0, 1, 2, 3, 4, 5, 6])
        ax[j].set_xticklabels(['$10^{-2}$', '$10^{-3}$', '$10^{-4}$',
                               '$10^{-5}$', '$10^{-6}$', '$10^{-7}$', '$10^{-8}$'],
              rotation=45)

        #ax[j].set_ylim([3.5, 6.5])

        if j==2: ax[j].set_ylim([0,100])
        if j==1: ax[j].set_ylim([0,100])

fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.4, hspace=0.2)

plt.savefig("Fig-3.pdf", bbox_inches='tight', dpi=300)
plt.show()
