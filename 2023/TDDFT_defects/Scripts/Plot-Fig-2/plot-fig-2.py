#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})
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

    allovlp = list(raw_.keys())
    alleigs = []
    allitertime = []
    allkerneltime = []
    allnpairs = []
    allforces = []

    for key in list(raw_.keys()):
        alleigs.append(raw_[key]["eigs"])
        allitertime.append(raw_[key]["wbse_time"])
        allkerneltime.append(raw_[key]["kernel_time"])
        allnpairs.append(raw_[key]["n_pairs"])

        tmp_forces = []
        for key2 in list(raw_[key]["forces"].keys()):
            tmp_forces.append(raw_[key]["forces"][key2])
        allforces.append(tmp_forces)

    alleigs = np.array(alleigs)

    allovlp = [float(i) for i in allovlp]
    return allovlp, alleigs, allitertime, allkerneltime, allnpairs, allforces

# Note: forces are forces_drhox1, forces_drhox2, forces_drhoz, forces_total, forces_collect

########
# Main #
########

fname = 'results.json'
allovlp, alleigs, allitertime, allkerneltime, allnpairs, allforces = read_results(fname)

##########################
# Calculate forces error #
##########################

ref_forces = allforces[0]

# ovlp: type
me = []
mse = []
mae = []

for i in range(len(allovlp)):
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

allovlp = [3, 2, 1, 0]

fig, ax = plt.subplots(2, 2, figsize=(10,7))

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
linestyles = ['', '', '-', ':', '-', ':']
labels1 = ['', '']

# nv npairs forces
#ax[0].axhline(y=433 / 2, c=colors[0], marker='s',
#              lw=1, linestyle='--', markersize=0,
#              label='No localization')
#ax[0].plot(allovlp, np.array(allnpairs) / 432, c=colors[0], marker='s',
#              lw=0, markersize=5, label='')

# nv vee error
error_vee = alleigs[:,0] - alleigs[0,0]
error_vee *= 13.605662285137

ax[0][0].plot(allovlp, 1000*error_vee[1:], c=colors[0], marker='s',
              lw=0, markersize=5,)
# nv forces error
ax[0][1].plot(allovlp, 1000*me[1:,3], c=colors[0], marker='s',
              lw=0, markersize=5, label='Max Abs Error')
ax[0][1].plot(allovlp, 1000*mse[1:,3], c=colors[1], marker='o',
              lw=0, markersize=5, label='MSE')
ax[0][1].plot(allovlp, 1000*mae[1:,3], c=colors[2], marker='^',
              lw=0, markersize=5, label='MAE')
# nv total wall time
ax[1][1].plot(allovlp, 100*np.array(allitertime)[1:]/allitertime[0], c=colors[-1], marker='s',
              lw=0, markersize=5,)

# nv k1d time
ax[1][0].plot(allovlp, 100*np.array(allkerneltime)[1:]/allkerneltime[0], c=colors[-1], marker='s',
              lw=0, markersize=5,)

ax[0][0].text(x=-0.3, y=1.02, s='(a)', fontsize=15,
              transform=ax[0][0].transAxes)
ax[0][1].text(x=-0.3, y=1.02, s='(b)', fontsize=15,
              transform=ax[0][1].transAxes)
ax[1][0].text(x=-0.3, y=1.02, s='(c)', fontsize=15,
              transform=ax[1][0].transAxes)
ax[1][1].text(x=-0.3, y=1.02, s='(d)', fontsize=15,
              transform=ax[1][1].transAxes)

for i in range(2):
    for j in range(2):
        #ax[j].set_xscale("log")

        if i==0 and j==1: ax[i][j].legend(fontsize=11,loc='best',edgecolor='black')

        if i==1: ax[i][j].set_xlabel('$S_{\mathrm{thr}}$')
        #if j==0: ax[j].set_ylabel('$N_{\mathrm{pairs}} / N_{\mathrm{occ}}$')
        if i==0 and j==0: ax[i][j].set_ylabel('Error in VEE (meV)')
        if i==0 and j==1: ax[i][j].set_ylabel('Error in Forces (meV Å$^{-1}$)')
        if i==1 and j==1: ax[i][j].set_ylabel('Total wall time (%)')
        if i==1 and j==0: ax[i][j].set_ylabel('$\mathcal{K}^{1d}$ wall time (%)')

        ax[i][j].spines['left'].set_color('black')
        ax[i][j].spines['right'].set_color('black')
        ax[i][j].tick_params(axis='y', colors='black')
        ax[i][j].tick_params(axis='both', direction='in')
        ax[i][j].tick_params(which='minor', direction='in')
        ax[i][j].xaxis.set_ticks_position('both')
        ax[i][j].yaxis.set_ticks_position('both')

        #ax[i][j].set_xticks(list(vo_allovlp[1:]))
   
        ax[i][j].set_xlim([-0.5, 3.5])
        ax[i][j].set_xticks([0, 1, 2, 3])
        if i==1: ax[i][j].set_xticklabels(['$10^{-1}$', '$10^{-2}$', '$10^{-3}$', '$10^{-4}$'])
        if i==0: ax[i][j].set_xticklabels([])
        #ax[j].set_ylim([3.5, 6.5])
        
        if i==1 and j==1: ax[i][j].set_ylim([0,100])
        if i==1 and j==0: ax[i][j].set_ylim([0,20])

fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.4, hspace=0.2)

plt.savefig("Fig-2.pdf", bbox_inches='tight', dpi=300)
plt.show()
