#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})
import json
from scipy import constants

fig, ax = plt.subplots(2, 2, figsize=(12, 7))

###############
# 3A2 -> 3Ehh #
###############

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58', '#AB47BC',
          '#FF6F61', '#46BDC6', '#FF9E80', '#7E57C2', '#9CCC65']
labels = ['$^3A_2$', '$^1E$', '$^1A_1$', '$^3E$']
markers = ['s', 'o', '^', 'v']

act_sp = np.genfromtxt('path_1_1a1_a1_results.txt', usecols=0, dtype='str')
e_ks = np.loadtxt('path_1_1a1_a1_results.txt', usecols=(1, 2))
lambda_z = np.loadtxt('path_1_1a1_a1_results.txt', usecols=3)
lambda_perp = np.loadtxt('path_1_1a1_a1_results.txt', usecols=4) / np.sqrt(2)
vees = np.loadtxt('path_1_1a1_a1_results.txt', usecols=(5, 6, 7, 8))

xaxis = np.linspace(-0.2, 1.2, 15) * 0.476630

print(lambda_perp[12] / lambda_perp[2])
print(lambda_perp[2] / lambda_perp[12])

# vee
for i in range(vees.shape[1]):
    ax[0][0].plot(xaxis[:], vees[:, i], color=colors[i], label=labels[i], marker=markers[i],
               markersize=6, linestyle='-', linewidth=0.)

# lambda
ax[1][0].plot(xaxis[:], lambda_z[:], color=colors[4], marker='s', label='$\lambda_z$',
           markersize=6, linestyle='-', linewidth=0.)
ax[1][0].plot(xaxis[:], lambda_perp[:], color=colors[6], marker='o', label='$\lambda_\perp$',
           markersize=6, linestyle='-', linewidth=0.)

coefficients = np.polyfit(xaxis, lambda_z, 1)
slope, intercept = coefficients
tx = np.linspace(-0.3, 1.0, 100)
ty = tx * slope + intercept
ax[1][0].plot(tx, ty, color=colors[4], linestyle='--', linewidth=0.5)


coefficients = np.polyfit(xaxis, lambda_perp, 1)
slope, intercept = coefficients
tx = np.linspace(-0.3, 1.0, 100)
ty = tx * slope + intercept
ax[1][0].plot(tx, ty, color=colors[6], linestyle='--', linewidth=0.5)

print(lambda_perp[12] / lambda_perp[2])

for i in range(2):
    ax[i][0].axvline(x=xaxis[2], ymin=0, ymax=1, color='gray', linewidth=1.0, linestyle='--')
    ax[i][0].axvline(x=xaxis[12], ymin=0, ymax=1, color='gray', linewidth=1.0, linestyle='--')

for i in range(2):
    ax[i][0].legend(fontsize=12, loc='upper right', edgecolor='black', ncol=2)

    ax[i][0].set_xlabel('$Q$ (amu$^{0.5}$ Å)')
    if i==0: ax[i][0].set_ylabel('VEE (eV)', color='black')
    if i==1: ax[i][0].set_ylabel('$\lambda$ (GHz)', color='black')
    ax[i][0].tick_params(axis='y', colors='black')
    ax[i][0].tick_params(axis='both', direction='in')
    ax[i][0].tick_params(which='minor', direction='in')
    ax[i][0].xaxis.set_ticks_position('both')
    ax[i][0].yaxis.set_ticks_position('both')
    ax[i][0].set_xlim([-0.2, 0.7])
    if i==0: ax[i][0].set_ylim([-0.2, 3.2])
    if i==1: ax[i][0].set_ylim([30, 70])

################
# 3Ehh -> 3E10 #
################

colors = ['#4285F4', '#DB4437', '#DB4437', '#F4B400', '#0F9D58', '#0F9D58', '#AB47BC',
          '#FF6F61', '#46BDC6', '#FF9E80', '#7E57C2', '#9CCC65']
labels = ['$^3A_2$', '$^1E_{\mathrm{low}}$', '$^1E_{\mathrm{high}}$', '$^1A_1$', '$^3E_{\mathrm{low}}$', '$^3E_{\mathrm{high}}$']
markers = ['s', 'o', 'o', '^', 'v', 'v']

act_sp = np.genfromtxt('path_2_a1_a1_e_results.txt', usecols=0, dtype='str')
e_ks = np.loadtxt('path_2_a1_a1_e_results.txt', usecols=(1, 2, 3))
lambda_z = np.loadtxt('path_2_a1_a1_e_results.txt', usecols=4)
lambda_perp = np.loadtxt('path_2_a1_a1_e_results.txt', usecols=5) / np.sqrt(2)
vees = np.loadtxt('path_2_a1_a1_e_results.txt', usecols=(6, 7, 8, 9, 10, 11))


xaxis = np.linspace(-0.2, 1.2, 8) * 0.280546

# vee
ax[0][1].plot(xaxis[:], vees[:, 3], color=colors[3], label=labels[3], marker=markers[3],
        markersize=6, linestyle='-', linewidth=0.)
for i in range(vees.shape[1]):
    if i==3: continue
    if i in [2, 5]:
        ax[0][1].plot(xaxis[:], vees[:, i], color=colors[i], label=labels[i], marker=markers[i],
               markersize=6, linestyle='-', linewidth=0., markerfacecolor='None')
    else:
        ax[0][1].plot(xaxis[:], vees[:, i], color=colors[i], label=labels[i], marker=markers[i],
               markersize=6, linestyle='-', linewidth=0.)

# lambda
ax[1][1].plot(xaxis[:], lambda_z[:], color=colors[6], marker='s', label='$\lambda_z$',
           markersize=6, linestyle='-', linewidth=0.)
ax[1][1].plot(xaxis[:], lambda_perp[:], color=colors[8], marker='o', label='$\lambda_\perp$',
           markersize=6, linestyle='-', linewidth=0.)

coefficients = np.polyfit(xaxis, lambda_z, 1)
slope, intercept = coefficients
tx = np.linspace(-0.3, 1.0, 100)
ty = tx * slope + intercept
ax[1][1].plot(tx, ty, color=colors[6], linestyle='--', linewidth=0.5)


coefficients = np.polyfit(xaxis, lambda_perp, 1)
slope, intercept = coefficients
tx = np.linspace(-0.3, 1.0, 100)
ty = tx * slope + intercept
ax[1][1].plot(tx, ty, color=colors[8], linestyle='--', linewidth=0.5)

for i in range(2):
    ax[i][1].axvline(x=xaxis[1], ymin=0, ymax=1, color='gray', linewidth=1.0, linestyle='--')
    ax[i][1].axvline(x=xaxis[6], ymin=0, ymax=1, color='gray', linewidth=1.0, linestyle='--')

for i in range(2):
    if i==1:
        ax[i][1].legend(fontsize=12, loc='upper right', edgecolor='black', ncol=2)
    elif i==0:
        ax[i][1].legend(fontsize=12, loc='upper right', edgecolor='black', ncol=3)

    ax[i][1].set_xlabel('$Q$ (amu$^{0.5}$ Å)')
    if i==0: ax[i][1].set_ylabel('VEE (eV)', color='black')
    if i==1: ax[i][1].set_ylabel('$\lambda$ (GHz)', color='black')
    ax[i][1].tick_params(axis='y', colors='black')
    ax[i][1].tick_params(axis='both', direction='in')
    ax[i][1].tick_params(which='minor', direction='in')
    ax[i][1].xaxis.set_ticks_position('both')
    ax[i][1].yaxis.set_ticks_position('both')
    ax[i][1].set_xlim([-0.1, 0.4])
    if i==0: ax[i][1].set_ylim([-0.2, 3.2])
    if i==1: ax[i][1].set_ylim([30, 70])


ss = ["(a)", "(b)", "(c)", "(d)"]
for i in range(2):
    for j in range(2):
        ax[i][j].text(x=-0.2, y=1.02, s=ss[2 * i + j], color='k', fontsize=14, transform=ax[i][j].transAxes)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.savefig("figure_s7.pdf", bbox_inches='tight', dpi=300)
plt.show()
