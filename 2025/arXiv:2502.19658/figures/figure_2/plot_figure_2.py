#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
plt.rcParams.update({'font.size': 10})
import json
from scipy import constants
from scipy.optimize import curve_fit

##########
# figure #
##########

fig = plt.figure(figsize=(6, 5))

gs = GridSpec(nrows=3, ncols=3, height_ratios=[1, 0.12, 0.8],
                                width_ratios=[0.8, 0.05, 1],
                                hspace=0.1, wspace=0.2,
                                left=0.02, right=0.98,
                                bottom=0.02, top=0.98)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 2])
ax2 = fig.add_subplot(gs[2, :])

ax = [ax0, ax1, ax2]

###############################
# DFT SOC finite-size effects #
###############################

cell_size = np.array([63, 215, 511, 999, 1727])
soc_matrix = np.array([
[-1, -1],
[73.690989, 198.21835985396015],
[45.662558, 124.54644670146465],
[39.538475, 106.4168231552068],
[37.797758, 101.20653019865077],
])

ref_hse = np.array([
[215, 38, 99],
[511, 21, 65],
[999, 17, 58]
])

ref_pbe = np.array([
[215, 47],
[511, 25],
[999, 21],
[1727, 20.5]
])
lambda_z = np.abs(soc_matrix[:, 0])
lambda_perp = np.abs(soc_matrix[:, 1]) / np.sqrt(2)

colors = ['#AB47BC', '#46BDC6']
labels = ['$\lambda_z^{\mathrm{DFT}}$', '$\lambda_\perp^{\mathrm{DFT}}$']

ax[0].plot(np.power((cell_size[1:] + 1) / 8, 1/3), lambda_z[1:], marker='s', markersize=6,
        linestyle='-', linewidth=0, color=colors[0], label=labels[0])
ax[0].plot(np.power((cell_size[1:] + 1) / 8, 1/3), lambda_perp[1:], marker='s', markersize=6,
        linestyle='-', linewidth=0, color=colors[1], label=labels[1])

# fitting
def func1(x, a, b, c):
    f = a * np.exp(x * b) + c
    return f

x = np.array([3.0, 4.0, 5.0, 6.0])
y = lambda_z[1:]

popt, pcov = curve_fit(func1, x, y, p0=[1, -3, 35])
p_lambda_z = popt[:]
xaxis = np.linspace(1, 7, 1001)

ax[0].axhline(y=p_lambda_z[-1], xmin=0, xmax=1, linestyle='--', c=colors[0], linewidth=1.0,
              label='$\lambda_{z,0}^{\mathrm{DFT}} = %.1f$ GHz'%(p_lambda_z[-1]))
print(popt)
print('lambda_z  %.6f'%p_lambda_z[-1])

x = np.array([3.0, 4.0, 5.0, 6.0])
y = lambda_perp[1:]
popt, pcov = curve_fit(func1, x, y, p0=[1, -3, 70])
p_lambda_perp = popt[:]
xaxis = np.linspace(1, 7, 1001)

ax[0].axhline(y=p_lambda_perp[-1], xmin=0, xmax=1, linestyle='--', c=colors[1], linewidth=1.0,
              label='$\lambda_{\perp,0}^{\mathrm{DFT}} = %.1f$ GHz'%(p_lambda_perp[-1]))
print(popt)
print('lambda_perp  %.6f'%p_lambda_perp[-1])

for i in range(3):
    if i==0:
        ax[i].legend(fontsize=10, loc='upper right', edgecolor='black',
                     borderpad=0.3, labelspacing=0.3, handlelength=1.2, handleheight=0.5,
                     handletextpad=0.5, borderaxespad=0.3, columnspacing=1.5)
        ax[i].set_xlabel('Number of Atoms')
        ax[i].set_xticks([3, 4, 5, 6])
        ax[i].set_xticklabels([215, 511, 999, 1727])
        ax[i].set_ylabel('$\lambda$ (GHz)', color='black')
        ax[i].tick_params(axis='y', colors='black')
        ax[i].tick_params(axis='both', direction='in')
        ax[i].tick_params(which='minor', direction='in')
        ax[i].xaxis.set_ticks_position('both')
        ax[i].yaxis.set_ticks_position('both')
        ax[i].set_xlim([2.5, 6.5])
        ax[i].set_ylim([0, 160])

print()
print()
print()

############
# QDET SOC #
############

act_sp = np.genfromtxt('qdet_511_soc_results.txt', usecols=0, dtype='str')
e_ks = np.loadtxt('qdet_511_soc_results.txt', usecols=(1, 2))
lambda_z = np.loadtxt('qdet_511_soc_results.txt', usecols=3)
lambda_perp = np.loadtxt('qdet_511_soc_results.txt', usecols=4) / np.sqrt(2)
vees = np.loadtxt('qdet_511_soc_results.txt', usecols=(5, 6, 7, 8))

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58', '#AB47BC',
          '#FF6F61', '#46BDC6', '#FF9E80', '#7E57C2', '#9CCC65']
labels = ['$^3A_2$', '$^1E$', '$^1A_1$', '$^3E$']
markers = ['s', 'o', '^', 'v']

xaxis = np.arange(12)

# lambda
ax[1].plot(xaxis[1:], lambda_z[1:], color=colors[4], marker='o', label='$\lambda_z^{\mathrm{QDET}}$',
           markersize=6, linestyle='-', linewidth=0.)
ax[1].plot(xaxis[1:], lambda_perp[1:], color=colors[6], marker='o', label='$\lambda_\perp^{\mathrm{QDET}}$',
           markersize=6, linestyle='-', linewidth=0.)

# band for range
ax[1].fill_between([0, 70], y1=22.898730432785417, y2=33.35775985509086, color='#E6BDE4')#colors[4], alpha=0.3)
ax[1].fill_between([0, 70], y1=49.148115995000154 / np.sqrt(2), y2=59.006625217051685 / np.sqrt(2), color='#BDE7EA')#colors[6], alpha=0.3)

ax[1].axhline(y=lambda_z[0], xmin=0, xmax=1, color=colors[4],
              linestyle='--', linewidth=1.0, label='$\lambda_z^{\mathrm{DFT}}$')
ax[1].axhline(y=lambda_perp[0], xmin=0, xmax=1, color=colors[6],
              linestyle='--', linewidth=1.0, label='$\lambda_\perp^{\mathrm{DFT}}$')


for i in range(3):
    if i==1: ax[i].legend(fontsize=10, loc='upper center', edgecolor='black', ncol=4,
                          borderpad=0.3, labelspacing=0.2, handlelength=0.8, handleheight=0.5,
                     handletextpad=0.4, borderaxespad=0.3, columnspacing=0.8)
    if i==1: ax[i].set_xlabel('Number of orbitals in active space')
    if i==1: ax[i].set_xticks(xaxis[1:])
    if i==1: ax[i].set_xticklabels([3, 5, 8, 12, 16, 24, 32, 40, 48, 56, 62])
    if i==1: ax[i].set_ylabel('$\lambda$ (GHz)', color='black')
    if i==1: ax[i].tick_params(axis='y', colors='black')
    if i==1: ax[i].tick_params(axis='both', direction='in')
    if i==1: ax[i].tick_params(which='minor', direction='in')
    if i==1: ax[i].xaxis.set_ticks_position('both')
    if i==1: ax[i].yaxis.set_ticks_position('both')
    if i==1: ax[i].set_xlim([0.5, 11.5])
    if i==1: ax[i].set_ylim([0, 105])

###########################################
# bar plot for comparison with experimens #
###########################################

lambda_data = np.array([
    [[37.5, 11.4, 70.6], [0.0, 0.0, 0.0]], # This work, PBE, dilute limit
    [[20.4, 6.20, -1], [0.0, 0.0, 0.0]], # DFT (PBE) a
    [[15.78, 4.80, 56.32], [0.0, 0.0, 0.0]], # DFT (HSE) a
    [[14.21, 4.32, 3.96], [0.0, 0.0, 0.0]], # CASSCF, C33H36N- cluster, b
    [[20.6, 6.3, 29.7 * 2], [0.0, 0.0, 0.0]], # CI-cRPA, c
    [[23.1, 7.0, 29.4], [4.3, 1.3, 2.8]], # This work, QDET, dilute limit
    [[17.53, 5.33, 21.06], [0.10, 0.03, 3.62]], # expt. d
])
# p = 0.304

names = [
    "DFT",
    "PBE\n(PAW)",
    "HSE\n(PAW)",
    "CASSCF\nC$_{33}$H$_{36}$N$^{-}$",
    "CI-cRPA",
    "QDET",
    "Expt.",
]

patterns = [None, "/" * 4, None]

width = 0.18
x = np.array([1, 2, 3, 4, 5, 6, 7])

for i in range(2, 3, 1):
    ax[i].bar(x - 1*width, lambda_data[:, 0, 0], yerr=lambda_data[:, 1, 0], capsize=4, width=width,
                   edgecolor='black', hatch=patterns[0], label='$\lambda_z$', color='#D48ACF')#colors[4], alpha=0.6)
    ax[i].bar(x - 0*width, lambda_data[:, 0, 1], yerr=lambda_data[:, 1, 1], capsize=4, width=width,
                   edgecolor='black', hatch=patterns[1], label='$p\lambda_z$', color='#D48ACF')#colors[4], alpha=0.6)
    ax[i].bar(x + 1*width, lambda_data[:, 0, 2], yerr=lambda_data[:, 1, 2], capsize=4, width=width,
                   edgecolor='black', hatch=patterns[0], label='$\lambda_\perp$', color='#80CED3')#colors[6], alpha=0.6)
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(names, rotation=0, fontsize=10)

    for label in ax[i].get_xticklabels():
        if label.get_text() in ['DFT', 'QDET']:
            label.set_fontweight('bold')

    ax[i].set_xlim([0.3, 7.7])
    ax[i].set_ylim([0, 80])
    ax[i].yaxis.set_ticks_position('both')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].legend(loc='upper right',
                ncol=3, edgecolor='black', fontsize=10,)

    ax[i].set_ylabel('$\lambda$ (GHz)', color='black')
    ax[i].tick_params(axis='both', direction='in')
    ax[i].yaxis.set_ticks_position('both')

ax[0].text(x=-0.3, y=1.0, s='(a)', fontsize=13, color='k', transform=ax[0].transAxes)
ax[1].text(x=-0.23, y=1.0, s='(b)', fontsize=13, color='k', transform=ax[1].transAxes)
ax[2].text(x=-0.11, y=1.0, s='(c)', fontsize=13, color='k', transform=ax[2].transAxes)

plt.savefig("figure_2.pdf", bbox_inches='tight', dpi=300)
plt.show()
