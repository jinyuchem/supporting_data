#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
plt.rcParams.update({'font.size': 11})
import json
from scipy import constants
from scipy.optimize import curve_fit

#######
# fxn #
#######

def isc_rate(lamda, f):
    """
    lamda: GHz
    f: eV^{-1}
    rate: MHz
    """
    prefactor = 77.04406985667521 / 80.06910547505998
    rate = (
        (2 * np.pi / constants.hbar)
        * 2
        * (lamda * 1e9 * constants.h) ** 2
        * (f / constants.eV)
        / 1e6
    )
    rate *= prefactor**2
    return rate

##########
# figure #
##########

fig = plt.figure(figsize=(5.5, 5))

gs = GridSpec(nrows=2, ncols=1, height_ratios=[0.6, 1],
                                width_ratios=[1],
                                hspace=0.1, wspace=0.2,
                                left=0.02, right=0.98,
                                bottom=0.02, top=0.98)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])

ax = [ax0, ax1]

##########################
# plot spectral function #
##########################

blue = "#4285F4"
red = "#DB4437"
yellow = "#F4B400"
green = "#0F9D58"
purple = "#AB47BC"


a1_lsp = np.loadtxt('final_a1_lsp.dat', usecols=(0, 1)).T
e12_lsp = np.loadtxt('final_e12_lsp.dat', usecols=(0, 1)).T

lambda_perp = 29.4
lambda_perp_minus = 29.4 - 2.8
lambda_perp_plus = 29.4 + 2.8

e12_rate = isc_rate(lambda_perp, e12_lsp[1])
e12_uncertainty = e12_rate * 2.8 * 2 / 29.4
e12_rate_minus = e12_rate - e12_uncertainty
e12_rate_plus = e12_rate + e12_uncertainty


dft_lambda_perp = 70.6
dft_e12_rate = isc_rate(dft_lambda_perp, e12_lsp[1])

ax[0].plot(e12_lsp[0], dft_e12_rate, color='k', linewidth=1, linestyle=':', label='Using $\lambda_{\perp,0}^{\mathrm{DFT}}$')

ax[0].plot(e12_lsp[0], e12_rate, color="#D99A00", linewidth=1.5, label='Using $\lambda_{\perp,0}^{\mathrm{QDET}}$')
ax[0].fill_between(a1_lsp[0], e12_rate_minus, e12_rate_plus, color="#F9D98A")
ax[0].plot(e12_lsp[0], 52.2 * np.ones(e12_lsp.shape[1]), color='gray', linewidth=0.5)
ax[0].fill_between(e12_lsp[0], y1=52.2 - 3.8 * (52.2 / 100.5), y2=52.2 + 3.8 * (52.2 / 100.5), color='lightgray', label="Expt. rate")


# left
tmp_i = np.argmin(abs(e12_rate_minus[3000:] - (52.2 + 3.8 * (52.2 / 100.5)))) + 3000
print('\Delta_{-}    % .5f    % 10.5f    % 10.5f'%(e12_lsp[0, tmp_i], e12_rate_minus[tmp_i], 52.2 + 3.8 * (52.2 / 100.5)))
ax[0].axvline(x=e12_lsp[0, tmp_i], ymin=0, ymax=1, linestyle='-', linewidth=0.5, color='gray')
ax[0].text(x=e12_lsp[0, tmp_i], y=400 * 0.3, s='$\Delta_{-}=%.3f$ eV'%(e12_lsp[0, tmp_i]), color='gray', ha='left', fontsize=10)
# right
tmp_i = np.argmin(abs(e12_rate_plus - (52.2 - 3.8 * (52.2 / 100.5))))
print('\Delta_{+}    % .5f    % 10.5f    % 10.5f'%(a1_lsp[0, tmp_i], e12_rate_plus[tmp_i], 52.2 - 3.8 * (52.2 / 100.5)))
ax[0].axvline(x=e12_lsp[0, tmp_i], ymin=0, ymax=1, linestyle='-', linewidth=0.5, color='gray')
ax[0].text(x=e12_lsp[0, tmp_i], y=230 * 0.3, s='$\Delta_{+}=%.3f$ eV'%(e12_lsp[0, tmp_i]), color='gray', ha='left', fontsize=10)

######################
# literature results #
######################


names = [
    "QDET",
    "TDDFT (DDH)",
    "NEVPT2-DMET",
    "CI-cRPA",
    "CASSCF (C$_{85}$H$_{76}$N$^{-}$)",
    "CASSCF (C$_{33}$H$_{36}$N$^{-}$)",
    "CASPT2 (C$_{33}$H$_{36}$N$^{-}$)",
    "ppRPA (PBE)",
]

vee = np.array([
[2.008, 1.475],
[2.372, 1.973],
[2.31, 1.56],
[2.02, 1.41],
[2.14, 1.60],
[2.30, 1.96],
[2.22, 1.57],
[1.95, 1.67],
])

old_colors = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#ffd700",  # Gold
    "#bcbd22"   # Olive
]
colors = [
    "#4285F4",  # Google Blue
    "#F4B400",  # Google Yellow
    "#0F9D58",  # Google Green
    "#AB47BC",  # Purple
    "#00ACC1",  # Cyan
    "#FF7043",  # Orange
    "#8D6E63",  # Brown
    "#5C6BC0",  # Indigo
    "#F06292",  # Pink
    "#26A69A",  # Teal
    "#FFD54F",  # Amber
    "#BDBDBD",  # Light Gray
    "#90CAF9"   # Light Blue
]

colors_alpha_0_5 = [
    "#85B8F9",  # Google Blue (alpha=0.5)
    "#F9D98A",  # Google Yellow (alpha=0.5)
    "#80CCA3",  # Google Green (alpha=0.5)
    "#D891D5",  # Purple (alpha=0.5)
    "#80D5E1",  # Cyan (alpha=0.5)
    "#FFB7A0",  # Orange (alpha=0.5)
    "#C4A39C",  # Brown (alpha=0.5)
    "#A4B0E0",  # Indigo (alpha=0.5)
    "#F7A6BB",  # Pink (alpha=0.5)
    "#80CDB8",  # Teal (alpha=0.5)
    "#FFEBA7",  # Amber (alpha=0.5)
    "#DEDEDE",  # Light Gray (alpha=0.5)
    "#C9E3F9"   # Light Blue (alpha=0.5)
]

new_colors = [
    "#A04294",  # Vibrant Purple
    "#4D2D82",  # Deep Violet
    "#E56278",  # Soft Coral
    "#037FB3",  # Ocean Blue
    "#D6A329",  # Goldenrod Yellow
    "#F05A2D",  # Fiery Orange
    "#02A650",  # Emerald Green
    "#BB2131",  # Crimson Red
    "#2A388F",  # Royal Blue
    "#632D24",  # Mahogany
]

new_colors_alpha_0_5 = [
    "#D391C2",  # Vibrant Purple
    "#9D80B8",  # Deep Violet
    "#F3B1BA",  # Soft Coral
    "#81BFD6",  # Ocean Blue
    "#EBD68F",  # Goldenrod Yellow
    "#F8AD91",  # Fiery Orange
    "#81D2A7",  # Emerald Green
    "#DD9198",  # Crimson Red
    "#959CBE",  # Royal Blue
    "#B19A93",  # Mahogany
]

# for delta_00, let's just use the DDH values
delta_vee = vee[:, 0] - vee[:, 1]
delta_00 = delta_vee + (- 0.256 + 0.016)
print("All delta")
print(delta_00)
print()

# for each delta, compute the rate
# gamma, gamma_-, gamma_+
collect_e12_rate = np.zeros((delta_00.shape[0], 3))
for i in range(delta_00.shape[0]):
    for j in range(a1_lsp.shape[1]):
        if abs(delta_00[i] - a1_lsp[0, j]) < 1e-4:
            collect_e12_rate[i, 0] = isc_rate(lambda_perp, e12_lsp[1, j])
            tmp_e12_rate_uncertainty = collect_e12_rate[i, 0] * 2.8 * 2 / 29.4
            collect_e12_rate[i, 1] = collect_e12_rate[i, 0] - tmp_e12_rate_uncertainty
            collect_e12_rate[i, 2] = collect_e12_rate[i, 0] + tmp_e12_rate_uncertainty
print("ISC rates")
print(collect_e12_rate)
print()

indices = np.argsort(delta_00)

sigma = 0.01
for i, ind in enumerate(indices):
    ax[1].hlines(y=collect_e12_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.0)
    ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_e12_rate[ind, 1], y2=collect_e12_rate[ind, 2], color=new_colors_alpha_0_5[i], label=names[ind])


ax[1].plot(a1_lsp[0], 52.2 * np.ones(a1_lsp.shape[1]), color='gray', linewidth=0.5, zorder=-1)
ax[1].fill_between(a1_lsp[0], y1=52.2 - 3.8 * (52.2 / 100.5), y2=52.2 + 3.8 * (52.2 / 100.5), color='lightgray', zorder=-2)#, label="Expt.")

# left
tmp_i = np.argmin(abs(e12_rate_minus[3000:] - (52.2 + 3.8 * (52.2 / 100.5)))) + 3000
ax[1].axvline(x=e12_lsp[0, tmp_i], ymin=0, ymax=1, linestyle='-', linewidth=0.5, color='gray')
# right
tmp_i = np.argmin(abs(e12_rate_plus - (52.2 - 3.8 * (52.2 / 100.5))))
ax[1].axvline(x=e12_lsp[0, tmp_i], ymin=0, ymax=1, linestyle='-', linewidth=0.5, color='gray')



option = 2

if option == 0: # old version, energy acsending
    indices = np.argsort(delta_00[:])
    sigma = 0.01
    for i, ind in enumerate(indices):
        ax[1].hlines(y=collect_e12_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.0)
        ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_e12_rate[ind, 1], y2=collect_e12_rate[ind, 2], color=new_colors_alpha_0_5[i], label=names[ind])
elif option == 1: # QDET first
    indices = np.argsort(delta_00[1:]) + 1
    ind = 0
    sigma = 0.01
    ax[1].hlines(y=collect_e12_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[0], linewidth=1.0)
    ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_e12_rate[ind, 1], y2=collect_e12_rate[ind, 2], color=new_colors_alpha_0_5[i], label=names[ind])
    for i, ind in enumerate(indices):
        ax[1].hlines(y=collect_e12_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i+1], linewidth=1.0)
        ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_e12_rate[ind, 1], y2=collect_e12_rate[ind, 2], color=new_colors_alpha_0_5[i+1], label=names[ind])
elif option == 2: # label each block manually
    names2 = [
        "QDET",
        "TDDFT\n(DDH)",
        "DMET\nNEVPT2",
        "CI-cRPA",
        "CASSCF\n(C$_{85}$H$_{76}$N$^{-}$)",
        "CASSCF\n(C$_{33}$H$_{36}$N$^{-}$)",
        "CASPT2\n(C$_{33}$H$_{36}$N$^{-}$)",
        "ppRPA\n(PBE)"
    ]

    indices = np.argsort(delta_00[:])
    sigma = 0.01
    for i, ind in enumerate(indices):
        if i in [0, 1, 2, 5]:
            ax[1].hlines(y=collect_e12_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.5, zorder=1)
            ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_e12_rate[ind, 1], y2=collect_e12_rate[ind, 2], color=new_colors_alpha_0_5[i], zorder=0)
            ax[1].text(x=delta_00[ind], y=collect_e12_rate[ind, 2] + 2, s=names2[ind], color=new_colors[i], fontsize=10, ha='center')
        elif i in [3]:
            ax[1].hlines(y=collect_e12_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.5, zorder=3)
            ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_e12_rate[ind, 1], y2=collect_e12_rate[ind, 2], color=new_colors_alpha_0_5[i], zorder=2)
            ax[1].text(x=delta_00[ind] + 0.0, y=collect_e12_rate[ind, 2] + 4, s=names2[ind], color=new_colors[i], fontsize=10, ha='center', fontweight='bold')
        elif i in [6]:
            ax[1].hlines(y=collect_e12_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.5, zorder=1)
            ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_e12_rate[ind, 1], y2=collect_e12_rate[ind, 2], color=new_colors_alpha_0_5[i], zorder=0)
            ax[1].text(x=delta_00[ind] + 0.055, y=collect_e12_rate[ind, 2] + -4, s=names2[ind], color=new_colors[i], fontsize=10, ha='center')
        elif i in [4]:
            ax[1].hlines(y=collect_e12_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.5, zorder=1)
            ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_e12_rate[ind, 1], y2=collect_e12_rate[ind, 2], color=new_colors_alpha_0_5[i], zorder=0)
            ax[1].text(x=delta_00[ind] + 0.06, y=collect_e12_rate[ind, 1] + 10, s=names2[ind], color=new_colors[i], fontsize=10, ha='center')
        elif i in [7]:
            ax[1].hlines(y=collect_e12_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.5, zorder=1)
            ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_e12_rate[ind, 1], y2=collect_e12_rate[ind, 2], color=new_colors_alpha_0_5[i], zorder=0)
            ax[1].text(x=delta_00[ind], y=collect_e12_rate[ind, 2] + 2, s=names2[ind], color=new_colors[i], fontsize=10, ha='center')


ax[0].text(x=-0.13, y=1.0, s='(a)', fontsize=14, color='k', transform=ax[0].transAxes)
ax[1].text(x=-0.13, y=1.0, s='(b)', fontsize=14, color='k', transform=ax[1].transAxes)


for i in range(2):

    ax[i].set_xlim((-0.05, 0.56))
    if i==0: ax[i].set_ylim((0, 320))
    if i==1: ax[i].set_ylim((0, 150))
    ax[i].tick_params(direction='in')
    if i==0:
        ax[i].legend(fontsize=10,loc='upper right',edgecolor='black')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].tick_params(axis='both', direction='in')
    ax[i].yaxis.set_ticks_position('both')
    if i==1: ax[i].set_xlabel("$\Delta$ (eV)")
    if i==0 or i==1: ax[i].set_ylabel('$\Gamma_{E_{1,2}}$ (MHz)')
    if i==0: ax[i].set_xticklabels([])


plt.savefig("figure_s10.pdf", bbox_inches='tight', dpi=300)
plt.show()


#########
# rates #
#########

for i in range(5):
    print()


names = [
    "TDDFT (PBE)",
    "TDDFT (DDH)",
    "QDET (EDC2022)",
    "QDET (EDC2024)",
    "QDET (HFDC)",
    "NEVPT2-DMET",
    "CI-cRPA",
    "CASSCF (C$_{85}$H$_{76}$N$^{-}$)",
    "CASSCF (C$_{33}$H$_{36}$N$^{-}$)",
    "CASPT2 (C$_{33}$H$_{36}$N$^{-}$)",
    "ppRPA (PBE)",
    "GW-BSE",
]

vee = np.array([
[2.089, 1.336],
[2.372, 1.973],
[2.162, 1.317],
[2.008, 1.475],
[1.921, 1.376],
[2.31, 1.56],
[2.02, 1.41],
[2.14, 1.60],
[2.30, 1.96],
[2.22, 1.57],
[1.95, 1.67],
[2.379, 1.169],
])


# for delta_00, let's just use the DDH values
delta_vee = vee[:, 0] - vee[:, 1]
delta_00 = delta_vee + (- 0.256 + 0.016)
print("All delta")
print(delta_00)
print()

# for each delta, compute the rate
for i in range(delta_00.shape[0]):
    for j in range(e12_lsp.shape[1]):
        if abs(delta_00[i] - e12_lsp[0, j]) < 1e-5:
            e12_rate = isc_rate(lambda_perp, e12_lsp[1, j])
            tmp_e12_rate_uncertainty = e12_rate * 2.8 * 2 / 29.4
            print(names[i], '%12.6f    %12.6f'%(e12_rate, tmp_e12_rate_uncertainty))
