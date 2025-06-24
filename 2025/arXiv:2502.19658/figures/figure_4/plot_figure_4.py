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

fig = plt.figure(figsize=(5.5, 6.5))

gs = GridSpec(nrows=4, ncols=1, height_ratios=[0.5, 0.9, 0.1, 0.6],
                                width_ratios=[1],
                                hspace=0.07, wspace=0.2,
                                left=0.02, right=0.98,
                                bottom=0.02, top=0.98)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[3])

ax = [ax0, ax1, ax2]

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

########
# test #
########

print()
for i in range(a1_lsp[0].shape[0]):
    if abs(a1_lsp[0, i] - 0.41) < 1e-7:
        print(a1_lsp[1, i])
        print(e12_lsp[1, i])
print()


lambda_perp = 29.4
lambda_perp_minus = 29.4 - 2.8
lambda_perp_plus = 29.4 + 2.8

a1_rate = isc_rate(lambda_perp, a1_lsp[1])
a1_uncertainty = a1_rate * 2.8 * 2 / 29.4
a1_rate_minus = a1_rate - a1_uncertainty
a1_rate_plus = a1_rate + a1_uncertainty


dft_lambda_perp = 70.6
dft_a1_rate = isc_rate(dft_lambda_perp, a1_lsp[1])


ax[0].plot(a1_lsp[0], a1_rate, color=red, linewidth=1, label='Using $\lambda_{\perp,0}^{\mathrm{QDET}}$')
ax[0].plot(a1_lsp[0], dft_a1_rate, color='k', linewidth=1, linestyle=':', label='Using $\lambda_{\perp,0}^{\mathrm{DFT}}$')


ax[0].fill_between(a1_lsp[0], a1_rate_minus, a1_rate_plus, color="#F7C4BF")#red, alpha=0.3, label="")
ax[0].plot(a1_lsp[0], 100.5 * np.ones(a1_lsp.shape[1]), color='gray', linewidth=0.5)
ax[0].fill_between(a1_lsp[0], y1=100.5 - 3.8, y2=100.5 + 3.8, color='lightgray', label="Expt. rate")


# left
tmp_i = np.argmin(abs(a1_rate_minus[3000:] - (100.5 + 3.8))) + 3000
print('\Delta_{-}    % .5f    % 10.5f    % 10.5f'%(a1_lsp[0, tmp_i], a1_rate_minus[tmp_i], 100.5 + 3.8))
ax[0].axvline(x=a1_lsp[0, tmp_i], ymin=0, ymax=1, linestyle='-', linewidth=0.5, color='gray')
ax[0].text(x=a1_lsp[0, tmp_i], y=220, s='$\Delta_{-}=%.3f$ eV'%(a1_lsp[0, tmp_i]), color='gray', ha='left', fontsize=10)
# right
tmp_i = np.argmin(abs(a1_rate_plus[3000:] - (100.5 - 3.8))) + 3000
print('\Delta_{+}    % .5f    % 10.5f    % 10.5f'%(a1_lsp[0, tmp_i], a1_rate_plus[tmp_i], 100.5 - 3.8))
ax[0].axvline(x=a1_lsp[0, tmp_i], ymin=0, ymax=1, linestyle='-', linewidth=0.5, color='gray')
ax[0].text(x=a1_lsp[0, tmp_i], y=120, s='$\Delta_{+}=%.3f$ eV'%(a1_lsp[0, tmp_i]), color='gray', ha='left', fontsize=10)

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
    "ppRPA (PBE)"
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
collect_a1_rate = np.zeros((delta_00.shape[0], 3))
for i in range(delta_00.shape[0]):
    for j in range(a1_lsp.shape[1]):
        if abs(delta_00[i] - a1_lsp[0, j]) < 1e-4:
            collect_a1_rate[i, 0] = isc_rate(lambda_perp, a1_lsp[1, j])
            tmp_a1_rate_uncertainty = collect_a1_rate[i, 0] * 2.8 * 2 / 29.4
            collect_a1_rate[i, 1] = collect_a1_rate[i, 0] - tmp_a1_rate_uncertainty
            collect_a1_rate[i, 2] = collect_a1_rate[i, 0] + tmp_a1_rate_uncertainty
            
print("ISC rates")
print(names)
print(collect_a1_rate)
print()


# left
tmp_i = np.argmin(abs(a1_rate_minus[3000:] - (100.5 + 3.8))) + 3000
ax[1].axvline(x=a1_lsp[0, tmp_i], ymin=0, ymax=1, linestyle='-', linewidth=0.5, color='gray')
# right
tmp_i = np.argmin(abs(a1_rate_plus[3000:] - (100.5 - 3.8))) + 3000
ax[1].axvline(x=a1_lsp[0, tmp_i], ymin=0, ymax=1, linestyle='-', linewidth=0.5, color='gray')


option = 2

if option == 0: # old version, energy acsending
    indices = np.argsort(delta_00[:])
    sigma = 0.01
    for i, ind in enumerate(indices):
        ax[1].hlines(y=collect_a1_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.0)
        ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_a1_rate[ind, 1], y2=collect_a1_rate[ind, 2], color=new_colors_alpha_0_5[i], label=names[ind])
elif option == 1: # QDET first
    indices = np.argsort(delta_00[1:]) + 1
    ind = 0
    sigma = 0.01
    ax[1].hlines(y=collect_a1_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[0], linewidth=1.0)
    ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_a1_rate[ind, 1], y2=collect_a1_rate[ind, 2], color=new_colors_alpha_0_5[i], label=names[ind])
    for i, ind in enumerate(indices):
        ax[1].hlines(y=collect_a1_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i+1], linewidth=1.0)
        ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_a1_rate[ind, 1], y2=collect_a1_rate[ind, 2], color=new_colors_alpha_0_5[i+1], label=names[ind])
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
        if i in [0, 1, 5]:
            ax[1].hlines(y=collect_a1_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.5, zorder=1)
            ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_a1_rate[ind, 1], y2=collect_a1_rate[ind, 2], color=new_colors_alpha_0_5[i], zorder=0)
            ax[1].text(x=delta_00[ind], y=collect_a1_rate[ind, 2] + 20, s=names2[ind], color=new_colors[i], fontsize=10, ha='center')
        elif i in [4]:
            ax[1].hlines(y=collect_a1_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.5, zorder=1)
            ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_a1_rate[ind, 1], y2=collect_a1_rate[ind, 2], color=new_colors_alpha_0_5[i], zorder=0)
            ax[1].text(x=delta_00[ind] + 0.0, y=collect_a1_rate[ind, 2] + 26, s=names2[ind], color=new_colors[i], fontsize=10, ha='center')
        elif i in [6]:
            ax[1].hlines(y=collect_a1_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.5, zorder=1)
            ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_a1_rate[ind, 1], y2=collect_a1_rate[ind, 2], color=new_colors_alpha_0_5[i], zorder=0)
            ax[1].text(x=delta_00[ind] + 0.03, y=collect_a1_rate[ind, 2] + 36, s=names2[ind], color=new_colors[i], fontsize=10, ha='center')
        elif i in [2]:
            ax[1].hlines(y=collect_a1_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.5, zorder=1)
            ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_a1_rate[ind, 1], y2=collect_a1_rate[ind, 2], color=new_colors_alpha_0_5[i], zorder=0)
            ax[1].text(x=delta_00[ind], y=collect_a1_rate[ind, 1] - 80, s=names2[ind], color=new_colors[i], fontsize=10, ha='center')
        elif i in [3]:
            ax[1].hlines(y=collect_a1_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.5, zorder=3)
            ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_a1_rate[ind, 1], y2=collect_a1_rate[ind, 2], color=new_colors_alpha_0_5[i], zorder=2)
            ax[1].text(x=delta_00[ind], y=collect_a1_rate[ind, 1] - 34, s=names2[ind], color=new_colors[i], fontsize=10, ha='center', fontweight='bold')
        elif i in [7]:
            ax[1].hlines(y=collect_a1_rate[ind, 0], xmin=delta_00[ind]-sigma, xmax=delta_00[ind]+sigma, color=new_colors[i], linewidth=1.5, zorder=1)
            ax[1].fill_between(x=np.linspace(delta_00[ind]-sigma, delta_00[ind]+sigma, 1001), y1=collect_a1_rate[ind, 1], y2=collect_a1_rate[ind, 2], color=new_colors_alpha_0_5[i], zorder=0)
            ax[1].text(x=delta_00[ind], y=collect_a1_rate[ind, 2] + 2, s=names2[ind], color=new_colors[i], fontsize=10, ha='center')

ax[1].plot(a1_lsp[0], 100.5 * np.ones(a1_lsp.shape[1]), color='gray', linewidth=0.5, zorder=-1)
ax[1].fill_between(a1_lsp[0], y1=100.5 - 3.8, y2=100.5 + 3.8, color='lightgray', zorder=-2)#, label="Expt.")



##################
# high temp rate #
##################

Batalov2008 = np.array([295, 7.8])
Robledo2011 = np.array([300, 7.3])
Toyli2012 = np.array([
    [300, 400, 475, 500, 550, 575, 600, 625, 650, 675],
    #[7.18, 6.38, 7.25, 6.75, 6.25, 6.65, 6.6, 4.95, 5.59, 5.58],
    [7.2013, 6.3873, 7.2623, 6.7865, 6.2913, 6.6269, 6.6269, 4.9591, 5.5701, 5.543],
])
Toyli2012_lower_bound_1_sigma = np.array([7.0708, 6.2427, 6.7798, 6.5871, 5.9756, 6.3992, 6.3992, 4.5443, 5.3353, 5.3657])
Toyli2012_upper_bound_1_sigma = np.array([7.3789, 6.5328, 7.7132, 7.0652, 6.5728, 7.0109, 7.0109, 5.3186, 5.9948, 6.1143])
Toyli2012_lower_bound_2_sigma = np.array([6.9123, 6.1012, 6.3366, 6.3722, 5.7265, 6.0153, 6.0153, 4.1883, 4.8521, 4.4626])
Toyli2012_upper_bound_2_sigma = np.array([7.4725, 6.6717, 8.2719, 7.2400, 6.8730, 7.1586, 7.1586, 5.7228, 6.1558, 6.2755])

print()
print("Toyli2012")
print(Toyli2012[1])
print('1 \sigma')
print(Toyli2012[1] - Toyli2012_lower_bound_1_sigma)
print(Toyli2012[1] - Toyli2012_upper_bound_1_sigma)
print('2 \sigma')
print(Toyli2012[1] - Toyli2012_lower_bound_2_sigma)
print(Toyli2012[1] - Toyli2012_upper_bound_2_sigma)
print()

thiswork = np.array([
    [300, 400, 500, 600, 700, 800],
    [7.08702913, 6.94483151, 6.82839995, 6.74759973, 6.70082689, 6.68237526]
])

thiswork_expt = np.array([
    [100, 150, 200, 250, 300],
    [7.4954210822430705, 7.288783938767406, 7.0954437, 7.065261, 7.056282179],
    [0.656355898, 0.3666295358267796, 0.3504929, 0.35732725, 0.165785441],
])

thiswork = np.array([
    [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700],
    [7.31370326, 7.3073782, 7.2777017, 7.22599323, 7.15979504, 7.08702913, 
     7.01410108, 6.94483151, 6.88239165, 6.82839995, 6.78348547, 6.74759973,
     6.72027722, 6.70082689],
])

ax[2].scatter(Batalov2008[0], Batalov2008[1], s=30, facecolors='none', edgecolors=green, marker='s', label="Batalov et al. (2008)")
ax[2].scatter(Robledo2011[0], Robledo2011[1], s=30, facecolors='none', edgecolors=yellow, marker='D', label="Robledo et al. (2011)")

ax[2].scatter(Toyli2012[0], Toyli2012[1], s=30, facecolors='none', edgecolors=red, marker='o', label="Toyli et al. (2012)")
one_sigma_error = [- Toyli2012_lower_bound_1_sigma + Toyli2012[1], Toyli2012_upper_bound_1_sigma - Toyli2012[1]]
ax[2].errorbar(Toyli2012[0], Toyli2012[1], yerr=one_sigma_error, fmt='none', ecolor=red, capsize=3)

ax[2].scatter(thiswork_expt[0], thiswork_expt[1], s=30, facecolors='none', edgecolors=purple, marker='^', label='This work, Expt.')
ax[2].errorbar(thiswork_expt[0], thiswork_expt[1], yerr=thiswork_expt[2], fmt='none', ecolor=purple, capsize=3)

ax[2].plot(thiswork[0, 1:], thiswork[1, 1:], linestyle='-', linewidth=1.0, marker='v', markersize=6, color=blue, label='This work, Theory')

ax[0].text(x=-0.12, y=1.0, s='(a)', fontsize=13, color='k', transform=ax[0].transAxes)
ax[1].text(x=-0.12, y=1.0, s='(b)', fontsize=13, color='k', transform=ax[1].transAxes)
ax[2].text(x=-0.12, y=1.0, s='(c)', fontsize=13, color='k', transform=ax[2].transAxes)


for i in range(3):

    if i==0 or i==1: ax[i].set_xlim((-0.05, 0.56))
    if i==2: ax[i].set_xlim((80, 720))
    if i==0: ax[i].set_ylim((0, 999))
    if i==1: ax[i].set_ylim((0, 650))
    if i==2: ax[i].set_ylim((4.5, 8.2))
    ax[i].tick_params(direction='in')
    if i==0: ax[i].legend(fontsize=10, loc='upper right', edgecolor='black', ncols=1,
                        borderpad=0.3, labelspacing=0.4, handlelength=0.8, handleheight=0.8,
                        handletextpad=0.6, borderaxespad=0.5, columnspacing=1.0)
    if (option == 0 or option == 1) and i==1:
        ax[i].legend(fontsize=10, loc='upper right', edgecolor='black', ncols=1,
                        borderpad=0.3, labelspacing=0.4, handlelength=0.8, handleheight=0.8,
                        handletextpad=0.6, borderaxespad=0.5, columnspacing=1.0)
    if i==2: ax[i].legend(fontsize=10, loc='lower left', edgecolor='black', ncols=2,
                        borderpad=0.3, labelspacing=0.4, handlelength=0.8, handleheight=0.8,
                        handletextpad=0.6, borderaxespad=0.5, columnspacing=1.0)
    ax[i].xaxis.set_ticks_position('both')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].tick_params(axis='both', direction='in')
    ax[i].yaxis.set_ticks_position('both')
    if i==0: ax[i].set_xticklabels([])
    if i==1: ax[i].set_xlabel("$\Delta$ (eV)")
    if i==0 or i==1: ax[i].set_ylabel('$\Gamma_{A_1}$ (MHz)')
    if i==2: ax[i].set_xlabel("$T$ (K)")
    if i==2: ax[i].set_ylabel("Lifetime (ns)")

plt.savefig("figure_4.pdf", bbox_inches='tight', dpi=300)
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
    for j in range(a1_lsp.shape[1]):
        if abs(delta_00[i] - a1_lsp[0, j]) < 1e-5:
            a1_rate = isc_rate(lambda_perp, a1_lsp[1, j])
            tmp_a1_rate_uncertainty = a1_rate * 2.8 * 2 / 29.4
            print(names[i], '%12.6f    %12.6f'%(a1_rate, tmp_a1_rate_uncertainty))
