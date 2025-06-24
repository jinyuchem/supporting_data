#!/usr/bin/env python3
import numpy as np
import sys
import json
from scipy.linalg import eigh
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import seaborn as sns
from scipy import constants

#############
# Functions #
#############

def parse_coeff_file(fname, num_eig, size):
    energies = np.zeros(num_eig, dtype=np.float64)
    A1_coeff = np.zeros((num_eig, size, 3), dtype=np.complex128)
    with open(fname, 'r') as f:
        for i in range(num_eig):
            line = f.readline()
            energies[i] = float(line.split()[3])
            for j in range(size):
                line = f.readline()
                A1_coeff[i, j, 0] = float(line.split()[0])
                A1_coeff[i, j, 1] = float(line.split()[1])
                A1_coeff[i, j, 2] = float(line.split()[2]) + 1j * float(line.split()[3])
            line = f.readline()

    return energies, A1_coeff


def partition_function(energies, T):
    part_fxn = np.exp(- ((energies - energies[0]) * 1e-3 * constants.eV) / (constants.k * T))
    part_fxn /= np.sum(part_fxn)
    return part_fxn


def Lorentzian(x, mu, gamma):
    pref = 1 / (np.pi * gamma)
    mp = gamma**2 / ((x - mu) ** 2 + gamma**2)
    return pref * mp

########
# main #
########

#########################
# load a1_spectral_fxns #
#########################

fnames = [
    '50K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '100K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '150K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '200K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '250K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '300K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '350K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '400K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '450K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '500K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '550K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '600K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '650K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
    '700K-SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat',
]

eneaxis = np.loadtxt(fnames[0], usecols=0)
a1_lsp = []
for fname in fnames:
    tmp = np.loadtxt(fname, usecols=1)
    a1_lsp.append(tmp)
a1_lsp = np.array(a1_lsp)

for i in range(a1_lsp.shape[0]):
    a1_lsp[i] *= (1 / np.pi / constants.hbar * 1e-15 * constants.eV / 1000 * 0.1)
    print(sum(a1_lsp[i]))
a1_lsp *= 1e4

###############
# remove ZPLs #
###############

ht_fnames = [
    "scaled-q-FC-NV-Temp-50-lsp.dat",
    "scaled-q-FC-NV-Temp-100-lsp.dat",
    "scaled-q-FC-NV-Temp-150-lsp.dat",
    "scaled-q-FC-NV-Temp-200-lsp.dat",
    "scaled-q-FC-NV-Temp-250-lsp.dat",
    "scaled-q-FC-NV-Temp-300-lsp.dat",
    "scaled-q-FC-NV-Temp-350-lsp.dat",
    "scaled-q-FC-NV-Temp-400-lsp.dat",
    "scaled-q-FC-NV-Temp-450-lsp.dat",
    "scaled-q-FC-NV-Temp-500-lsp.dat",
    "scaled-q-FC-NV-Temp-550-lsp.dat",
    "scaled-q-FC-NV-Temp-600-lsp.dat",
    "scaled-q-FC-NV-Temp-650-lsp.dat",
    "scaled-q-FC-NV-Temp-700-lsp.dat",
]

fcht_lsp = []
for fname in ht_fnames:
    tmp = np.loadtxt(fname, usecols=(1, 2, 3, 4), skiprows=1).T * 1e3
    fcht_lsp.append(tmp)
fcht_lsp = np.array(fcht_lsp)


total_a1_lsp = a1_lsp + fcht_lsp[:, 1, :] + fcht_lsp[:, 2, :]

###########################################
# compute the weighted spectral functions #
###########################################

fname = "collect_A1_coeff_30.out"

num_eig = 300
# tetatively
num_eig_to_use = 30
num_ph = 30
size = int((num_ph + 1) * (num_ph + 2) / 2)
temps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]

raw_vib_enes, raw_vib_A1_coeff = parse_coeff_file(fname, num_eig, size)
vib_enes = raw_vib_enes[: num_eig_to_use]
vib_A1_coeff = raw_vib_A1_coeff[: num_eig_to_use]

part_fxns = []
for temp in temps:
    part_fxn = partition_function(vib_enes, temp)
    part_fxns.append(part_fxn)
part_fxns = np.array(part_fxns)

print()
for j in range(num_eig_to_use):
    print("%4d    %12.8f    %5.1f    %12.8f    %12.8f    %12.8f"%(
        j+1, vib_enes[j], vib_enes[j], part_fxns[0, j], part_fxns[-1, j], np.linalg.norm(vib_A1_coeff[j, :, 2])**2))
print()

# compute the energy shifts and c2_coefficients for each vib_level
rel_vib_enes = vib_enes - vib_enes[0]
ene_shifts = np.zeros((num_eig_to_use, size), dtype=np.float64)
c2_coeffs = np.zeros((num_eig_to_use, size), dtype=np.float64)
for i in range(num_eig_to_use):
    # NOTE: all need to substract the relative vibrational energy
    ene_shifts[i] = 77.6 * (vib_A1_coeff[i, :, 0].real + vib_A1_coeff[i, :, 1].real) - rel_vib_enes[i]
    c2_coeffs[i] = (vib_A1_coeff[i, :, 2] * vib_A1_coeff[i, :, 2].conj()).real


temp_lsp = np.zeros(total_a1_lsp.shape, dtype=np.float64)
# for each temperature
for a in range(total_a1_lsp.shape[0]):
    # for each vibrational level
    for i in range(num_eig_to_use):
        # for each size
        for j in range(size):
            st = int(ene_shifts[i, j] / 0.1)
            if st > temp_lsp.shape[1]:
                continue
            elif st >= 0:
                ed = min(temp_lsp.shape[1] - st, temp_lsp.shape[1])
                temp_lsp[a, st:] += part_fxns[a, i] * c2_coeffs[i, j] * total_a1_lsp[a, :ed]
            elif st < 0:
                ed = temp_lsp.shape[1] + st
                temp_lsp[a, :ed] += part_fxns[a, i] * c2_coeffs[i, j] * total_a1_lsp[a, -st:]

########
# plot #
########

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58',
          '#FF6F61', '#46BDC6', '#FF9E80', '#7E57C2', '#9CCC65']
labels = ["50 K", "100 K", "150 K", "200 K", "250 K", "300 K", "350 K", "400 K", "450 K", "500 K", "550 K", "600 K", "650 K", "700 K"]

cmap = plt.cm.plasma
color_values = np.linspace(0, 1, len(fnames)+3)
h_colors = [cmap(value) for value in color_values]



print()
print("check only a_1 intensities")
print(0.3545)
for a in range(len(fnames)):
    for i in range(eneaxis.shape[0]):
        if abs(eneaxis[i] - 354.5) < 1e-5:
            print(labels[a], a1_lsp[a, i], total_a1_lsp[a, i])
print()



# plot spectral function
for i in range(1, a1_lsp.shape[0], 2):
    ax[0].plot(eneaxis * 1e-3, temp_lsp[i], color=h_colors[i],
                    linewidth=1, linestyle='-', label=labels[i])

# references
blue = "#4285F4"
red = "#DB4437"
yellow = "#F4B400"
green = "#0F9D58"
purple = "#AB47BC"



print()
print("check total intensities")
print(0.3545)
my_collect_spectral_fxn = []
for a in range(len(fnames)):
    for i in range(eneaxis.shape[0]):
        if abs(eneaxis[i] - 354.5) < 1e-5:
            print(labels[a], temp_lsp[a, i])
            my_collect_spectral_fxn.append(temp_lsp[a, i])
print()

print(0.334)
my_collect_spectral_fxn_left = []
for a in range(len(fnames)):
    for i in range(eneaxis.shape[0]):
        if abs(eneaxis[i] - 334) < 1e-5:
            print(labels[a], temp_lsp[a, i])
            my_collect_spectral_fxn_left.append(temp_lsp[a, i])
print()
print(0.375)
my_collect_spectral_fxn_right = []
for a in range(len(fnames)):
    for i in range(eneaxis.shape[0]):
        if abs(eneaxis[i] - 375) < 1e-5:
            print(labels[a], temp_lsp[a, i])
            my_collect_spectral_fxn_right.append(temp_lsp[a, i])
print()



# compute the temperature dependent lifetime
exp_temp = np.array([300, 400, 480, 500, 550, 575, 600, 625, 650, 675])
exp_lifetime = np.array([7.18, 6.38, 7.25, 6.75, 6.25, 6.65, 6.6, 4.95, 5.59, 5.58])
exp_rad_rate = 2 * np.pi * 13.2
exp_rad_rate_left = 2 * np.pi * (13.2 - 0.5)
exp_rad_rate_right = 2 * np.pi * (13.2 + 0.5)

my_temp = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
my_collect_spectral_fxn = np.array(my_collect_spectral_fxn)[:]

my_collect_spectral_fxn_left = np.array(my_collect_spectral_fxn_left)[:]
my_collect_spectral_fxn_right = np.array(my_collect_spectral_fxn_right)[:]

my_lambda_perp = 29.4
my_lambda_perp_left = 29.4 - 2.8
my_lambda_perp_right = 29.4 + 2.8
prefactor = 77.04406985667521 / 80.06910547505998
# 0.354 eV
my_isc_rate = (2 * np.pi / constants.hbar) * 2.0 * (my_lambda_perp * 1e9 * constants.h) ** 2 * (my_collect_spectral_fxn / constants.eV) / 1e6
# 0.504 eV
my_isc_rate *= prefactor**2
print("my_isc_rate", my_isc_rate)

my_isc_rate_left = prefactor**2 * (2 * np.pi / constants.hbar) * 2.0 * (my_lambda_perp * 1e9 * constants.h) ** 2 * (my_collect_spectral_fxn_left / constants.eV) / 1e6 * my_collect_spectral_fxn[0] / my_collect_spectral_fxn_left[0]
my_isc_rate_right = prefactor**2 * (2 * np.pi / constants.hbar) * 2.0 * (my_lambda_perp * 1e9 * constants.h) ** 2 * (my_collect_spectral_fxn_right / constants.eV) / 1e6 * my_collect_spectral_fxn[0] / my_collect_spectral_fxn_right[0]

# HT radative rate
high_temp_rad_lifetime = 12.05719265847692 / (1 + 3420 * np.exp(- (0.48 * constants.eV) / (constants.k * my_temp)))
print("high_temp_rad_lifetime", high_temp_rad_lifetime)
exp_rad_rate_high_temp = 1 / high_temp_rad_lifetime * 1e3
print("exp_rad_rate_high_temp", exp_rad_rate_high_temp)

fraction = 0.5
my_lifetime = 1e3 / (my_isc_rate + exp_rad_rate)
my_lifetime_lower = 1e3 / (my_isc_rate_right + exp_rad_rate)
my_lifetime_higher = 1e3 / (my_isc_rate_left + exp_rad_rate)

my_lifetime_f_high_temp = 1e3 / (my_isc_rate + (1 - fraction) * exp_rad_rate + fraction * exp_rad_rate_high_temp)
my_lifetime_high_temp = 1e3 / (my_isc_rate + exp_rad_rate_high_temp)
print("my_lifetime", my_lifetime)
print("my_lifetime_high_temp", my_lifetime_high_temp)
print("my_lifetime_f_high_temp", my_lifetime_f_high_temp)


print()
print()
print("data to be used for temperature dependent rate")
print(my_temp)
print(my_lifetime)
print()
print()


ax[0].axvline(x=0.3545, ymin=0, ymax=1, color='k', linewidth=1, linestyle='--')


# temperature dependent rate
ax[1].plot(my_temp[1:], my_isc_rate[1:], marker='o', markersize=5, linestyle='--', color=blue, linewidth=1.0)


ax[0].text(x=-0.2, y=1.02, s="(a)", fontsize=14, color='k', transform=ax[0].transAxes)
ax[1].text(x=-0.2, y=1.02, s="(b)", fontsize=14, color='k', transform=ax[1].transAxes)


for i in range(2):
    if i==0: ax[i].set_xlim((-0.15, 0.68))
    if i==1: ax[i].set_ylim((80, 720))
    if i==0: ax[i].set_ylim((0.0, 0.8))
    if i==1: ax[i].set_ylim((52, 68))
    ax[i].tick_params(direction='in')
    if i==0: ax[i].legend(fontsize=12, loc='upper right', edgecolor='black', ncols=1)
    ax[i].xaxis.set_ticks_position('both')
    ax[i].yaxis.set_ticks_position('both')
    if i==0: ax[i].set_xlabel("$\Delta$ (eV)")
    if i==0: ax[i].set_ylabel("$F(\Delta, T)$ (eV$^{-1}$)")
    if i==1: ax[i].set_xlabel("$T$ (K)")
    if i==1: ax[i].set_ylabel("$\Gamma_{\mathrm{ISC}}$ (MHz)")

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.savefig("figure_s9.pdf", bbox_inches='tight', dpi=300)
plt.show()
