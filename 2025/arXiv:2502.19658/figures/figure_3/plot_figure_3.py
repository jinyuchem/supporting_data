#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
from sklearn.metrics import auc
from scipy import constants

#######
# fxn #
#######

def Lorentzian(x, mu, gamma):
    pref = 1 / (np.pi * gamma)
    mp = gamma**2 / ((x - mu) ** 2 + gamma**2)
    return pref * mp

#########
# Input #
#########

fnames = [
'5K-1A1-SCa1-PL-All-B-1A1-3E.dat-a1-modes.dat',
]

# load the data of FCHT and HT terms
# fc, fcht, ht, total
fcht_1d_lsp = np.loadtxt('scaled-q-FC-NV-Temp-5-lsp.dat', usecols=(1, 2, 3, 4), skiprows=1).T * 1e3

########
# Data #
########

eneaxis = np.loadtxt(fnames[0], usecols=0)
lsp = []
for i in range(len(fnames)):
    p = np.loadtxt(fnames[i], usecols=1)
    lsp.append(p)
lsp = np.array(lsp)

triplet_lsp = np.array([np.loadtxt('5K-3A2-SCa1-PL-All-B-3A2-3E.dat-a1-modes.dat', usecols=1)])

print("lsp data")
for i in range(len(fnames)):
    print(sum(lsp[i]) / np.pi / constants.hbar * 1e-15 * constants.eV / 1000 * 0.1)
    print(auc(eneaxis, lsp[i]))
    # normalization
    lsp[i] = lsp[i] / auc(eneaxis,lsp[i]) * 1000
    print(sum(lsp[i]))
    print()
print('remove ZPL')
zpl_lsp = Lorentzian(eneaxis, 0.0, 0.3)
print(sum(zpl_lsp))
lsp -= zpl_lsp[np.newaxis, :] * 1e3 * 0.1081

print()
print("triplet lsp data")
for i in range(len(fnames)):
    print(sum(triplet_lsp[i]) / np.pi / constants.hbar * 1e-15 * constants.eV / 1000 * 0.1)
    print(auc(eneaxis, triplet_lsp[i]))
    # normalization
    triplet_lsp[i] = triplet_lsp[i] / auc(eneaxis, triplet_lsp[i]) * 1000
    print(sum(triplet_lsp[i]))
    print()
print('remove ZPL')
zpl_lsp = Lorentzian(eneaxis, 0.0, 0.3)
print(sum(zpl_lsp))
triplet_lsp -= zpl_lsp[np.newaxis, :] * 1e3 * 0.0551

print()

print("fcht_1d_lsp data")
for i in range(4):
    print(auc(eneaxis,fcht_1d_lsp[i]))
    # normalization
    #lsp[i] = lsp[i] / auc(eneaxis,lsp[i]) * 1000
    print(sum(fcht_1d_lsp[i]))
print()

#######################
# add fcht correction #
#######################

original_lsp = np.copy(lsp)
lsp[:, :] = 0.0

for i in range(lsp.shape[0]):
    lsp[i] = original_lsp[i, :] + fcht_1d_lsp[1, :] + fcht_1d_lsp[2, :]

print()
print('original lsp', sum(original_lsp[0]))
print('fcht lsp', sum(fcht_1d_lsp[1]))
print('ht lsp', sum(fcht_1d_lsp[2]))
print('corrected lsp', sum(lsp[0]))

#########################
# shift and coefficient #
#########################

# mev
shifts = 77.6 * np.array([0, 1, 2, 3, 4])
# c2
c2_coeff = np.array([0.5751, 0.0, 0.0712, 0.0026, 0.0016])
# d2
d2_coeff = np.array([0.0, 0.3303, 0.0039, 0.0133, 0.0007])
# f2
f2_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

a1_lsp = np.zeros(lsp.shape)
e12_lsp = np.zeros(lsp.shape)

for i in range(lsp.shape[0]):
    for j in range(len(shifts)):
        st = int(shifts[j] / 0.1)
        ed = lsp.shape[1] - st
        print(st, ed)
        a1_lsp[i, st:] += c2_coeff[j] * lsp[i, :ed]
        e12_lsp[i, st:] += d2_coeff[j] / 2 * lsp[i, :ed]

# write to files
energy_range = np.linspace(200, 800, 61)
indices = [int((energy_range[i] + 150) / 0.1) for i in range(len(energy_range))]

# fname = 'to_use_f.dat'
# with open(fname, 'w') as w:
#     for i in indices:
#         nline = '%4d    %12.6e\n'%(eneaxis[i], lsp[0, i])
#         w.writelines(nline)

# fname = 'to_use_f_A1.dat'
# with open(fname, 'w') as w:
#     for i in indices:
#         nline = '%4d    %12.6e\n'%(eneaxis[i], a1_lsp[0, i])
#         w.writelines(nline)

# fname = 'to_use_f_E12.dat'
# with open(fname, 'w') as w:
#     for i in indices:
#         nline = '%4d    %12.6e\n'%(eneaxis[i], e12_lsp[0, i])
#         w.writelines(nline)


########
# Plot #
########

fig, ax = plt.subplots(3, 1, figsize=(7.5, 6))

colors = ['#4285F4', '#DB4437',
          '#F4B400', '#0F9D58', '#F4B400', '#0F9D58', '#F4B400', '#0F9D58',]
linestyles = ['-', '--', '--', '-', '-', '--',
              '-', '--', '--', '-', '-', '--',
              '-']

##############
# 3A2 vs 1A1 #
##############

more_colors = ["#632D24", '#AB47BC', '#FF6F61', '#46BDC6', '#FF9E80', '#7E57C2', '#9CCC65']

ax[0].plot(eneaxis / 1000, original_lsp[0], color=more_colors[1], label="$F^{a_1}(\Delta)$, ${}^3E \\to {}^1A_1$")
ax[0].plot(eneaxis / 1000, triplet_lsp[0], color=colors[0], label="$F^{a_1}(\Delta)$, ${}^3E \\to {}^3A_2$")

# Experimental data
fname_exp = 'NV-3E-3A2-exp.csv'
r_exp_eneaxis = np.loadtxt(open(fname_exp, "rb"), delimiter=",", usecols=1, skiprows=1) * 1000 - 1946.34
r_exp_lsp = np.loadtxt(open(fname_exp, "rb"), delimiter=",", usecols=2, skiprows=1)
exp_eneaxis = r_exp_eneaxis[1544:14046]
exp_lsp = r_exp_lsp[1544:14046]
# normalization
exp_lsp = exp_lsp / (1.945 + exp_eneaxis * 1e-3)**3
exp_lsp = exp_lsp / sum(exp_lsp) / (r_exp_eneaxis[1] - r_exp_eneaxis[0]) * 1000
# remove zpl
zpl_lsp = Lorentzian(exp_eneaxis, 0.0, 0.3)
print(sum(zpl_lsp))
exp_lsp = exp_lsp - zpl_lsp * 1e3 * 0.033
ax[0].fill_between(- exp_eneaxis * 1e-3, exp_lsp, color='lightgray', label="Expt. PL, ${}^3E \\to {}^3A_2$")


# print 0.3 to 0.4 eV
print()
for i in range(eneaxis.shape[0]):
    if abs(eneaxis[i] - 300) < 1e-5:
        print(0.3, original_lsp[0][i], triplet_lsp[0][i])
    if abs(eneaxis[i] - 400) < 1e-5:
        print(0.4, original_lsp[0][i], triplet_lsp[0][i])    
for i in range(exp_eneaxis.shape[0]):
    if abs(exp_eneaxis[i] + 300) < 1e-2:
        print(0.3, -exp_eneaxis[i], "exp", exp_lsp[i])
    if abs(exp_eneaxis[i] + 400) < 2e-2:
        print(0.4, -exp_eneaxis[i], "exp", exp_lsp[i])


# original
ax[1].plot(eneaxis / 1000, original_lsp[0],
          color='#666666',
          label='FC',
          linewidth=1.5,
          linestyle='--')
ax[1].plot(eneaxis / 1000, fcht_1d_lsp[1, :],
          color='#666666',
          label='FCHT',
          linewidth=1.5,
          linestyle='-.')
ax[1].plot(eneaxis / 1000, fcht_1d_lsp[2, :],
          color='#666666',
          label='HT',
          linewidth=1.5,
          linestyle=':')
ax[1].plot(eneaxis / 1000, lsp[0],
          color='black',
          label='Total',
          linewidth=1.5,
          linestyle='-')
ax[1].text(x=0.08, y=0.68, s='$F^{a_1}(\Delta)$\n${}^3E \\to {}^1A_1$', fontsize=12, transform=ax[1].transAxes, ha='center')

# A1
ax[2].plot(eneaxis / 1000, a1_lsp[0],
          color=colors[1],
          label='$F_{A_1}(\Delta)$',
          linewidth=1.5,
          linestyle='-')
# E12
ax[2].plot(eneaxis / 1000, e12_lsp[0],
          color="#D99A00",
          label='$F_{E_{1,2}}(\Delta)$',
          linewidth=1.5,
          linestyle='-')

np.savetxt('final_a1_lsp.dat', np.array([eneaxis / 1000, a1_lsp[0]]).T, delimiter=' ')
np.savetxt('final_e12_lsp.dat', np.array([eneaxis / 1000, e12_lsp[0]]).T, delimiter=' ')

ax[0].text(x=-0.11, y=0.94, s="(a)", fontsize=14, color='k', transform=ax[0].transAxes)
ax[1].text(x=-0.11, y=0.94, s="(b)", fontsize=14, color='k', transform=ax[1].transAxes)
ax[2].text(x=-0.11, y=0.94, s="(c)", fontsize=14, color='k', transform=ax[2].transAxes)

for i in range(3):
    ax[i].set_xlim((-0.05, 0.56))
    if i==0: ax[i].set_ylim((0, 6))
    if i==0: ax[i].set_yticks([0, 2, 4, 6])
    if i==0: ax[i].set_yticklabels(["0.0", "2.0", "4.0", "6.0"])
    if i==1: ax[i].set_ylim((-1., 6.))
    if i==1: ax[i].set_yticks([0, 2, 4, 6])
    if i==1: ax[i].set_yticklabels(["0.0", "2.0", "4.0", "6.0"])
    if i==2: ax[i].set_ylim((0., 3.))
    if i==2: ax[i].set_yticks([0, 1, 2, 3])
    if i==2: ax[i].set_yticklabels(["0.0", "1.0", "2.0", "3.0"])
    ax[i].tick_params(direction='in')
    if i==1:
        ax[i].legend(fontsize=11, loc='upper right', edgecolor='black', ncols=1)
    else:
        ax[i].legend(fontsize=11, loc='upper right', edgecolor='black')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].xaxis.set_ticks_position('both')
    if i==2: ax[i].set_xlabel("$\Delta$ (eV)")
    if i==0: ax[i].set_ylabel('$F(\Delta)$ (eV$^{-1}$)')
    if i==1: ax[i].set_ylabel('$F(\Delta)$ (eV$^{-1}$)')
    if i==2: ax[i].set_ylabel('$F(\Delta)$ (eV$^{-1}$)')
    ax[i].tick_params(axis='both', direction='in')
    ax[i].yaxis.set_ticks_position('both')
    if i==0 or i==1: ax[i].set_xticklabels([])

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.3, hspace=0.1)

plt.savefig("figure_3.pdf",bbox_inches='tight',dpi=300)
plt.show()
