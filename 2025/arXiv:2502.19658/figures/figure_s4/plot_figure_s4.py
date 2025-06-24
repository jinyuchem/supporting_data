#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})
import json
from scipy import constants

# load soc mat
soc_mat = np.load("soc_matrices.npy")[:, 0, 0, :, :]

# change unit
# ry to eV
soc_mat *= 13.605693122994017
# NOTE: need to figure out this factor of 2
soc_mat *= 2.0

# use only partial soc mat
indices = [962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023]
partial_soc_mat = soc_mat[:, indices, :][:, :, indices]

# rotate direction
R = np.array([[1/np.sqrt(2), 1/np.sqrt(6), 1/np.sqrt(3)],
              [-1/np.sqrt(2), 1/np.sqrt(6), 1/np.sqrt(3)],
              [0, -2/np.sqrt(6), 1/np.sqrt(3)]])

rot_partial_soc_mat = np.einsum('ij,jmn->imn', R.T, partial_soc_mat)
print(rot_partial_soc_mat[:, -2, -1])

##################
# print elements #
##################

raw_indices = [1009, 1019, 1022, 1023, 1024]
t_indices = np.array([1009, 1019, 1022, 1023, 1024], dtype=np.int32) - 1
t_indices -= 962
test_partial_soc_mat = rot_partial_soc_mat[:, t_indices, :][:, :, t_indices].imag * constants.eV / constants.h / 1e9

print()
for i in range(5):
    for j in range(5):
        print('%4d    %4d    % 12.6f    % 12.6f    % 12.6f'%(
            raw_indices[i], raw_indices[j], test_partial_soc_mat[0, i, j], test_partial_soc_mat[1, i, j], test_partial_soc_mat[2, i, j]
        ))
print()

########
# plot #
########

fig, ax = plt.subplots(3, 1, figsize=(12, 8))

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58', '#AB47BC',
          '#FF6F61', '#46BDC6', '#FF9E80', '#7E57C2', '#9CCC65']
labels = [r'$z_x$', r'$z_y$', r'$z_z$']
markers = ['o', 's', '^']
texts = [
    r'(a)    $\lambda = \langle a_1 | z_{\alpha} | \varphi \rangle$',
    r'(b)    $\lambda = \langle e_x | z_{\alpha} | \varphi \rangle$',
    r'(c)    $\lambda = \langle e_y | z_{\alpha} | \varphi \rangle$',
]

xaxis = np.copy(np.array(indices, dtype=np.int32) + 1)

soc_mat_a1 = np.copy(rot_partial_soc_mat[:, -3, :])
soc_mat_a1_p = np.copy(rot_partial_soc_mat[:, :, -3])
soc_mat_e1 = np.copy(rot_partial_soc_mat[:, -2, :])
soc_mat_e1_p = np.copy(rot_partial_soc_mat[:, :, -2])
soc_mat_e2 = np.copy(rot_partial_soc_mat[:, -1, :])
soc_mat_e2_p = np.copy(rot_partial_soc_mat[:, :, -1])

for i in range(3):
    ax[0].plot(xaxis, np.abs(soc_mat_a1[i]) * constants.eV / constants.h / 1e9, 
               label=labels[i], color=colors[4+i], marker=markers[i], markersize=4, linewidth=0.)
    ax[1].plot(xaxis, np.abs(soc_mat_e1[i]) * constants.eV / constants.h / 1e9, 
               label=labels[i], color=colors[4+i], marker=markers[i], markersize=4, linewidth=0.)
    ax[2].plot(xaxis, np.abs(soc_mat_e2[i]) * constants.eV / constants.h / 1e9, 
               label=labels[i], color=colors[4+i], marker=markers[i], markersize=4, linewidth=0.)

for i in range(3):
    ax[i].text(x=-0.07, y=1.08, s=texts[i], ha='left', transform=ax[i].transAxes, fontsize=13)
    ax[i].axvline(x=1022, ymin=0, ymax=1, color='gray', 
                  linestyle='--', linewidth=0.5)#, label=r'$a_1$')
    ax[i].axvline(x=1023, ymin=0, ymax=1, color='gray', 
                  linestyle='--', linewidth=0.5)#, label=r'$e_x$')
    ax[i].axvline(x=1024, ymin=0, ymax=1, color='gray', 
                  linestyle='--', linewidth=0.5)#, label=r'$e_y$')
    ax[i].text(x=63 / 68, y=1.05, s=r'$a_1$ $e_x$ $e_y$', ha='center', color='gray', transform=ax[i].transAxes, fontsize=10)
    # 1009
    ax[i].axvline(x=1009, ymin=0, ymax=1, color='gray', 
                  linestyle='--', linewidth=0.5)
    ax[i].text(x=49 / 68, y=1.05, s=r'$a_1^{\prime\prime}$', ha='center', color='gray', transform=ax[i].transAxes, fontsize=10)
    # 1019
    ax[i].axvline(x=1019, ymin=0, ymax=1, color='gray', 
                  linestyle='--', linewidth=0.5)
    ax[i].text(x=59 / 68, y=1.05, s=r'$a_1^{\prime}$', ha='center', color='gray', transform=ax[i].transAxes, fontsize=10)


for i in range(3):
    ax[i].legend(fontsize=12, loc='upper left', edgecolor='black', ncol=3)

    if i in [0, 1, 2]: ax[i].set_xlabel('Index of active space orbital')
    ax[i].set_ylabel('$|\lambda|$ (GHz)', color='black')
    ax[i].tick_params(axis='y', colors='black')
    ax[i].tick_params(axis='both', direction='in')
    ax[i].tick_params(which='minor', direction='in')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].yaxis.set_ticks_position('both')
    ax[i].set_xlim([960, 1028])
    ax[i].set_ylim([-30, 500])
    
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.0, hspace=0.5)

plt.savefig("figure_s4.pdf", bbox_inches='tight', dpi=300)
plt.show()
