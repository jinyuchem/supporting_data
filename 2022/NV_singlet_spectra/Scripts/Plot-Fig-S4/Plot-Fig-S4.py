#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})
from scipy.optimize import curve_fit
from scipy import constants
from scipy.linalg import eigh
from matplotlib.gridspec import GridSpec

#######
# fxn #
#######

def harmonic(x, y, eph):
    x = x * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
    t_eph = eph * 1e-3 * constants.eV / constants.hbar
    x = x * np.sqrt(t_eph / constants.hbar)

    y = y * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
    t_eph = eph * 1e-3 * constants.eV / constants.hbar
    y = y * np.sqrt(t_eph / constants.hbar)
    return 0.5 * eph * (x**2 + y**2)

def numerical_solver(x, y, eph, Le, Ft, Gt, F, G, coeff):
    x = x * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
    t_eph = eph * 1e-3 * constants.eV / constants.hbar
    x = x * np.sqrt(t_eph / constants.hbar)

    y = y * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
    t_eph = eph * 1e-3 * constants.eV / constants.hbar
    y = y * np.sqrt(t_eph / constants.hbar)

    mat = np.array([
          [      Le,        Ft * x,        Ft * y],
          [  Ft * x,        F * x,        - F * y],
          [  Ft * y,      - F * y,        - F * x]
          ])
    w, v = eigh(mat)
    e_1 = w[0] + 0.5 * eph * (x**2 + y**2)
    e_2 = w[1] + 0.5 * eph * (x**2 + y**2)
    e_3 = w[2] + 0.5 * eph * (x**2 + y**2) - Le
    ev_1 = abs(v[:,0])**2
    ev_2 = abs(v[:,1])**2
    ev_3 = abs(v[:,2])**2
    return e_1, e_2, e_3, ev_1, ev_2, ev_3

###############
## parameters #
###############

# path along y=0
coeff = 0
Gt = 0
G = 0

Le = 821
eph = 62.9506828
Ft = 133.22436286
F = 62.37653058

# Å amu^0.5 to m kg^0.5
anchor = 0.825090 / 2 * 1e-10 * constants.physical_constants['atomic mass constant'][0]**0.5
t_eph = eph * 1e-3 * constants.eV / constants.hbar
anchor = anchor * np.sqrt(t_eph / constants.hbar)

y_axis = np.linspace(-0.8,0.8,800)
x_axis = np.ones(y_axis.shape[0]) * anchor * 0

ene_harmonic = harmonic(x_axis, y_axis, eph)

path_a_ene = np.zeros(x_axis.shape[0])
path_e_ene = np.zeros(x_axis.shape[0])
path_ep_ene = np.zeros(x_axis.shape[0])

eigen_a = np.zeros((x_axis.shape[0], 3))
eigen_e = np.zeros((x_axis.shape[0], 3))
eigen_ep = np.zeros((x_axis.shape[0], 3))


for i in range(x_axis.shape[0]):
    path_a_ene[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[2]
    path_ep_ene[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[1]
    path_e_ene[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[0]
    
    eigen_a[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[5]
    eigen_ep[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[4]
    eigen_e[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[3]

##########
# PJT only

Le = 821
eph = 62.9506828
Ft = 133.22436286
F = 0

path_a_ene_pjt = np.zeros(x_axis.shape[0])
path_e_ene_pjt = np.zeros(x_axis.shape[0])
path_ep_ene_pjt = np.zeros(x_axis.shape[0])

eigen_a_pjt = np.zeros((x_axis.shape[0], 3))
eigen_e_pjt = np.zeros((x_axis.shape[0], 3))
eigen_ep_pjt = np.zeros((x_axis.shape[0], 3))


for i in range(x_axis.shape[0]):
    path_a_ene_pjt[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[2]
    path_ep_ene_pjt[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[1]
    path_e_ene_pjt[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[0]

    eigen_a_pjt[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[5]
    eigen_ep_pjt[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[4]
    eigen_e_pjt[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[3]

##########
# DJT only

Le = 821
eph = 62.9506828
F = 62.37653058
Ft = 0

path_a_ene_djt = np.zeros(x_axis.shape[0])
path_e_ene_djt = np.zeros(x_axis.shape[0])
path_ep_ene_djt = np.zeros(x_axis.shape[0])

eigen_a_djt = np.zeros((x_axis.shape[0], 3))
eigen_e_djt = np.zeros((x_axis.shape[0], 3))
eigen_ep_djt = np.zeros((x_axis.shape[0], 3))


for i in range(x_axis.shape[0]):
    path_a_ene_djt[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[2]
    path_ep_ene_djt[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[1]
    path_e_ene_djt[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[0]

    eigen_a_djt[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[5]
    eigen_ep_djt[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[4]
    eigen_e_djt[i] = numerical_solver(x_axis[i], y_axis[i], eph, Le, Ft, Gt, F, G, coeff)[3]

########
# Plot #
########

fig = plt.figure(figsize=(12, 7))
gs = GridSpec(nrows=5, ncols=3, height_ratios=[3, 0.4, 1, 1, 1],
                                width_ratios=[1, 1, 1],
                                hspace=0.1, wspace=0.15,
                                left=0.05, right=0.9,
                                bottom=0.02, top=0.98)

ax = [
[fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2])],
[fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,1]), fig.add_subplot(gs[2,2])],
[fig.add_subplot(gs[3,0]), fig.add_subplot(gs[3,1]), fig.add_subplot(gs[3,2])],
[fig.add_subplot(gs[4,0]), fig.add_subplot(gs[4,1]), fig.add_subplot(gs[4,2])]
]

colors = ['#4285F4', '#DB4437', '#0F9D58']
labels = ['$^3A_2$', '$^1E$', '$^1E^{\prime}$', '$^1A_1$']
markers = ['s', 'o', '^', 'v']

ax[0][0].plot(y_axis, path_e_ene, color=colors[0], label='')
ax[0][0].plot(y_axis, path_ep_ene, color=colors[1], label='')
ax[0][0].plot(y_axis, path_a_ene, color=colors[2], label='')
ax[0][0].plot(y_axis, ene_harmonic, color='black',
              linestyle='--', linewidth=0.5, label='')


ax[0][1].plot(y_axis, path_e_ene_pjt, color=colors[0], label='')
ax[0][1].plot(y_axis, path_ep_ene_pjt, color=colors[1], label='')
ax[0][1].plot(y_axis, path_a_ene_pjt, color=colors[2], label='')
ax[0][1].plot(y_axis, ene_harmonic, color='black',
              linestyle='--', linewidth=0.5, label='')


ax[0][2].plot(y_axis, path_e_ene_djt, color=colors[0],
              label='$|^1E_{low}\\rangle$')
ax[0][2].plot(y_axis, path_ep_ene_djt, color=colors[1],
              label='$|^1E_{high}\\rangle$')
ax[0][2].plot(y_axis, path_a_ene_djt, color=colors[2],
              label='$|^1A_1\\rangle$')
ax[0][2].plot(y_axis, ene_harmonic, color='black',
              linestyle='--', linewidth=0.5,
              label='harmonic')



ax[1][0].plot(y_axis, eigen_a[:,0], color=colors[2], linestyle='-')
ax[1][0].plot(y_axis, eigen_a[:,1], color=colors[2], linestyle='--')
ax[1][0].plot(y_axis, eigen_a[:,2], color=colors[2], linestyle=':')

ax[2][0].plot(y_axis, eigen_ep[:,0], color=colors[1], linestyle='-')
ax[2][0].plot(y_axis, eigen_ep[:,1], color=colors[1], linestyle='--')
ax[2][0].plot(y_axis, eigen_ep[:,2], color=colors[1], linestyle=':')

ax[3][0].plot(y_axis, eigen_e[:,0], color=colors[0], linestyle='-')
ax[3][0].plot(y_axis, eigen_e[:,1], color=colors[0], linestyle='--')
ax[3][0].plot(y_axis, eigen_e[:,2], color=colors[0], linestyle=':')




ax[1][1].plot(y_axis, eigen_a_pjt[:,0], color=colors[2], linestyle='-',
              label='$|\langle ^1A_1(Q) |^1A_1^{(0)} \\rangle|^2$')
ax[1][1].plot(y_axis, eigen_a_pjt[:,1], color=colors[2], linestyle='--',
              label='$|\langle ^1A_1(Q) |^1E_x^{(0)} \\rangle|^2$')
ax[1][1].plot(y_axis, eigen_a_pjt[:,2], color=colors[2], linestyle=':',
              label='$|\langle ^1A_1(Q) |^1E_y^{(0)} \\rangle|^2$')

ax[2][1].plot(y_axis, eigen_ep_pjt[:,0], color=colors[1], linestyle='-',
              label='$|\langle ^1E_{high}(Q) |^1A_1^{(0)} \\rangle|^2$')
ax[2][1].plot(y_axis, eigen_ep_pjt[:,1], color=colors[1], linestyle='--',
              label='$|\langle ^1E_{high}(Q) |^1E_x^{(0)} \\rangle|^2$')
ax[2][1].plot(y_axis, eigen_ep_pjt[:,2], color=colors[1], linestyle=':',
              label='$|\langle ^1E_{high}(Q) |^1E_y^{(0)} \\rangle|^2$')

ax[3][1].plot(y_axis, eigen_e_pjt[:,0], color=colors[0], linestyle='-',
              label='$|\langle ^1E_{low}(Q) |^1A_1^{(0)} \\rangle|^2$')
ax[3][1].plot(y_axis, eigen_e_pjt[:,1], color=colors[0], linestyle='--',
              label='$|\langle ^1E_{low}(Q) |^1E_x^{(0)} \\rangle|^2$')
ax[3][1].plot(y_axis, eigen_e_pjt[:,2], color=colors[0], linestyle=':',
              label='$|\langle ^1E_{low}(Q) |^1E_y^{(0)} \\rangle|^2$')



ax[1][2].plot(y_axis, eigen_a_djt[:,0], color=colors[2], linestyle='-',
              label='$|\langle ^1A_1(Q) |^1A_1(0) \\rangle|^2$')
ax[1][2].plot(y_axis, eigen_a_djt[:,1], color=colors[2], linestyle='--',
              label='$|\langle ^1A_1(Q) |^1E_x(0) \\rangle|^2$')
ax[1][2].plot(y_axis, eigen_a_djt[:,2], color=colors[2], linestyle=':',
              label='$|\langle ^1A_1(Q) |^1E_y(0) \\rangle|^2$')

ax[2][2].plot(y_axis, eigen_ep_djt[:,0], color=colors[1], linestyle='-',
              label='$|\langle ^1E_{high}(Q) |^1A_1(0) \\rangle|^2$')
ax[2][2].plot(y_axis, eigen_ep_djt[:,1] - 0.02, color=colors[1], linestyle='--',
              label='$|\langle ^1E_{high}(Q) |^1E_x(0) \\rangle|^2$')
ax[2][2].plot(y_axis, eigen_ep_djt[:,2] + 0.02, color=colors[1], linestyle=':',
              label='$|\langle ^1E_{high}(Q) |^1E_y(0) \\rangle|^2$')

ax[3][2].plot(y_axis, eigen_e_djt[:,0], color=colors[0], linestyle='-',
              label='$|\langle ^1E_{low}(Q) |^1A_1(0) \\rangle|^2$')
ax[3][2].plot(y_axis, eigen_e_djt[:,1] - 0.02, color=colors[0], linestyle='--',
              label='$|\langle ^1E_{low}(Q) |^1E_x(0) \\rangle|^2$')
ax[3][2].plot(y_axis, eigen_e_djt[:,2] + 0.02, color=colors[0], linestyle=':',
              label='$|\langle ^1E_{low}(Q) |^1E_y(0) \\rangle|^2$')





for i in range(3):
    for j in range(4):
        ax[j][i].set_xticks([-0.6,-0.3,0.0,0.3,0.6])
        ax[j][i].axvline(x=anchor, ymin=0, ymax=1, color='gray',
                         linestyle='--', linewidth=0.5)
        ax[j][i].axvline(x=-anchor, ymin=0, ymax=1, color='gray',
                         linestyle='--', linewidth=0.5)
        ax[j][i].tick_params(axis='both', direction='in')
        ax[j][i].tick_params(which='minor', direction='in')
        ax[j][i].xaxis.set_ticks_position('both')
        ax[j][i].yaxis.set_ticks_position('both')
        ax[j][i].set_xlim(-0.75,0.75)

    ax[0][i].set_ylim(-120, 500)
    ax[1][i].set_xticklabels([])
    ax[2][i].set_xticklabels([])
    ax[1][i].set_ylim(-0.1, 1.1)
    ax[2][i].set_ylim(-0.1, 1.1)
    ax[3][i].set_ylim(-0.1, 1.1)

ax[0][0].text(x=-0.49, y=0.78*620-120, s='PJT + DJT')
ax[0][1].text(x=-0.49, y=0.78*620-120, s='PJT')
ax[0][2].text(x=-0.49, y=0.78*620-120, s='DJT')
ax[1][0].text(x=-0.49, y=0.78*2.4 - 1.2, s='PJT + DJT')
ax[1][1].text(x=-0.49, y=0.78*2.4 - 1.2, s='PJT')
ax[1][2].text(x=-0.49, y=0.78*2.4 - 1.2, s='DJT')

ax[0][0].text(x=-4.5/3*0.7, y=620*0.97-120, s='a', fontsize=15, weight='bold')
ax[0][1].text(x=-3.7/3*0.7, y=620*0.97-120, s='b', fontsize=15, weight='bold')
ax[0][2].text(x=-3.7/3*0.7, y=620*0.97-120, s='c', fontsize=15, weight='bold')
ax[1][0].text(x=-4.5/3*0.7, y=2.4*0.97-1.2, s='d', fontsize=15, weight='bold')
ax[1][1].text(x=-3.7/3*0.7, y=2.4*0.97-1.2, s='e', fontsize=15, weight='bold')
ax[1][2].text(x=-3.7/3*0.7, y=2.4*0.97-1.2, s='f', fontsize=15, weight='bold')



ax[0][2].legend(fontsize=10, loc='upper center',
                edgecolor='black', handlelength=1.0, labelspacing=0.2,
                handleheight=0.5, handletextpad=0.5, columnspacing=1.0, borderaxespad=0.3)
ax[1][1].legend(fontsize=10, loc='center',
                edgecolor='black', handlelength=1.0, labelspacing=0.2,
                handleheight=0.5, handletextpad=0.5, columnspacing=1.0, borderaxespad=0.3)
ax[2][1].legend(fontsize=10, loc='center',
                edgecolor='black', handlelength=1.0, labelspacing=0.2,
                handleheight=0.5, handletextpad=0.5, columnspacing=1.0, borderaxespad=0.3)
ax[3][1].legend(fontsize=10, loc='center',
                edgecolor='black', handlelength=1.0, labelspacing=0.2,
                handleheight=0.5, handletextpad=0.5, columnspacing=1.0, borderaxespad=0.3)




ax[0][1].set_xlabel("$Q_\\beta$ (amu$^{0.5}$ Å)")
ax[3][1].set_xlabel("$Q_\\beta$ (amu$^{0.5}$ Å)")

ax[0][1].set_yticklabels([])
ax[0][2].set_yticklabels([])
ax[1][1].set_yticklabels([])
ax[1][2].set_yticklabels([])
ax[2][1].set_yticklabels([])
ax[2][2].set_yticklabels([])
ax[3][1].set_yticklabels([])
ax[3][2].set_yticklabels([])

ax[0][0].set_ylabel('$E$ (meV)')
ax[2][0].set_ylabel('Wavefunction')


plt.savefig("Fig-S4.pdf", bbox_inches='tight', dpi=300)
plt.show()
