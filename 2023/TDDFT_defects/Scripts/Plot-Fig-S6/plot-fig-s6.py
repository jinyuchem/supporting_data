#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})
from scipy.optimize import curve_fit

#
pbecells = np.array([54, 64, 128, 216, 250, 512])
pbelevels = np.array([
[8.1957,  10.2181,  13.1866,  14.8899],
[8.1728,  10.4276,  13.1672,  14.7810],
[8.2643,  10.6017,  12.9550,  14.5631],
[8.2893,  10.7002,  12.8764,  14.3310],
[8.2978,  10.7112,  12.8590,  14.3046],
[8.3179,  10.7470,  12.8046,  13.8947],
])
# singlet
pbevee1 = np.array([
[0.24345893,   0.38347330],
[0.23129586,   0.37169999],
[0.19070026,   0.31944233],
[0.17236905,   0.28345621],
[0.16832478,   0.27841971],
[0.15626965,   0.23473187],
]) * 13.605662285137
# triplet
pbevee2 = np.array([
[0.20981065, 0.30433015],
[0.19227222, 0.27842814],
[0.16822887, 0.25917961],
[0.15693438, 0.24174941],
[0.15530267, 0.24145352],
[0.15004643, 0.22379908],
]) * 13.605662285137

#
ddhcells = np.array([54, 64, 128, 216, 250, 512])
# KS orbital levels
ddhlevels = np.array([
[6.4325,   9.7066,  15.5028,  17.1154],
[6.4372,   9.9159,  15.4525,  16.9736],
[6.5418,  10.0616,  15.2515,  16.8016],
[6.5641,  10.1404,  15.1744,  16.6253],
[6.5972,  10.1392,  15.1500,  16.5858],
[6.5957,  10.1755,  15.1108,  16.2541],
])
# singlet
ddhvee1 = np.array([
[0.33034870,   0.45224022],
[0.32023619,   0.43116656],
[0.30539692,   0.40844187],
[0.30506974,   0.39273489],
[0.30484900,   0.39101599],
[0.31127465,   0.37858380],
]) * 13.605662285137
# triplet
ddhvee2 = np.array([
[0.30560467, 0.32710401],
[0.29297110, 0.29949725],
[0.29145185, 0.28881069],
[0.29563004, 0.27881490],
[0.29682947, 0.27999923],
[0.30655133, 0.28040133],
]) * 13.605662285137

########
# Plot #
########

fig, ax = plt.subplots(2, 3, figsize=(11,7))

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
linestyles = ['', '', '-', ':', '-', ':']
labels1 = ['$^3A_{2u}$', '$^3E_{u}$', '$^3E_{u}$', '$^3A_{1u}$']
labels2 = ['$^3E_{g}$', '$^3E_{g}$']
labels3 = ['$^3A_{1g}$', '$^3E_{g}^\prime$', '$^3E_{g}^\prime$', '$^3A_{2g}$']

# VEE as a function of L^3
for i in range(1):
    ax[0][0].plot(1/np.power(pbecells, 1/3) * 2 / 4.26, pbevee1[:,i],
                  linewidth=1.5, linestyle=linestyles[0], marker='s',
                  markersize=6, color=colors[i], label='')
    ###
    x = 1/(np.power(pbecells, 1/3))**3 * 2 / 4.26
    y = pbevee1[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.linalg.lstsq(A, y, rcond=None)[1]
    r2 = 1 - resid / (y.size * y.var())
    print('pbe', '$E_{\mathrm{VE}}$', m, c, r2)
    xaxis = np.linspace(0,0.003,101)
    ###

    ax[1][0].plot(1/(np.power(ddhcells, 1/3)) * 2 / 4.19, ddhvee1[:,i],
                  linewidth=1.5, linestyle=linestyles[0], marker='s',
                  markersize=6, color=colors[i],)# label=labels2[i])

    ###
    x = 1/(np.power(ddhcells, 1/3))**3 * 2 / 4.19
    y = ddhvee1[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.linalg.lstsq(A, y, rcond=None)[1]
    r2 = 1 - resid / (y.size * y.var())
    print('ddh', '$E_{\mathrm{VE}}$', m, c, r2)
    xaxis = np.linspace(0,0.003,101)
    ###


# energy difference between KS orbital levels
labels5 = ['$a_{1g} \\to$ CBM']
for i in range(1):
    ax[0][1].plot(1/(np.power(pbecells, 1/3))**3 * (2 / 4.26)**3, pbelevels[:,2] - pbelevels[:,1],
                  linewidth=0, marker='o', markersize=6,
                  color=colors[i], label='')
    ###
    x = 1/(np.power(pbecells, 1/3))**3 * (2 / 4.26)**3
    y = pbelevels[:,2] - pbelevels[:,1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.linalg.lstsq(A, y, rcond=None)[1]
    r2 = 1 - resid / (y.size * y.var())
    print('pbe', '$e_{e_{g}} - e_{\mathrm{VBM}}$', m, c, r2)
    xaxis = np.linspace(0,0.02,101)
    ax[0][1].plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i])
    ediffm_pbe = m
    ediffc_pbe = c
    ###
    print('pbe KS diff slope', m)
    print('pbe KS diff int', c)

    ax[1][1].plot(1/(np.power(ddhcells, 1/3))**3 * (2 / 4.19)**3, ddhlevels[:,2] - ddhlevels[:,1],
                  linewidth=0, marker='o', markersize=6,
                  color=colors[i], label=labels5[i])
    ###
    x = 1/(np.power(ddhcells, 1/3))**3 * (2 / 4.19)**3
    y = ddhlevels[:,2] - ddhlevels[:,1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.linalg.lstsq(A, y, rcond=None)[1]
    r2 = 1 - resid / (y.size * y.var())
    print('ddh', '$e_{e_{g}} - e_{\mathrm{VBM}}$', m, c, r2)
    xaxis = np.linspace(0,0.02,101)
    ax[1][1].plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i])
    ediffm = m
    ediffc = c
    print('ddh KS diff slope', m)
    print('ddh KS diff int', c)
    ###


# Exciton binding energy
for i in range(1):
    ax[0][2].plot(1/(np.power(pbecells, 1/3)) * 2 / 4.26,
                  pbelevels[:,2] - pbelevels[:,1] - pbevee1[:,i],
                  linewidth=0, marker='o', markersize=6,
                  color=colors[i], label='')
    ###
    x = 1/(np.power(pbecells, 1/3)) * 2 / 4.26
    y = pbelevels[:,2] - pbelevels[:,1] - pbevee1[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.linalg.lstsq(A, y, rcond=None)[1]
    r2 = 1 - resid / (y.size * y.var())
    print('pbe', '$e_{e_{g}} - e_{\mathrm{VBM}} - E_{\mathrm{VE}}$', m, c, r2)
    xaxis = np.linspace(0,0.3,101)
    ###

    def funclinear(x, m):
        return x * m

    popt, pcov = curve_fit(funclinear, x, y)
    pbe_nebm0 = popt[0]
    ax[0][2].plot(xaxis, funclinear(xaxis, pbe_nebm0), linestyle='--',
                  c=colors[0], label='')
    print(pbe_nebm0)


    ax[1][2].plot(1/(np.power(ddhcells, 1/3)) * 2 / 4.19,
                  ddhlevels[:,2] - ddhlevels[:,1] - ddhvee1[:,i],
                  linewidth=0, marker='o', markersize=6,
                  color=colors[i])#, label=labels5[i])
    ###
    x = 1/(np.power(ddhcells, 1/3)) * 2 / 4.19
    y = ddhlevels[:,2] - ddhlevels[:,1] - ddhvee1[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.linalg.lstsq(A, y, rcond=None)[1]
    r2 = 1 - resid / (y.size * y.var())
    print('ddh', '$e_{e_{g}} - e_{\mathrm{VBM}} - E_{\mathrm{VE}}$', m, c, r2)
    xaxis = np.linspace(0,0.3,101)
    ebm = m
    ebc = c

    def func1(x, c):
        return x * 5.058167592667465 / (2 / 4.19) * np.exp(- 1 / x / 21) + c

    def func4(x, c):
        return x * 5.058167592667465 / (2 / 4.19) * np.exp(- 1 / x / 42) + c


    popt, pcov = curve_fit(funclinear, x, y)
    nebm0 = popt[0]
    ax[1][2].plot(xaxis, funclinear(xaxis, nebm0), linestyle='--',
                  c=colors[0], label='$D=\infty$')
    print(nebm0)

    popt, pcov = curve_fit(func4, x, y)
    nebc4 = popt[0]
    ax[1][2].plot(xaxis, func4(xaxis, nebc4), linestyle='--',
                  c=colors[1], label='$D=42$ Å')

    popt, pcov = curve_fit(func1, x, y)
    nebc1 = popt[0]
    ax[1][2].plot(xaxis, func1(xaxis, nebc1), linestyle='--',
                  c=colors[2], label='$D=21$ Å')
    ###

print('pbe')
print(pbe_nebm0)
print(ediffc_pbe)

print('ddh')
xaxis = np.linspace(0,0.03,100001)
#
ax[0][0].plot(xaxis**(1/3), ediffm_pbe*xaxis - pbe_nebm0 * xaxis**(1/3) + ediffc_pbe,
              linestyle='--', color=colors[0], label='')
#

ax[1][0].plot(xaxis**(1/3), ediffm*xaxis - nebm0 * xaxis**(1/3) + ediffc,
              linestyle='--', color=colors[0], label='$D=\infty$')

ax[1][0].plot(xaxis**(1/3), ediffm*xaxis + ediffc - func4(xaxis**(1/3), nebc4),
              linestyle='--', c=colors[1], label='$D=42$ Å')

ax[1][0].plot(xaxis**(1/3), ediffm*xaxis + ediffc - func1(xaxis**(1/3), nebc1),
              linestyle='--', color=colors[2], label='$D=21$ Å')

print(ediffc)
print(ediffc - nebc1)
print(ediffc - nebc4)


ax[0][0].set_title('$E_{\mathrm{VE}}$')
ax[0][1].set_title('$e_{\mathrm{CBM}} - e_{a_{1g}}$')
ax[0][2].set_title('$e_{\mathrm{CBM}} - e_{a_{1g}} - E_{\mathrm{VE}}$')
ax[1][0].set_title('$E_{\mathrm{VE}}$')
ax[1][1].set_title('$e_{\mathrm{CBM}} - e_{a_{1g}}$')
ax[1][2].set_title('$e_{\mathrm{CBM}} - e_{a_{1g}} - E_{\mathrm{VE}}$')


ax[0][0].text(x=-0.3, y=1.03, s='(a)', fontsize=15,
              transform=ax[0][0].transAxes)
ax[0][1].text(x=-0.3, y=1.03, s='(b)', fontsize=15,
              transform=ax[0][1].transAxes)
ax[0][2].text(x=-0.3, y=1.03, s='(c)', fontsize=15,
              transform=ax[0][2].transAxes)
ax[1][0].text(x=-0.3, y=1.03, s='(d)', fontsize=15,
              transform=ax[1][0].transAxes)
ax[1][1].text(x=-0.3, y=1.03, s='(e)', fontsize=15,
              transform=ax[1][1].transAxes)
ax[1][2].text(x=-0.3, y=1.03, s='(f)', fontsize=15,
              transform=ax[1][2].transAxes)


for i in range(2):
    for j in range(3):
        if i==0 or i==1:
            if j==0: ax[i][j].set_xlabel('$\\frac{1}{L}$ (Å$^{-1})$')
            if j==1: ax[i][j].set_xlabel('$\\frac{1}{L^3}$ (Å$^{-3})$')
            if j==2: ax[i][j].set_xlabel('$\\frac{1}{L}$ (Å$^{-1})$')

        if i==1 and j==2: ax[i][j].legend(fontsize=12,loc='best',edgecolor='black')
        if i==1 and j==0: ax[i][j].legend(fontsize=12,loc='best',edgecolor='black')

        if j==0: ax[i][j].set_ylabel('$E$ (eV)', color = 'black')
        ax[i][j].spines['left'].set_color('black')
        ax[i][j].spines['right'].set_color('black')
        ax[i][j].tick_params(axis='y', colors='black')
        ax[i][j].tick_params(axis='both', direction='in')
        ax[i][j].tick_params(which='minor', direction='in')
        ax[i][j].xaxis.set_ticks_position('both')
        ax[i][j].yaxis.set_ticks_position('both')

        if j==1: ax[i][j].set_xticks([0, 0.001, 0.002])

        if j==1: ax[i][j].set_xlim([0, 0.002])
        if j==0 or j==2: ax[i][j].set_xlim([0, 0.15])
        if i==0:
            if j==0: ax[i][j].set_ylim([1.7, 4.1])
            elif j==1: ax[i][j].set_ylim([1.7, 4.1])
        if i==1:
            if j==0: ax[i][j].set_ylim([3.9, 5.1])
            elif j==1: ax[i][j].set_ylim([4.7, 5.9])
        if i==0 and j==2: ax[i][j].set_ylim([-0.85, 0.85])
        if i==1 and j==2: ax[i][j].set_ylim([-0.1, 1.6])

fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.savefig("Fig-S6.pdf",bbox_inches = 'tight',dpi=300)
plt.show()
