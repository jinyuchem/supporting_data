#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 12})
from scipy.optimize import curve_fit

#
pbecells = np.array([64, 216, 512, 1000])
pbevee1 = np.array([
[0.167660, 0.172234, 0.172234, 0.183848],
[0.122200, 0.129804, 0.129804, 0.147560],
[0.102586, 0.105868, 0.105868, 0.111546],
[0.088018, 0.088963, 0.088963, 0.090416],
]) * 13.605662285137

pbevee2 = np.array([
[0.120178, 0.120179],
[0.073590, 0.073590],
[0.062232, 0.062232],
[0.058042, 0.058043],
]) * 13.605662285137

pbevee3 = np.array([
[0.161742, 0.172457, 0.172457, 0.195814],
[0.090000, 0.092202, 0.092202, 0.102852],
[0.068892, 0.069922, 0.069922, 0.075747],
[0.061200, 0.061733, 0.061733, 0.064784]
]) * 13.605662285137

#
ddhcells = np.array([64, 216, 512, 1000])
ddhvee1 = np.array([
[0.180872, 0.185553, 0.185553, 0.197631],
[0.129695, 0.136306, 0.136306, 0.156029],
[0.119244, 0.124712, 0.124712, 0.139264],
[0.11404, 0.11890, 0.11898, 0.12892],
]) * 13.605662285137

ddhvee2 = np.array([
[0.136749, 0.136749],
[0.099014, 0.099014],
[0.096890, 0.096890],
[0.09956, 0.09956],
]) * 13.605662285137

ddhvee3 = np.array([
[0.166309, 0.174865, 0.174865, 0.195600],
[0.111920, 0.113750, 0.113750, 0.122764],
[0.102067, 0.102986, 0.102986, 0.107420],
[0.10302, 0.10375, 0.10375, 0.10613],
]) * 13.605662285137


fname = 'KS-levels-GS-Geo.txt'

PBEup = np.loadtxt(fname, skiprows=1, usecols=2, max_rows=16)
PBEdown = np.loadtxt(fname, skiprows=1, usecols=3, max_rows=16)
DDHup = np.loadtxt(fname, skiprows=1, usecols=4, max_rows=16)
DDHdown = np.loadtxt(fname, skiprows=1, usecols=5, max_rows=16)

PBEup = np.reshape(PBEup, (4, 4))
PBEdown = np.reshape(PBEdown, (4, 4))
DDHup = np.reshape(DDHup, (4,4))
DDHdown = np.reshape(DDHdown, (4,4))

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
    ax[0][0].plot(1/(np.power(pbecells/8, 1/3) * 3.568), pbevee2[:,i],
                  linewidth=1.5, linestyle=linestyles[0], marker='s',
                  markersize=6, color=colors[i], label='$')
    ###
    x = 1/(np.power(pbecells/8, 1/3) * 3.568)**3
    y = pbevee2[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.linalg.lstsq(A, y, rcond=None)[1]
    r2 = 1 - resid / (y.size * y.var())
    print('pbe', '$E_{\mathrm{VE}}$', m, c, r2)
    xaxis = np.linspace(0,0.003,101)
    ###

    ax[1][0].plot(1/(np.power(ddhcells/8, 1/3) * 3.55), ddhvee2[:,i],
                  linewidth=1.5, linestyle=linestyles[0], marker='s',
                  markersize=6, color=colors[i],)# label=labels2[i])

    ###
    x = 1/(np.power(ddhcells/8, 1/3) * 3.55)**3
    y = ddhvee2[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.linalg.lstsq(A, y, rcond=None)[1]
    r2 = 1 - resid / (y.size * y.var())
    print('ddh', '$E_{\mathrm{VE}}$', m, c, r2)
    xaxis = np.linspace(0,0.003,101)
    ###

# energy difference between KS orbital levels
labels5 = ['VBM $\\to\ e_g$']
for i in range(1):
    ax[0][1].plot(1/(np.power(pbecells/8, 1/3) * 3.568)**3, PBEdown[:,2] - PBEdown[:,i+1],
                  linewidth=0, marker='o', markersize=6,
                  color=colors[i], label='')
    ###
    x = 1/(np.power(pbecells/8, 1/3) * 3.568)**3
    y = PBEdown[:,2] - PBEdown[:,i+1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.linalg.lstsq(A, y, rcond=None)[1]
    r2 = 1 - resid / (y.size * y.var())
    print('pbe', '$e_{e_{g}} - e_{\mathrm{VBM}}$', m, c, r2)
    xaxis = np.linspace(0,0.003,101)
    ax[0][1].plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i])
    ediffm_pbe = m
    ediffc_pbe = c
    ###

    ax[1][1].plot(1/(np.power(ddhcells/8, 1/3) * 3.55)**3, DDHdown[:,2] - DDHdown[:,i+1],
                  linewidth=0, marker='o', markersize=6,
                  color=colors[i], label=labels5[i])
    ###
    x = 1/(np.power(ddhcells/8, 1/3) * 3.55)**3
    y = DDHdown[:,2] - DDHdown[:,i+1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.linalg.lstsq(A, y, rcond=None)[1]
    r2 = 1 - resid / (y.size * y.var())
    print('ddh', '$e_{e_{g}} - e_{\mathrm{VBM}}$', m, c, r2)
    xaxis = np.linspace(0,0.003,101)
    ax[1][1].plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i])
    ediffm = m
    ediffc = c
    ###

# Exciton binding energy
for i in range(1):
    ax[0][2].plot(1/(np.power(pbecells/8, 1/3) * 3.568), PBEdown[:,2] - PBEdown[:,i+1] - pbevee2[:,i],
                  linewidth=0, marker='o', markersize=6,
                  color=colors[i], label='')
    ###
    x = 1/(np.power(pbecells/8, 1/3) * 3.568)
    y = PBEdown[:,2] - PBEdown[:,i+1] - pbevee2[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.linalg.lstsq(A, y, rcond=None)[1]
    r2 = 1 - resid / (y.size * y.var())
    print('pbe', '$e_{e_{g}} - e_{\mathrm{VBM}} - E_{\mathrm{VE}}$', m, c, r2)
    xaxis = np.linspace(0,0.15,101)
    ax[0][2].plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i])
    print('pbe eb slope', m)
    print('pbe eb int', c)
    ###

    ax[1][2].plot(1/(np.power(ddhcells/8, 1/3) * 3.55), DDHdown[:,2] - DDHdown[:,i+1] - ddhvee2[:,i],
                  linewidth=0, marker='o', markersize=6,
                  color=colors[i])#, label=labels5[i])
    ###
    x = 1/(np.power(ddhcells/8, 1/3) * 3.55)
    y = DDHdown[:,2] - DDHdown[:,i+1] - ddhvee2[:,i]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.linalg.lstsq(A, y, rcond=None)[1]
    r2 = 1 - resid / (y.size * y.var())
    print('ddh', '$e_{e_{g}} - e_{\mathrm{VBM}} - E_{\mathrm{VE}}$', m, c, r2)
    xaxis = np.linspace(0,0.15,101)
    ax[1][2].plot(xaxis, xaxis * m + c, linestyle='--', c=colors[i], label='$D=\infty$')
    ebm = m
    ebc = c
    print('ddh eb slope', m)
    print('ddh eb int', c)

    def func1(x, m, c):
        return x * m * np.exp(- 1 / x / 20) + c

    def func2(x, m, c):
        return x * m * np.exp(- 1 / x / 37.6) + c

    def func3(x, m, c):
        return x * m * np.exp(- 1 / x / 10) + c

    def func4(x, m, c):
        return x * m * np.exp(- 1 / x / 40) + c

    print('screened eb slope')
    popt, pcov = curve_fit(func4, x, y)
    nebm4 = popt[0]
    nebc4 = popt[1]
    ax[1][2].plot(xaxis, func4(xaxis, nebm4, nebc4), linestyle='--',
                  c=colors[1], label='$D=40$ Å')
    print(40, nebm4)

    popt, pcov = curve_fit(func1, x, y)
    nebm1 = popt[0]
    nebc1 = popt[1]
    ax[1][2].plot(xaxis, func1(xaxis, nebm1, nebc1), linestyle='--',
                  c=colors[2], label='$D=20$ Å')
    print(20, nebm1)

    popt, pcov = curve_fit(func2, x, y)
    nebm2 = popt[0]
    nebc2 = popt[1]
    print(37.6, nebm2)

    popt, pcov = curve_fit(func3, x, y)
    nebm3 = popt[0]
    nebc3 = popt[1]
    ax[1][2].plot(xaxis, func3(xaxis, nebm3, nebc3), linestyle='--',
                  c=colors[3], label='$D=10$ Å')
    ###
    print(10, nebm3)


xaxis = np.linspace(0,0.02,100001)
#
ax[0][0].plot(xaxis**(1/3), ediffm_pbe*xaxis + ediffc_pbe,
              linestyle='--', color=colors[0], label='')
print('pbe slope')
print(ediffm_pbe)
print('pbe vee')
print(ediffc_pbe)
#

ax[1][0].plot(xaxis**(1/3), ediffm*xaxis - ebm * xaxis**(1/3) + ediffc,
              linestyle='--', color=colors[0], label='$D=\infty$')

ax[1][0].plot(xaxis**(1/3), ediffm*xaxis + ediffc - func4(xaxis**(1/3), nebm4, nebc4),
              linestyle='--', c=colors[1], label='$D=40$ Å')

ax[1][0].plot(xaxis**(1/3), ediffm*xaxis + ediffc - func1(xaxis**(1/3), nebm1, nebc1),
              linestyle='--', color=colors[2], label='$D=20$ Å')


ax[1][0].plot(xaxis**(1/3), ediffm*xaxis + ediffc - func3(xaxis**(1/3), nebm3, nebc3),
              linestyle='--', c=colors[3], label='$D=10$ Å')

print('ddh slope')
print(ediffm)
print('ddh vee')
print(ediffc)
print(ediffc - nebc1)
print(ediffc - nebc2)
print(ediffc - nebc3)
print(ediffc - nebc4)

ax[0][0].set_title('$E_{\mathrm{VE}}$')
ax[0][1].set_title('$e_{e_{g}} - e_{\mathrm{VBM}}$')
ax[0][2].set_title('$e_{e_{g}} - e_{\mathrm{VBM}} - E_{\mathrm{VE}}$')
ax[1][0].set_title('$E_{\mathrm{VE}}$')
ax[1][1].set_title('$e_{e_{g}} - e_{\mathrm{VBM}}$')
ax[1][2].set_title('$e_{e_{g}} - e_{\mathrm{VBM}} - E_{\mathrm{VE}}$')

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
            if j==0: ax[i][j].set_xlabel('$\\frac{1}{L}$ (Å$^{-1}$)')
            if j==1: ax[i][j].set_xlabel('$\\frac{1}{L^3}$ (Å$^{-3}$)')
            if j==2: ax[i][j].set_xlabel('$\\frac{1}{L}$ (Å$^{-1}$)')

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

        if j==1: ax[i][j].set_xlim([0, 0.003])
        if j==0 or j==2: ax[i][j].set_xlim([0, 0.15])
        if i==0:
            if j==0 or j==1: ax[i][j].set_ylim([0.5, 2.1])
        if i==1:
            if j==0: ax[i][j].set_ylim([1.2, 2.2])
            elif j==1: ax[i][j].set_ylim([1.4, 2.8])
        if j==2: ax[i][j].set_ylim([0, 1.0])

fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.savefig("Fig-S4.pdf",bbox_inches = 'tight',dpi=300)
plt.show()
