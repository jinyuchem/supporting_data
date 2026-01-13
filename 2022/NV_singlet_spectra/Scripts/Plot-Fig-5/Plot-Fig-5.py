#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 14})
from matplotlib.gridspec import GridSpec
import os
from matplotlib.cbook import get_sample_data
from sklearn.metrics import auc
import os

current_working_directory = os.getcwd()

#############
# Functions #
#############

def Gaussian(x,mu,sigma):
    prefix=1/np.sqrt(2*np.pi*sigma**2)
    eeee=np.exp( -(x-mu)**2/(2*sigma**2) )
    return prefix*eeee

def sigma(freq):
    f = 6 - (6 - 2) * freq / 160
    return f

########
# Main #
########

fnames = [
         'DDH-dq-PBE-ph-All-B-3A2-3E-13823-rc2-5-cph-5.dat',
         'DDH-dq-PBE-ph-All-B-1A1-1E-13823-rc2-5-cph-5.dat',
         ]

bf = []
freq = []
for i in range(2):
    bf.append(np.loadtxt(fnames[i], usecols=3))
    freq.append(np.loadtxt(fnames[i], usecols=6) * 1000/8065.544)

# deg?
wd = [
      np.zeros(freq[0].shape[0]),
      np.zeros(freq[1].shape[0]),
     ]
for i in range(2):
    for j in range(1,freq[i].shape[0]):
        if abs(freq[i][j] - freq[i][j-1]) < 1e-5:
            wd[i][j] = 1
            wd[i][j-1] = 1

bf_a = [
        np.zeros(freq[0].shape[0]),
        np.zeros(freq[1].shape[0]),
       ]
bf_e = [
        np.zeros(freq[0].shape[0]),
        np.zeros(freq[1].shape[0]),
       ]

for i in range(2):
    bf_e[i][:] = bf[i][:] * wd[i][:]
    bf_a[i][:] = bf[i][:] - bf_e[i][:]

reso = 2001

DATA = np.zeros((2,reso))
DATA_A = np.zeros((2,reso))
DATA_E = np.zeros((2,reso))

ENEAXIS = np.linspace(0, 200, reso, endpoint = True )

for i in range(reso):
    DATA[0,i] = sum(bf[0][:]**2 * Gaussian(ENEAXIS[i], freq[0][:], sigma(freq[0][:])))
    DATA[1,i] = sum(bf[1][:]**2 * Gaussian(ENEAXIS[i], freq[1][:], sigma(freq[1][:])))

    DATA_A[0,i] = sum(bf_a[0][:]**2 * Gaussian(ENEAXIS[i], freq[0][:], sigma(freq[0][:])))
    DATA_A[1,i] = sum(bf_a[1][:]**2 * Gaussian(ENEAXIS[i], freq[1][:], sigma(freq[1][:])))

    DATA_E[0,i] = sum(bf_e[0][:]**2 * Gaussian(ENEAXIS[i], freq[0][:], sigma(freq[0][:])))
    DATA_E[1,i] = sum(bf_e[1][:]**2 * Gaussian(ENEAXIS[i], freq[1][:], sigma(freq[1][:])))

########
# Plot #
########

fig = plt.figure(figsize=(12, 7))

gs = GridSpec(nrows=6, ncols=2, height_ratios=[1, 1, 1, 1, 1, 1],
                                width_ratios=[1, 1],
                                hspace=1.6, wspace=0.4,
                                left=0.05, right=0.9,
                                bottom=0.02, top=0.98)

ax00 = fig.add_subplot(gs[0:3,1])
ax10 = fig.add_subplot(gs[3:6,1])

ax02 = fig.add_subplot(gs[0:3,0])
ax12 = fig.add_subplot(gs[3:6,0])

# plot a
ylim = 1.

colors = ['#DB4437', '#4285F4', '#F4B400', '#0F9D58']
linestyles = ['-', '-', '-', '--']
labels = [
         'Total',
         '$a_1$ phonons',
         '$e$ phonons'
         ]

##########################
# fig-11
ax00.plot(ENEAXIS[:], DATA_A[0], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[1])
ax00.plot(ENEAXIS[:], DATA_E[0], linewidth=1.5, linestyle=linestyles[2],
           color=colors[2], label=labels[2])
ax00.plot(ENEAXIS[:], DATA[0], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])

ax00.legend(fontsize=15,loc='upper right',edgecolor='black',
              framealpha=1,
              labelspacing=0.2, handlelength=1.0, handleheight=0.5,
              handletextpad=0.5, columnspacing=1.0, borderaxespad=0.3)

ax00.set_ylim((0,ylim/10))
ax00.set_xlim((0,200))
ax00.set_xlabel('$\hbar \omega$ (meV)')
ax00.set_ylabel('S($\hbar \omega$) (meV$^{-1}$)', color='black')
ax00.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])

ax00.tick_params(axis='both', direction='in')
ax00.tick_params(which='minor', direction='in')
ax00.xaxis.set_ticks_position('both')
ax00.yaxis.set_ticks_position('both')

ax00.text(x=-40, y=ylim/10*1.04, s='c', fontsize=15, weight='bold')

ax00.text(x=10, y=ylim/10*0.85, s='$^3E \\to ^3A_2$', fontsize=15)

##########################
ylim = 0.4

# fig-21
ax10.plot(ENEAXIS[:], DATA[1], linewidth=1.5, linestyle=linestyles[0],
           color=colors[0], label=labels[0])
ax10.plot(ENEAXIS[:], DATA_A[1], linewidth=1.5, linestyle=linestyles[1],
           color=colors[1], label=labels[1])
ax10.plot(ENEAXIS[:], DATA_E[1], linewidth=1.5, linestyle=linestyles[2],
           color=colors[2], label=labels[2])

ax10.set_ylim((0,ylim/10))
ax10.set_xlim((0,200))
ax10.set_xlabel('$\hbar \omega$ (meV)')
ax10.set_ylabel('S($\hbar \omega$) (meV$^{-1}$)', color='black')
ax10.set_yticks([0.0,  0.01, 0.02, 0.03, 0.04])

ax10.tick_params(axis='both', direction='in')
ax10.tick_params(which='minor', direction='in')
ax10.xaxis.set_ticks_position('both')
ax10.yaxis.set_ticks_position('both')

ax10.text(x=-40, y=ylim/10*1.04, s='d', fontsize=15, weight='bold')

ax10.text(x=10, y=ylim/10*0.85, s='$^1E \\to ^1A_1$', fontsize=15)


cwd = os.getcwd()
im1 = plt.imread(get_sample_data(current_working_directory + '/3A2-mode-60.png'))
newax = fig.add_axes([0.58, 0.8, 0.17, 0.17], anchor='NE')
newax.imshow(im1)
newax.axis('off')

ax00.annotate("",
              xy=(60, 0.067), xycoords='data',
              xytext=(90, 0.09), textcoords='data',
              arrowprops=dict(arrowstyle="->", color='black',
                              shrinkA=0, shrinkB=0,
                              patchA=None, patchB=None, lw=1,
                              mutation_scale=15, ls='-'
                              ),
              annotation_clip=False
              )

im1 = plt.imread(get_sample_data(current_working_directory + '/3A2-mode-162.png'))
newax = fig.add_axes([0.717, 0.64, 0.16, 0.16], anchor='NE')
newax.imshow(im1)
newax.axis('off')

ax00.annotate("",
              xy=(163, 0.01), xycoords='data',
              xytext=(163, 0.02), textcoords='data',
              arrowprops=dict(arrowstyle="->", color='black',
                              shrinkA=0, shrinkB=0,
                              patchA=None, patchB=None, lw=1,
                              mutation_scale=15, ls='-'
                              ),
              annotation_clip=False
              )



im1 = plt.imread(get_sample_data(current_working_directory + '/1A1-mode-73.png'))
newax = fig.add_axes([0.64, 0.27, 0.15, 0.15], anchor='NE')
newax.imshow(im1)
newax.axis('off')

ax10.annotate("",
              xy=(73, 0.026), xycoords='data',
              xytext=(100, 0.035), textcoords='data',
              arrowprops=dict(arrowstyle="->", color='black',
                              shrinkA=0, shrinkB=0,
                              patchA=None, patchB=None, lw=1,
                              mutation_scale=15, ls='-'
                              ),
              annotation_clip=False
              )


im1 = plt.imread(get_sample_data(current_working_directory + '/1A1-mode-170.png'))
newax = fig.add_axes([0.744, 0.17, 0.15, 0.15], anchor='NE')
newax.imshow(im1)
newax.axis('off')

ax10.annotate("",
              xy=(170, 0.009), xycoords='data',
              xytext=(170, 0.015), textcoords='data',
              arrowprops=dict(arrowstyle="->", color='black',
                              shrinkA=0, shrinkB=0,
                              patchA=None, patchB=None, lw=1,
                              mutation_scale=15, ls='-'
                              ),
              annotation_clip=False
              )

######
# PL #
######

fnames = [
'DDH-dq-PBE-ph-PL-All-B-3A2-3E-13823-rc2-5-cph-5.dat',
]

# energy of the ZPL: meV
EZPL = 1945

eneaxis = np.loadtxt(fnames[0], usecols=0)
lsp = []
for i in range(len(fnames)):
    p = np.loadtxt(fnames[i], usecols=1)
    lsp.append(p)
lsp = np.array(lsp)

# \omega^3 prefactor
for i in range(len(fnames)):
    lsp[i,:] = lsp[i,:] * (1945 - eneaxis[:])**3
    # normalization
    lsp[i] = lsp[i] / sum(lsp[i]) / (eneaxis[1] - eneaxis[0]) * 1000

# Experimental data
fname_exp = 'NV-3E-3A2-exp.csv'
r_exp_eneaxis = np.loadtxt(open(fname_exp, "rb"), delimiter=",",
                         usecols=1, skiprows=1) * 1000 - 1946.34
r_exp_lsp = np.loadtxt(open(fname_exp, "rb"), delimiter=",",
                         usecols=2, skiprows=1)

exp_eneaxis = r_exp_eneaxis[1544:14046]
exp_lsp = r_exp_lsp[1544:14046]

# normalization
exp_lsp = exp_lsp / sum(exp_lsp) / (r_exp_eneaxis[1] - r_exp_eneaxis[0]) * 1000

colors = ['#DB4437', '#F4B400', '#0F9D58']
labels = [
           'DDH$-\Delta Q$, PBE$-ph$',
         ]
linestyles = ['-', '-', '-', '--', '--', '--']

for i in range(len(fnames)):
    ax02.plot(-eneaxis, lsp[i], color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i])
ax02.fill_between(exp_eneaxis, exp_lsp, color='gray', label='Expt.$^a$', alpha=0.4)

ax02.set_xlim((-400,50))
ax02.set_ylim((0,7))
ax02.set_yticklabels([])
ax02.set_xticks([-400,-300,-200,-100,0])
ax02.tick_params(direction='in')
ax02.xaxis.set_ticks_position('both')
ax02.yaxis.set_ticks_position('both')
ax02.set_xlabel( "Energy Shift from ZPL (meV)" )
ax02.set_ylabel( "Photoluminescence (arb. unit)" )

ax02.text(x=(50 - -50)*5/9 + -400, y=7*0.85, s='$^3E \\to ^3A_2$', fontsize=15)

ax02.text(x=-450, y=7*1.04, s='a', fontsize=15, weight='bold')

#######
# Abs #
#######

fnames = [
         'DDH-dq-PBE-ph-Abs-All-B-1A1-1E-13823-rc2-5-cph-5.dat',
         ]

# energy of the ZPL: meV
EZPL = 1190

eneaxis = np.loadtxt(fnames[0], usecols=0)
lsp = []
for i in range(len(fnames)):
    p = np.loadtxt(fnames[i], usecols=1)
    lsp.append(p)
lsp = np.array(lsp)

# \omega prefactor
for i in range(len(fnames)):
    lsp[i,:] = lsp[i,:] * (EZPL + eneaxis[:])
    # normalization
    lsp[i] = lsp[i] / auc(eneaxis,lsp[i]) * 1000

# Experimental data
fname_exp = 'NV-1E-1A1-exp.csv'
r_exp_eneaxis = np.loadtxt(open(fname_exp, "rb"), delimiter=",",
                         usecols=0, skiprows=1)
r_exp_lsp = np.loadtxt(open(fname_exp, "rb"), delimiter=",",
                         usecols=1, skiprows=1)

exp_eneaxis = r_exp_eneaxis[:]
exp_lsp = r_exp_lsp[:] * (EZPL + r_exp_eneaxis[:])

# normalization
nn = auc(exp_eneaxis,exp_lsp)
nn = nn / 0.6
exp_lsp = exp_lsp / nn * 1000

labels = [
          '$^1E \\to ^1A_1$, DDH$-\Delta Q$, PBE$-ph$',
         ]

for i in range(len(fnames)):
    ax12.plot(eneaxis, lsp[i], color=colors[i], label='', linewidth=2, linestyle=linestyles[i])
ax12.fill_between(exp_eneaxis, exp_lsp, color='gray', label='Expt.$^b$', alpha=0.4)

ax12.set_xlim((-50,400))
ax12.set_xticks([0, 100, 200, 300, 400])
ax12.set_ylim((0,12))
ax12.set_yticklabels([])
ax12.tick_params(direction='in')
ax12.xaxis.set_ticks_position('both')
ax12.yaxis.set_ticks_position('both')
ax12.set_xlabel( "Energy Shift from ZPL (meV)" )
ax12.set_ylabel( "Absorption (arb. unit)" )

ax12.text(x=(50 - -50)*5/9 + -50, y=12*0.85, s='$^1E \\to ^1A_1$', fontsize=15)

ax12.text(x=-100, y=12*1.04, s='b', fontsize=15, weight='bold')


plt.savefig("Fig-5.pdf", bbox_inches='tight', dpi=300)
plt.show()
