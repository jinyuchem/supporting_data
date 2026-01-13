#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import os
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.patches import ConnectionPatch
import numpy.polynomial.hermite as Herm
import math
import os

current_working_directory = os.getcwd()

########
# Plot #
########

fig = plt.figure(figsize=(10, 5))

gs = GridSpec(nrows=2, ncols=4, height_ratios=[1, 0.8],
                                width_ratios=[1.6, 1., 0.05, 2], 
                                hspace=0.1, wspace=0.25,
                                left=0.05, right=0.9, 
                                bottom=0.02, top=0.98)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,:2])

ax = fig.add_subplot(gs[:,3])

######
# NV #
######

ax1.axis('off')
ax1.set_xlim((0,1))
ax1.set_ylim((0,1))
cwd = os.getcwd()
im1 = plt.imread(get_sample_data(current_working_directory + '/NV.png'))
newax = fig.add_axes([-0.12, 0.53, 0.33, 0.33], anchor='NE')
newax.imshow(im1)
newax.axis('off')
ax1.text(x=-0.29, y=1.0, s='a', weight='bold', fontsize=15)

#################
# defect levels #
#################

# DDH
hhPS = np.array([12.6320, 18.3370])
hhup = np.array([13.3071, 14.4298, 14.4298])
hhdown = np.array([14.2228, 17.4324, 17.4324])
hhup = hhup - hhPS[0]
hhdown = hhdown - hhPS[0]

my_ylim = np.array([-0.7, 6.4])

ax2.fill_between(np.linspace(0,1,10), [my_ylim[0] for i in range(10)], [hhPS[0]-hhPS[0] for i in range(10)], color='dodgerblue', alpha=0.3)
ax2.fill_between(np.linspace(0,1,10), [my_ylim[1] for i in range(10)], [hhPS[1]-hhPS[0] for i in range(10)], color='grey', alpha=0.3)
ax2.axvline(x=0.5, ymin=0, ymax=1, color='black', linestyle='--', linewidth=0.5)

ax2.text(x=0.87, y=5.8, s='CB', ha='center')
ax2.text(x=0.87, y=-0.5, s='VB', ha='center')

ax2.axhline(y=hhup[0], xmin=0.19, xmax=0.31, color='black', linestyle='-', linewidth=1.5)
ax2.text(x=0.1, y=hhup[0]+0.45, s='$a_1$', va='center', color='red') 
ax2.axhline(y=hhup[1], xmin=0.31, xmax=0.43, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhup[2], xmin=0.07, xmax=0.19, color='black', linestyle='-', linewidth=1.5)
ax2.text(x=0.05, y=hhup[1]+0.55, s='$e_x$', va='center', color='red')
ax2.text(x=0.3, y=hhup[1]+0.55, s='$e_y$', va='center', color='red')

ax2.axhline(y=hhdown[0], xmin=0.69, xmax=0.81, color='black', linestyle='-', linewidth=1.5)
ax2.text(x=0.6, y=hhdown[0]+0.35, s='$\overline{a}_1$', va='center', color='red')
ax2.axhline(y=hhdown[1], xmin=0.81, xmax=0.93, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=hhdown[2], xmin=0.57, xmax=0.69, color='black', linestyle='-', linewidth=1.5)
ax2.text(x=0.55, y=hhdown[1]+0.35, s='$\overline{e}_x$', va='center', color='red')
ax2.text(x=0.8, y=hhdown[1]+0.35, s='$\overline{e}_y$', va='center', color='red')

ax2.arrow(0.25, hhup[0]-0.23, 0., 0.4, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.13, hhup[1]-0.23, 0., 0.4, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.37, hhup[2]-0.23, 0., 0.4, head_width=0.02, head_length=0.1, color='red')
ax2.arrow(0.75, hhdown[0]+0.23, 0., -0.4, head_width=0.02, head_length=0.1, color='red')

ax2.set_xlim([0,1])
ax2.set_ylim(my_ylim)
ax2.tick_params(direction='in')
ax2.yaxis.set_ticks_position('both')
ax2.set_xticks([])

ax2.set_ylabel("Energy (eV)", labelpad=10, fontsize=12)

ax2.text(x=-0.32, y=1.0 * 7.1 - 0.7, s='b', weight='bold', fontsize=15)

###############
# KS orbitals #
###############

ax3.axis('off')
ax3.set_xlim((0,1))
ax3.set_ylim((0,1))
cwd = os.getcwd()
im2 = plt.imread(get_sample_data(current_working_directory + '/KS-orbitals.png'))
newax = fig.add_axes([-0.155, 0.02, 0.7, 0.25], anchor='NE')
newax.imshow(im2)
newax.axis('off')
ax3.text(x=-0.156, y=0.75, s='c', weight='bold', fontsize=15)
ax3.text(x=0.22, y=0.65, s='$\overline{a}_1$', color='red')
ax3.text(x=0.585, y=0.65, s='$\overline{e}_x$', color='red')
ax3.text(x=0.95, y=0.65, s='$\overline{e}_y$', color='red')

######
# CC #
######

def QUA(x,a,b,c):
    return a * (x - b)**2 + c

def QUA_1(x):
    return QUA(x, 1, 0, 0)

def QUA_2(x):
    return QUA(x, 1, 1, 9)

def QUA_3(x):
    return QUA(x, 1.2, 0.2, 6.8)

def QUA_4(x):
    return QUA(x, 0.8, 0.8, 2.2)

def BACK_L_1(x):
    return -np.sqrt(x)

def BACK_R_1(x):
    return np.sqrt(x)

def BACK_L_2(x):
    return (1.4 - np.sqrt(1.4**2 - 2.8*(7.5-x))) / (1.4)

def BACK_R_2(x):
    return (1.4 + np.sqrt(1.4**2 - 2.8*(7.5-x))) / (1.4)

# gaussian function
def Gaussian(x, mu, sigma):
    pref = 1/np.sqrt(2*np.pi * sigma**2)
    expp = np.exp(-(x - mu)**2 / (2 * sigma**2))
    return pref * expp

# lorentzian function
def Lorentzian(x, mu, gamma):
    pref = 1/(np.pi * gamma)
    mp = gamma**2 / ((x - mu)**2 + gamma**2)
    return pref * mp

#Choose simple units
m=1.
hbar=1.

def hermite(x, w, n):
    xi = np.sqrt(m*w/hbar)*x
    herm_coeffs = np.zeros(n+1)
    herm_coeffs[n] = 1
    return Herm.hermval(xi, herm_coeffs)

def stationary_state(x, w, n):
    xi = np.sqrt(m*w/hbar)*x
    prefactor = 1./np.sqrt(2.**n * math.factorial(n)) * (m*w/(np.pi*hbar))**(0.25)
    psi = prefactor * np.exp(- xi**2 / 2) * hermite(x,w,n)
    return psi

# range of the main plot
xlimit = [-2,5]
ylimit = [-1,12]

# Paraballas
XAXIS_1 = np.linspace(-1.4,1.4,101,endpoint = True)
XAXIS_2 = np.linspace(-0.4,2.4,101,endpoint = True)
XAXIS_3 = np.linspace(-1.2,1.6,101,endpoint = True)
XAXIS_4 = np.linspace(-0.7,2.3,101,endpoint = True)
PRA_1 = np.zeros(101)
PRA_2 = np.zeros(101)
PRA_3 = np.zeros(101)
PRA_4 = np.zeros(101)
for i in range(101):
    PRA_1[i] = QUA_1(XAXIS_1[i])
    PRA_2[i] = QUA_2(XAXIS_2[i])
    PRA_3[i] = QUA_3(XAXIS_3[i])
    PRA_4[i] = QUA_4(XAXIS_4[i])

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Hide ticks
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params( axis='both', which='both', bottom=False,
    top=False, left=False, right=False, labelbottom=False)

# Axes
ax.arrow(x=xlimit[0]+0.1, y=ylimit[0]+0.15, dx=4.8-xlimit[0]-0.15, dy=0,
          color='black', width=0.01, length_includes_head=True,
          head_width=0.2, head_length=0.25, overhang=0.5)
ax.arrow(x=xlimit[0]+0.1, y=ylimit[0]+0.15, dy=ylimit[1]-ylimit[0]-0.15, dx=0,
          color='black', width=0.01, length_includes_head=True,
          head_width=0.15, head_length=0.35, overhang=0.5)

# Paraballes
# 3A2
ax.plot(XAXIS_1, PRA_1, linewidth=2., linestyle='-', color='black')
ax.text(x=-1.8, y=2.3, s='$^3A_2$', fontsize=12, weight='bold')
# 3E
ax.plot(XAXIS_2, PRA_2, linewidth=2., linestyle='-', color='black')
ax.text(x=-0.9, y=11.1, s='$^3E$', fontsize=12, weight='bold')
# 1A1
ax.plot(XAXIS_3, PRA_3, linewidth=2., linestyle='-', color='black')
ax.text(x=-1.3, y=9.5, s='$^1A_1$', fontsize=12, weight='bold')
# 1E
ax.plot(XAXIS_4, PRA_4, linewidth=2., linestyle='-', color='black')
ax.text(x=2.3, y=4.3, s='$^1E$', fontsize=12, weight='bold')


# 3A2 wavefunction
w = 1
for i in range(5):
    x = np.linspace(-3.3,3.3,100)
    y = stationary_state(x, w, i)
    x = x * 0.55
    y = y * 0.27 + 0.2 + 0.4*i
    ax.fill_between(x, 0.2+0.4*i, y, color='lightgray')

# 1A1 wavefunction
w = np.sqrt(1.2)
for i in range(4):
    x = np.linspace(-3.5,3.5,100)
    y = stationary_state(x, w, i)
    x = x * 0.55 + 0.2
    y = y * 0.27 + (0.2 + 0.4*i) * w + 6.8
    ax.fill_between(x, (0.2+0.4*i) * w + 6.8, y, color='lightgray')

# 3E wavefunction
w = 1
for i in range(4):
    x = np.linspace(-3.5,3.5,100)
    y = stationary_state(x, w, i)
    x = x * 0.55 + 1.0
    y = y * 0.24 + (0.2 + 0.4*i) * w + 9
    ax.fill_between(x, (0.2+0.4*i) * w + 9, y, color='lightgray')

# 1E wavefunction
w = np.sqrt(0.8)
for i in range(4):
    x = np.linspace(-3.5,3.5,100)
    y = stationary_state(x, w, i)
    x = x * 0.55 + 0.8
    y = y * 0.24 + (0.2 + 0.4*i) * w + 2.2
    ax.fill_between(x, (0.2+0.4*i)*w + 2.2, y, color='lightgray')

# Arrows for transitions from 3E to 3A2
start = 0.35
stop = 1.0
number_of_lines = 16
cm_subsection = np.linspace(start, stop, number_of_lines)
colors = [ cm.jet(x) for x in cm_subsection ]
for i in range(5):
    ax.plot(1.4-0.13*i, 9+0.2, linewidth=0, marker='o', markersize=3, color=colors[i])
    ax.arrow(x=1.4-0.13*i, y=9+0.2,
          dx=0, dy=-8.8 - 0.2 + 0.4*i + 0.02,
          color=colors[i], width=0.015, length_includes_head=True,
          head_width=0.1, head_length=0.3, overhang=0.6)



## Abs 1E to 1A1
for i in range(4):
    ax.plot(0.46-0.13*i, 2.2+0.2*np.sqrt(0.8), linewidth=0, marker='o', markersize=3, color=colors[-i+10])
    ax.arrow(x=0.46-0.13*i, y=2.2+0.2*np.sqrt(1.2),
          dx=0, dy=4.6 - 0.2*np.sqrt(0.8) + (0.2 + 0.4*i)*np.sqrt(1.2) - 0.08,
          color=colors[-i+10], width=0.015, length_includes_head=True, linestyle='-',
          head_width=0.1, head_length=0.3, overhang=0.6)



ax.set_ylim((ylimit))
ax.set_xlim((xlimit))
ax.set_xlabel('Configuration Coordinate', fontsize=12)
ax.xaxis.set_label_coords(0.5, -0.0)
ax.set_ylabel('Energy', fontsize=12)
ax.tick_params(direction='in')

axins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(3.7, -0.74, 2.5, 3.6+0.2*np.sqrt(1)+0.9+0.1),
                   bbox_transform=ax.transData, loc=2, borderpad=0)
# fc factors
fc_int = np.array([0.03856348227809676,
                   0.1303773051198606,
                   0.2153685650317733,
                   0.2315808611785081,
                   0.18219392072730664,
                   0.11176114658214611,
                   0.055622683129708646,
                   0.023075938606841057,
                   0.008136312497498676,
                   0.002473431064750208,
                   0.0006554043282670026])

for i in range(5):
    axins.axhline(y=0.2+i*0.4, xmin=0.036, xmax=fc_int[i]/max(fc_int)*0.6,
                  color=colors[i], linestyle='-', linewidth=1.3)

# inset limit
inset_xlimit = [0,1]
inset_ylimit = [-0.74,0.9 - 0.74 + 3.6+0.2*np.sqrt(1)+0.1]
axins.set_xlim((inset_xlimit[0],inset_xlimit[1]))
axins.set_ylim((inset_ylimit[0],inset_ylimit[1]))

# PL line shape
eneaxis = np.linspace(-0.5,3.8,1000)
gf = np.zeros(1000)
freq = np.zeros(11)
sigma=0.12
gamma=0.014
for i in range(1,11):
    freq[i] = 0.2 + i*0.4
    gf[:] = gf[:] + fc_int[i] * Gaussian(eneaxis[:], freq[i], sigma)
gf[:] = gf[:] + fc_int[0] * Lorentzian(eneaxis[:], 0.2, gamma)
gf[:] = gf[:] + 0.025
axins.plot(gf/max(gf)/1.2, eneaxis, color='black', linestyle='-', linewidth=1)

# Hide the right and top spines
axins.spines['right'].set_visible(False)
axins.spines['top'].set_visible(False)
axins.spines['left'].set_visible(False)
axins.spines['bottom'].set_visible(False)

# Hide the ticks
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.tick_params( axis='both', which='both', bottom=False,
    top=False, left=False, right=False, labelbottom=False)

# Axes
axins.arrow(x=inset_xlimit[0]+0.025, y=inset_ylimit[1]-0.1, dx=inset_xlimit[1]-inset_xlimit[0]-0.1, dy=0,
          color='black', width=0.001, length_includes_head=True,
          head_width=0.15, head_length=0.1, overhang=0.5)
axins.arrow(x=inset_xlimit[0]+0.025, y=inset_ylimit[1]-0.1, dy=inset_ylimit[0]-inset_ylimit[1]+0.3, dx=0,
          color='black', width=0.001, length_includes_head=True,
          head_width=0.04, head_length=0.4, overhang=0.5)

# Axes labels
axins.set_xlabel('Photoluminescence', fontsize=10)
axins.xaxis.set_label_coords(0.45, 1.02)
axins.xaxis.set_label_position('top')
axins.set_ylabel('Photon Energy', fontsize=10, rotation=270)
axins.yaxis.set_label_coords(-0.15, 0.55)

axins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(3.7, 6.4, 2.5, 3.6+0.2*np.sqrt(1.2)+0.9+0.1),
                   bbox_transform=ax.transData, loc=2, borderpad=0)
# fc factors
fc_int2 = np.array([0.08856348227809676,
                   0.2603773051198606,
                   0.2153685650317733,
                   0.1215808611785081,
                   0.06219392072730664,
                   0.02176114658214611,
                   0.0055622683129708646,
                   0.0023075938606841057,
                   0.008136312497498676,
                   0.002473431064750208,
                   0.0006554043282670026])

for i in range(4):
    axins.axhline(y=6.8+(i+0.5)*0.4*np.sqrt(1.2), xmin=0.036, xmax=fc_int2[i]/max(fc_int2)*0.3,
                  color=colors[-i+10], linestyle='-', linewidth=1.3)

# inset limit
inset_xlimit = [0,1]
inset_ylimit = [6.4,6.4+3.6+0.2*np.sqrt(1.2)+1]
axins.set_xlim((inset_xlimit[0],inset_xlimit[1]))
axins.set_ylim((inset_ylimit[0],inset_ylimit[1]))

# PL line shape 2
eneaxis2 = np.linspace(6.6,10,1000)
gf2 = np.zeros(1000)
freq = np.zeros(11)
sigma=0.12
gamma=0.014
for i in range(1,8):
    freq[i] =  6.8 + (i+0.5)*0.4*np.sqrt(1.2)
    gf2[:] = gf2[:] + fc_int2[i] * Gaussian(eneaxis2[:], freq[i], sigma)
gf2[:] = gf2[:] + fc_int2[0] * Lorentzian(eneaxis2[:], 6.8 + (0.5)*0.4*np.sqrt(1.2), gamma)
gf2[:] = gf2[:] + 0.06
axins.plot(gf2/max(gf2)/1.2, eneaxis2, color='black', linestyle='-', linewidth=1)

# Hide the right and top spines
axins.spines['right'].set_visible(False)
axins.spines['top'].set_visible(False)
axins.spines['left'].set_visible(False)
axins.spines['bottom'].set_visible(False)

# Hide the ticks
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.tick_params( axis='both', which='both', bottom=False,
    top=False, left=False, right=False, labelbottom=False)

# Axes
axins.arrow(x=inset_xlimit[0]+0.025, y=inset_ylimit[0]+0.1, dx=inset_xlimit[1]-inset_xlimit[0]-0.1, dy=0,
          color='black', width=0.001, length_includes_head=True,
          head_width=0.15, head_length=0.1, overhang=0.5)
axins.arrow(x=inset_xlimit[0]+0.025, y=inset_ylimit[0]+0.1, dy=inset_ylimit[1]-inset_ylimit[0]-0.3, dx=0,
          color='black', width=0.001, length_includes_head=True,
          head_width=0.04, head_length=0.4, overhang=0.5)

# Axes labels
axins.set_xlabel('Absorption', fontsize=10)
axins.xaxis.set_label_coords(0.45, -0.06)
axins.xaxis.set_label_position('top')
axins.set_ylabel('Photon Energy', fontsize=10, rotation=270)
axins.yaxis.set_label_coords(-0.15, 0.45)

def draw_brace(ax, yspan):
#    Draws an annotated brace on the axes.
    ymin, ymax = yspan
    yspan = ymax - ymin
    ax_ymin, ax_ymax = ax.get_ylim()
    yax_span = ax_ymax - ax_ymin
    xmin, xmax = ax.get_xlim()
    xspan = xmax - xmin
    resolution = int(yspan/yax_span*100)*2+1 # guaranteed uneven
    beta = 300./yax_span # the higher this is, the smaller the radius
    y = np.linspace(ymin, ymax, resolution)
    y_half = y[:resolution//2+1]
    x_half_brace = (1/(1.+np.exp(-beta*(y_half-y_half[0])))
                    + 1/(1.+np.exp(-beta*(y_half-y_half[-1]))))
    x = np.concatenate((x_half_brace, x_half_brace[-2::-1]))
    x = xmin + (.05*x - .01)*xspan + 0.75 # adjust vertical position
    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1)

draw_brace(axins, (0.5, 3.5))
draw_brace(axins, (4, 6))





ax.text(x=-2.9, y=11.98, s='d', weight='bold', fontsize=15)



plt.savefig('Fig-1.pdf',dpi=300,bbox_inches='tight' )
plt.show()
