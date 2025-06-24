#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import matplotlib.patches as patches

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{xcolor}'

#############
# functions #
#############

def draw_arrows(start_x, start_y, dx, dy, ax, ls='-', color='black'):
    ax.annotate(
        text="",
        xy=(start_x, start_y),
        xytext=(start_x + dx, start_y + dy),
        arrowprops=dict( 
            arrowstyle="<-",
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=15,
            color=color,
        ),
    )
    return

########
# main #
########

# colors
blue = "#4285F4"
red = "#DB4437"
yellow = "#F4B400"
green = "#0F9D58"


# Get the colormap
cmap = plt.cm.YlOrRd
# Generate evenly spaced values between 0 and 1
values = np.array([0.0, 0.1, 0.2, 0.3, 0.4]) * 1.5
# Get colors from the colormap
colors = [cmap(val) for val in values]


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-0.38, 1.41)
ax.set_ylim(0.18, 0.92)
ax.axis('off')


# Step 1: dft references
bottom_left_x = 0.2
bottom_left_y = 0.8
width = 1.0 - 2 * bottom_left_x
height = 0.1

rounded_rect = patches.FancyBboxPatch(
    (bottom_left_x, bottom_left_y),  # Bottom-left corner of the rectangle
    width, height,    # Width and height of the rectangle
    boxstyle="round,pad=0.0,rounding_size=0.03",  # Rounded corners
    edgecolor="black",
    facecolor=colors[2],
    linewidth=1.
)
ax.add_patch(rounded_rect)
text = r"Ground State Electronic Structure"
ax.text(x=0.5, y=0.87, s=text, va='center', ha='center', multialignment='center', usetex=True)
text = r"\textbf{(DFT)} \texttt{[Quantum ESPRESSO]}"
ax.text(x=0.5, y=0.83, s=text, va='center', ha='center', multialignment='center', usetex=True)


# Step 2.1: qdet
bottom_left_x = -0.35
bottom_left_y = 0.6
width = 0.5
height = 0.1

rounded_rect = patches.FancyBboxPatch(
    (bottom_left_x, bottom_left_y),  # Bottom-left corner of the rectangle
    width, height,    # Width and height of the rectangle
    boxstyle="round,pad=0.0,rounding_size=0.03",  # Rounded corners
    edgecolor="black",
    facecolor=colors[4],
    linewidth=1.
)
ax.add_patch(rounded_rect)
text = r"Many-Body Electronic States"
ax.text(x=-0.1, y=0.67, s=text, va='center', ha='center', usetex=True)
text = r"\textbf{(QDET)} \texttt{[WEST]}"
ax.text(x=-0.1, y=0.63, s=text, va='center', ha='center', usetex=True)


# Step 2.2: tddft
bottom_left_x = 0.19
bottom_left_y = 0.6
width = 0.42
height = 0.1

rounded_rect = patches.FancyBboxPatch(
    (bottom_left_x, bottom_left_y),  # Bottom-left corner of the rectangle
    width, height,    # Width and height of the rectangle
    boxstyle="round,pad=0.0,rounding_size=0.03",  # Rounded corners
    edgecolor="black",
    facecolor=colors[3],
    linewidth=1.
)
ax.add_patch(rounded_rect)
text = r"Excited State Geometry"
ax.text(x=0.40, y=0.67, s=text, va='center', ha='center', usetex=True)
text = r"\textbf{(TDDFT)} \texttt{[WEST]}"
ax.text(x=0.40, y=0.63, s=text, va='center', ha='center', usetex=True)


# Step 2.3: frozen phonon calculation (optional)
bottom_left_x = 0.66
bottom_left_y = 0.6
width = 0.72
height = 0.1

rounded_rect = patches.FancyBboxPatch(
    (bottom_left_x, bottom_left_y),  # Bottom-left corner of the rectangle
    width, height,    # Width and height of the rectangle
    boxstyle="round,pad=0.0,rounding_size=0.03",  # Rounded corners
    edgecolor="black",
    facecolor=colors[4],
    linewidth=1.
)
ax.add_patch(rounded_rect)
text = r"Phonons"
ax.text(x=1.025, y=0.67, s=text, va='center', ha='center', usetex=True)
text = r"\textbf{(DFT\&TDDFT)} \texttt{[QE, WEST, Phonopy]}"
ax.text(x=1.025, y=0.63, s=text, va='center', ha='center', usetex=True)


# Step 3.1: soc
bottom_left_x = -0.35
bottom_left_y = 0.4
width = 0.45
height = 0.1

rounded_rect = patches.FancyBboxPatch(
    (bottom_left_x, bottom_left_y),  # Bottom-left corner of the rectangle
    width, height,    # Width and height of the rectangle
    boxstyle="round,pad=0.0,rounding_size=0.03",  # Rounded corners
    edgecolor="black",
    facecolor=colors[1],
    linewidth=1.
)
ax.add_patch(rounded_rect)
text = r"Spin-Orbit Coupling, $\lambda$"
ax.text(x=-0.13, y=0.47, s=text, va='center', ha='center', usetex=True)
text = r"\texttt{[PyFRSOC]}"
ax.text(x=-0.13, y=0.43, s=text, va='center', ha='center', usetex=True)


# Step 3.2: Delta
bottom_left_x = 0.19
bottom_left_y = 0.4
width = 0.4
height = 0.1

rounded_rect = patches.FancyBboxPatch(
    (bottom_left_x, bottom_left_y),  # Bottom-left corner of the rectangle
    width, height,    # Width and height of the rectangle
    boxstyle="round,pad=0.0,rounding_size=0.03",  # Rounded corners
    edgecolor="black",
    facecolor=colors[0],
    linewidth=1.
)
ax.add_patch(rounded_rect)
text = r"Energy gap, $\Delta$"
ax.text(x=0.4, y=0.45, s=text, va='center', ha='center', usetex=True)


# Step 3.3: vibrational overlap function
bottom_left_x = 0.66
bottom_left_y = 0.4
width = 0.72
height = 0.1

rounded_rect = patches.FancyBboxPatch(
    (bottom_left_x, bottom_left_y),  # Bottom-left corner of the rectangle
    width, height,    # Width and height of the rectangle
    boxstyle="round,pad=0.0,rounding_size=0.03",  # Rounded corners
    edgecolor="black",
    facecolor=colors[1],
    linewidth=1.
)
ax.add_patch(rounded_rect)
text = r"Vibrational Overlap Function, $F(\Delta)$"
ax.text(x=1.025, y=0.47, s=text, va='center', ha='center', usetex=True)
text = r"\textbf{(Huang-Rhys Theory)} \texttt{[PyPL]}"
ax.text(x=1.025, y=0.43, s=text, va='center', ha='center', usetex=True)


# step 4: ISC rates
bottom_left_x = 0.3
bottom_left_y = 0.2
width = 1.0 - 2 * bottom_left_x
height = 0.1

rounded_rect = patches.FancyBboxPatch(
    (bottom_left_x, bottom_left_y),  # Bottom-left corner of the rectangle
    width, height,    # Width and height of the rectangle
    boxstyle="round,pad=0.0,rounding_size=0.03",  # Rounded corners
    edgecolor="black",
    facecolor=colors[0],
    linewidth=1.
)
ax.add_patch(rounded_rect)
text = r"ISC rate"
ax.text(x=0.5, y=0.27, s=text, va='center', ha='center', multialignment='center', usetex=True)
text = r"$\Gamma = \frac{2\pi}{\hbar}|\lambda|^2 F (\Delta)$"
ax.text(x=0.5, y=0.23, s=text, va='center', ha='center', multialignment='center', usetex=True)


# 1 - 2.1
start_x = 0.5
start_y = 0.8
dx = -0.6
dy = -0.1 + 0.003
draw_arrows(start_x, start_y, dx, dy, ax)

# 1 - 2.2
start_x = 0.5
start_y = 0.8
dx = 0.0
dy = -0.1 + 0.003
draw_arrows(start_x, start_y, dx, dy, ax)

# 2.1 - 3.1
start_x = -0.1
start_y = 0.6
dx = -0.0
dy = -0.1 + 0.003
draw_arrows(start_x, start_y, dx, dy, ax)

# 2.1 - 3.2
start_x = -0.1
start_y = 0.6
dx = 0.6
dy = -0.1 + 0.003
draw_arrows(start_x, start_y, dx, dy, ax)

# 2.2 - 3.2
start_x = 0.5
start_y = 0.6
dx = 0.
dy = -0.1 + 0.003
draw_arrows(start_x, start_y, dx, dy, ax)

# 2.2 - 3.3
start_x = 0.5
start_y = 0.6
dx = 0.46 + 0.06
dy = -0.1 + 0.003
draw_arrows(start_x, start_y, dx, dy, ax)

# 3.1 - 4
start_x = -0.1
start_y = 0.4
dx = 0.6
dy = -0.1 + 0.003
draw_arrows(start_x, start_y, dx, dy, ax)

# 3.2 - 4
start_x = 0.5
start_y = 0.4
dx = 0.
dy = -0.1 + 0.003
draw_arrows(start_x, start_y, dx, dy, ax)

# 3.3 - 4
start_x = 0.96
start_y = 0.4
dx = -0.46
dy = -0.1 + 0.003
draw_arrows(start_x, start_y, dx, dy, ax)

# 1 - 2.3
start_x = 0.5
start_y = 0.8
dx = 1.08 - 0.55
dy = -0.1 + 0.003
draw_arrows(start_x, start_y, dx, dy, ax, ls='--', color='gray')

# 2.2 - 2.3
start_x = 0.61
start_y = 0.65
dx = 0.05
dy = 0
draw_arrows(start_x, start_y, dx, dy, ax, ls='--', color='gray')

# 2.3 - 3.3
start_x = 1.02
start_y = 0.6
dx = -0.
dy = -0.1 + 0.003
draw_arrows(start_x, start_y, dx, dy, ax, ls='--', color='gray')

plt.savefig('figure_s1.pdf', bbox_inches='tight', dpi=300)
plt.show()
