#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13})
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Affine2D
import numpy.polynomial.hermite as Herm
import math

#########
# arrow #
#########

blue = "#4285F4"
red = "#DB4437"
yellow = "#D99A00"
green = "#0F9D58"
purple = "#AB47BC"

fig, ax = plt.subplots(1, 1, figsize=(7, 6))

rot_angle = 0

ax.set_xlim(0.2, 2.6)
ax.set_ylim(-0.7, 4)

##################
# shift function #
##################


def f_shift(x):
    """
    Shift function to add slope for the 3D effect
    """
    y = -0.5 * (x - x[0])
    return y


def projection(x):
    """
    Projection function
    """
    y = 2 * (x - x[0])
    return y


def plot_text_along_slope(x_start=0, y_start=0, slope=-0.5, text="Your Text Here", spacing=0.2, ax=ax, color=None, bbox=None, rot_angle=0):
    """
    Plots text along a line with a specified slope.

    Parameters:
    - x_start, y_start: float, starting coordinates of the line.
    - slope: float, the slope of the line.
    - text: str, the text to plot along the line.
    - spacing: float, the distance between each character.
    """
    skew_transform = Affine2D().skew_deg(slope * 0, 0)

    # Plot each character along the line
    for i, char in enumerate(text):
        x_pos = x_start + i * spacing / np.sqrt(1 + slope**2)
        y_pos = y_start + i * spacing * slope / np.sqrt(1 + slope**2)
        ax.text(x_pos, y_pos, char, ha='center', va='center', color=color, bbox=bbox, rotation=rot_angle, transform=plt.gca().transData + skew_transform)

##########
# brace2 #
##########

def draw_brace2(ax, xc, yspan):
    """Draws an annotated brace on the axes."""
    ymin, ymax = yspan
    yspan = ymax - ymin
    ax_ymin, ax_ymax = ax.get_ylim()
    yax_span = ax_ymax - ax_ymin
    xmin, xmax = ax.get_xlim()
    xspan = 3
    resolution = int(yspan / yax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 200.0 / yax_span  # the higher this is, the smaller the radius
    y = np.linspace(ymin, ymax, resolution)
    y_half = y[: resolution // 2 + 1]
    x_half_brace = 1 / (1.0 + np.exp(-beta * (y_half - y_half[0]))) + 1 / (
        1.0 + np.exp(-beta * (y_half - y_half[-1]))
    )
    x = np.concatenate((x_half_brace, x_half_brace[-2::-1]))
    x = -xmin - (0.02 * x - 0.01) * xspan + xc  # adjust vertical position
    ax.autoscale(False)

    y = y + f_shift(x)

    ax.plot(x, y, color="black", lw=1)


###################################
# plot the vibrational background #
###################################


def QUA(x, a, b, c):
    return a * (x - b) ** 2 + c


# 3A2
def QUA_1(x):
    return QUA(x, 4, 1.225 + xshift, 0.2 - 0.125 + 0.5 - 0.04 - 0.2)


# 3E
def QUA_2(x):
    return QUA(x, 4.5, 1.225 + xshift, 2.8 - 0.25 + 0.5 - 0.03 - 0.1)


shift_singlet = -0.5

# 1A1
def QUA_3(x):
    return QUA(x, 4, 2.475 + xshift, 2.4 + 0.3 - 0.4 - 0.05 + shift_singlet)


# 1E
def QUA_4(x):
    return QUA(x, 4.5, 2.5 + xshift, 0.6 + 0.3 - 0.06 + shift_singlet)


m = 1.0
hbar = 1.0


def hermite(x, w, n):
    xi = np.sqrt(m * w / hbar) * x
    herm_coeffs = np.zeros(n + 1)
    herm_coeffs[n] = 1
    return Herm.hermval(xi, herm_coeffs)


def stationary_state(x, w, n):
    xi = np.sqrt(m * w / hbar) * x
    prefactor = (
        1.0 / np.sqrt(2.0**n * math.factorial(n)) * (m * w / (np.pi * hbar)) ** (0.25)
    )
    psi = prefactor * np.exp(-(xi**2) / 2) * hermite(x, w, n)
    return psi


xshift = -0.3
yshift = -0.4 + 0.5

XAXIS_1 = np.linspace(0.825 + xshift, 1.625 + xshift, 101, endpoint=True)
XAXIS_2 = np.linspace(0.825 + xshift, 1.625 + xshift, 101, endpoint=True)
XAXIS_3 = np.linspace(2.075 + xshift, 2.875 + xshift, 101, endpoint=True)
XAXIS_4 = np.linspace(2.1 + xshift, 2.9 + xshift, 101, endpoint=True)
PRA_1 = np.zeros(101)
PRA_2 = np.zeros(101)
PRA_3 = np.zeros(101)
PRA_4 = np.zeros(101)
for i in range(101):
    PRA_1[i] = QUA_1(XAXIS_1[i])
    PRA_2[i] = QUA_2(XAXIS_2[i])
    PRA_3[i] = QUA_3(XAXIS_3[i])
    PRA_4[i] = QUA_4(XAXIS_4[i])


# Paraballes
# 3A2
ax.plot(
    XAXIS_1,
    PRA_1 + yshift + f_shift(XAXIS_1),
    linewidth=1.5,
    linestyle="-",
    color="dimgray",
    alpha=0.3,
    zorder=1,
)
# 3E
ax.plot(
    XAXIS_2,
    PRA_2 + yshift + f_shift(XAXIS_1) + 0.1,
    linewidth=1.5,
    linestyle="-",
    color="dimgray",
    alpha=0.3,
    zorder=1,
)
# 1A1
ax.plot(
    XAXIS_3,
    PRA_3 + yshift + f_shift(XAXIS_1),
    linewidth=1.5,
    linestyle="-",
    color="dimgray",
    alpha=0.3,
    zorder=1,
)
# 1E
ax.plot(
    XAXIS_4,
    PRA_4 + yshift + f_shift(XAXIS_1),
    linewidth=1.5,
    linestyle="-",
    color="dimgray",
    alpha=0.3,
    zorder=1,
)

# 3A2 wavefunction
w = 1
for i in range(3):
    x = np.linspace(-3.5, 3.5, 100)
    y = stationary_state(x, w, i)
    x = x * 0.15 + 1.225 + xshift
    y = y * 0.1 + 0.2 * i + 0.2 - 0.125 + 0.5 + 0.1 + yshift
    ax.fill_between(
        x,
        0.2 * i + 0.2 - 0.125 + 0.5 + 0.1 + yshift + f_shift(x) - 0.2,
        y + f_shift(x) - 0.2,
        color="dimgray",
        alpha=0.3,
        zorder=0,
    )

# 1A1 wavefunction
w = 1
for i in range(3):
    x = np.linspace(-2.8, 2.8, 100)
    y = stationary_state(x, w, i)
    x = x * 0.15 + 2.475 + xshift
    y = y * 0.1 + 0.2 * i + 2.4 + 0.3 + 0.1 + yshift - 0.4 + shift_singlet
    ax.fill_between(
        x,
        0.2 * i + 2.4 + 0.3 + 0.1 + yshift + f_shift(x) - 0.4 - 0.05 + shift_singlet,
        y + f_shift(x) - 0.05,
        color="dimgray",
        alpha=0.3,
        zorder=0,
    )

# 3E wavefunction
w = np.sqrt(0.7)
for i in range(4):
    x = np.linspace(-3.5, 3.5, 100)
    y = stationary_state(x, w, i)
    # x = x * 0.15 + xshift + 1.4
    x = x * 0.15 + xshift + 1.4 - 0.175
    y = y * 0.1 + (0.1 + 0.2 * i) * np.sqrt(0.7) + 2.8 - 0.25 + 0.5 + yshift
    ax.fill_between(
        x,
        (0.1 + 0.2 * i) * np.sqrt(0.7) + 2.8 - 0.25 + 0.5 + yshift + f_shift(x) + 0.05,
        y + f_shift(x) + 0.05,
        color="dimgray",
        alpha=0.3,
        zorder=0,
    )

# 1E wavefunction
w = np.sqrt(0.7)
for i in range(4):
    x = np.linspace(-3.6, 2.6, 100)
    y = stationary_state(x, w, i)
    x = x * 0.15 + 2.5 + xshift
    y = y * 0.1 + (0.1 + 0.2 * i) * np.sqrt(0.7) + 0.6 + 0.3 + yshift + shift_singlet
    ax.fill_between(
        x,
        (0.1 + 0.2 * i) * np.sqrt(0.7) + 0.6 + 0.3 + yshift + f_shift(x) + 0.02 + shift_singlet,
        y + f_shift(x) + 0.02,
        color="dimgray",
        alpha=0.3,
        zorder=0,
    )


# NOTE: add the energy spacing between 3E lowest level and 1A1 lowest level
x_start = 3.5 * 0.15 + xshift + 1.4 - 0.175
x_end = -2.8 * 0.15 + 2.475 + xshift
x = np.linspace(x_start, x_end, 100)
y_high = 0.1 * np.sqrt(0.7) + 2.8 - 0.25 + 0.5 + yshift + f_shift(x) + 0.05 - 0.525
y_low = 2.4 + 0.3 + 0.1 + yshift + f_shift(x) - 0.4 - 0.05 + shift_singlet + 0.15
ax.plot(x, y_high, color='dimgray', linewidth=0.5, linestyle='-')
ax.plot(x, y_low, color='dimgray', linewidth=0.5, linestyle='-')

ax.annotate(
    text="",
    xy=(x[50], y_high[50]),
    xytext=(x[50], y_low[50]),
    arrowprops=dict(
        arrowstyle="<->",
        linestyle="-",  # linestyle=(0, (10, 5)),
        shrinkA=0,
        shrinkB=0,
        linewidth=1,
        mutation_scale=15,
        color='k',
    ),
)
ax.text(x=x[50] + 0.02, y=(y_high[50] + y_low[50])/2, s='$\Delta$', va='center')



# let's try draw a gray background
xaxis = np.linspace(0.4, 2.6, 101)
y0 = f_shift(xaxis) + 0.5
y1 = y0 + 3.4
ax.fill_between(xaxis, y0, y1, color="lightgray", alpha=0.3, zorder=-1)

ax.vlines(x=0.4, ymin=y0[0], ymax=y1[0], color='lightgray', alpha=1, zorder=0, linestyles='-', linewidth=1.5)

xaxis = np.linspace(0.2, 0.4, 101)
y0 = projection(xaxis) + 0.1
y1 = y0 + 3.4
ax.fill_between(xaxis, y0, y1, color="lightgray", alpha=0.3, zorder=-1)


##########
# levels #
##########

# 1E
xaxis = np.linspace(1.85, 2.3, 101)
ax.plot(xaxis, 0.66 + f_shift(xaxis) + shift_singlet, linewidth=1, linestyle="-", color="k", zorder=6)
ax.plot(xaxis, 0.64 + f_shift(xaxis) + shift_singlet, linewidth=1, linestyle="-", color="k", zorder=6)
# ax.hlines(y=0.6, xmin=1.95, xmax=2.2, linewidth=1, linestyle="-", color="black")

# let's draw white background for 1E
xaxis = np.linspace(1.85, 2.3, 101)
y_0 = f_shift(xaxis) + 0.61 + shift_singlet
y_1 = y_0 + 0.08
ax.fill_between(xaxis, y_0, y_1, color='#E0FFFF', alpha=0.8, zorder=3)

# plot projections #
x0 = 1.85
y0 = 0.6 + 0.05

xaxis = np.linspace(x0, x0 + 0.14, 101)
yaxis = y0 + projection(xaxis) + shift_singlet

ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")

x_points = []
y_points = []
x_points.append(xaxis[0])
y_points.append(yaxis[0])
x_points.append(xaxis[-1])
y_points.append(yaxis[-1])

xaxis = np.linspace(x0 + 0.45, x0 + 0.45 + 0.14, 101)
yaxis = y0 + f_shift(np.array([x0, x0 + 0.45]))[1] + projection(xaxis) + shift_singlet
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")

x_points.append(xaxis[-1])
y_points.append(yaxis[-1])
x_points.append(xaxis[0])
y_points.append(yaxis[0])

# fill the projection region
ax.fill(x_points, y_points, color='darkgray', alpha=0.6, fill=False, hatch='-----', zorder=2)



# 1A1
xaxis = np.linspace(1.85, 2.3, 101)
ax.plot(xaxis, 2.05 + f_shift(xaxis) + shift_singlet, linewidth=1, linestyle="-", color="k", zorder=6)
# let's draw white background for 1A1
xaxis = np.linspace(1.85, 2.3, 101)
y_0 = f_shift(xaxis) + 2.03 + shift_singlet
y_1 = y_0 + 0.04
ax.fill_between(xaxis, y_0, y_1, color='#E0FFFF', alpha=0.8, zorder=3)

# plot projections #
x0 = 1.85
y0 = 2.05
xaxis = np.linspace(x0, x0 + 0.135, 101)
yaxis = y0 + projection(xaxis) + shift_singlet
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")

x_points = []
y_points = []
x_points.append(xaxis[0])
y_points.append(yaxis[0])
x_points.append(xaxis[-1])
y_points.append(yaxis[-1])

xaxis = np.linspace(x0 + 0.45, x0 + 0.45 + 0.135, 101)
yaxis = y0 + f_shift(np.array([x0, x0 + 0.45]))[1] + projection(xaxis) + shift_singlet
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")

x_points.append(xaxis[-1])
y_points.append(yaxis[-1])
x_points.append(xaxis[0])
y_points.append(yaxis[0])

# fill the projection region
ax.fill(x_points, y_points, color='darkgray', alpha=0.6, fill=False, hatch='-----', zorder=2)

##################
# 3E fine levels #
##################

x0 = 0.6
y0 = 2.8
x1 = x0 + 0.45
y1 = y0

# let's draw white background for 3E
xaxis = np.linspace(x0, x1, 101)
y_0 = f_shift(xaxis) + y0 - 0.25
y_1 = y_0 + 0.55
ax.fill_between(xaxis, y_0, y_1, color='#E0FFFF', alpha=0.8, zorder=3)

# plot projections #
# the middle line
xaxis = np.linspace(x0, x0 + 0.15, 101)
yaxis = y0 + projection(xaxis)
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")

x_points = []
y_points = []
x_points.append(xaxis[0])
y_points.append(yaxis[0])
x_points.append(xaxis[-1])
y_points.append(yaxis[-1])

xaxis = np.linspace(x0 + 0.45, x0 + 0.45 + 0.15, 101)
yaxis = y0 + f_shift(np.array([x0, x0 + 0.45]))[1] + projection(xaxis)
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")

x_points.append(xaxis[-1])
y_points.append(yaxis[-1])
x_points.append(xaxis[0])
y_points.append(yaxis[0])

# fill the projection region
ax.fill(x_points, y_points, color='darkgray', alpha=0.6, fill=False, hatch='-----', zorder=2)



# the upper line
def projection2(x):
    y = (x - x[0]) * -0.04
    return y


xaxis = np.linspace(x0, x0 + 0.15, 101)
yaxis = y0 + projection2(xaxis) + 0.3
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")

xaxis = np.linspace(x0 + 0.45, x0 + 0.45 + 0.15, 101)
yaxis = y0 + f_shift(np.array([x0, x0 + 0.45]))[1] + projection2(xaxis) + 0.3
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")


# the bottom line
def projection3(x):
    y = (x - x[0]) * 3.6
    return y


xaxis = np.linspace(x0, x0 + 0.15, 101)
yaxis = y0 + projection3(xaxis) - 0.25
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")

xaxis = np.linspace(x0 + 0.45, x0 + 0.45 + 0.15, 101)
yaxis = y0 + f_shift(np.array([x0, x0 + 0.45]))[1] + projection3(xaxis) - 0.25
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")


# e_xy
ax.plot(
    np.linspace(x0, x1, 11),
    np.linspace(y0, y1, 11) + f_shift(np.linspace(x0, x1, 11)),
    color="k",
    linestyle="-",
    linewidth=1,
    zorder=6,
)
ax.plot(
    np.linspace(x0, x1, 11),
    np.linspace(y0, y1, 11) + 0.02 + f_shift(np.linspace(x0, x1, 11)),
    color="k",
    linestyle="-",
    linewidth=1,
    zorder=6,
)
# e_12
ax.plot(
    np.linspace(x0, x1, 11),
    np.linspace(y0 - 0.25, y1 - 0.25, 11) + f_shift(np.linspace(x0, x1, 11)),
    color="k",
    linestyle="-",
    linewidth=1,
    zorder=6,
)
ax.plot(
    np.linspace(x0, x1, 11),
    np.linspace(y0 - 0.25, y1 - 0.25, 11) + 0.02 + f_shift(np.linspace(x0, x1, 11)),
    color="k",
    linestyle="-",
    linewidth=1,
    zorder=6,
)
# a1
ax.plot(
    np.linspace(x0, x1, 11),
    np.linspace(y0 + 0.25, y1 + 0.25, 11) + f_shift(np.linspace(x0, x1, 11)),
    color="k",
    linestyle="-",
    linewidth=1,
    zorder=6,
)
# a2
ax.plot(
    np.linspace(x0, x1, 11),
    np.linspace(y0 + 0.30, y1 + 0.30, 11) + f_shift(np.linspace(x0, x1, 11)),
    color="k",
    linestyle="-",
    linewidth=1,
    zorder=6,
)

# lambda_z
shift = f_shift(np.array([x0, x0 + 0.31]))[1]
ax.annotate(
    text="",
    xy=(x0 + 0.31, y0 + shift),
    xytext=(x0 + 0.31, y0 + 0.25 + shift + 0.035),
    arrowprops=dict(
        arrowstyle="<->",
        linestyle="-",
        shrinkA=0,
        shrinkB=0,
        mutation_scale=15,
        color="black",
    ),
)
shift = f_shift(np.array([x0, x0 + 0.35]))[1]
ax.text(
    x=x0 + 0.35,
    y=2.925 + shift,
    s="$\lambda_z$",
    color="black",
    fontsize=13,
    va="center",
    rotation=rot_angle,
)
shift = f_shift(np.array([x0, x0 + 0.31]))[1]
ax.annotate(
    text="",
    xy=(x0 + 0.31, y0 + shift),
    xytext=(x0 + 0.31, y0 - 0.25 + shift),
    arrowprops=dict(
        arrowstyle="<->",
        linestyle="-",
        shrinkA=0,
        shrinkB=0,
        mutation_scale=15,
        color="black",
    ),
)
shift = f_shift(np.array([x0, x0 + 0.35]))[1]
ax.text(
    x=x0 + 0.35,
    y=2.675 + shift,
    s="$\lambda_z$",
    color="black",
    fontsize=13,
    va="center",
    rotation=rot_angle,
)

draw_v3e = True
if draw_v3e:
    t_shift_x = -0.18
    t_shift_y = 0.0
    draw_brace2(ax, x0 - 0.00, (y0 - 0.35 + 0.1, y0 + 0.35 + 0.15))
    
    plot_text_along_slope(x0 + t_shift_x, 
                          y0 + 0.45 + t_shift_y, 
                          -0.5, 
                          text=["$|$", "$\\widetilde{A}$", "$_{2}$", "$\\rangle$"], 
                          spacing=0.043, 
                          ax=ax, 
                          color="black", 
                          bbox=None, 
                          rot_angle=rot_angle)
    
    plot_text_along_slope(x0 + t_shift_x, 
                          y0 + 0.29 + t_shift_y, 
                          -0.5, 
                          text=["$|$", "$\\widetilde{A}$", "$_{1}$", "$\\rangle$"], 
                          spacing=0.043, 
                          ax=ax, 
                          color="black", 
                          bbox=None, 
                          rot_angle=rot_angle)
    
    plot_text_along_slope(x0 + t_shift_x, 
                          y0 + 0.07 + t_shift_y, 
                          -0.5, 
                          text=["$|$", "$\\widetilde{E}$", "$_{x,}$", "$_{y}$", "$\\rangle$"], 
                          spacing=0.043, 
                          ax=ax, 
                          color="black", 
                          bbox=None, 
                          rot_angle=rot_angle)
    
    plot_text_along_slope(x0 + t_shift_x, 
                          y0 - 0.18 + t_shift_y, 
                          -0.5, 
                          text=["$|$", "$\\widetilde{E}$", "$_{1,}$", "$_{2}$", "$\\rangle$"], 
                          spacing=0.043, 
                          ax=ax, 
                          color="black", 
                          bbox=None, 
                          rot_angle=rot_angle)


# \Gamma_{A_1}
shift = f_shift(np.array([x0, x0 + 0.65]))[1]

start = (x0 + 0.45, y0 + 0.35 + shift)
end = (x0 + 1.25 + 0.225, y0 - 0.4 - 0.4 + 0.05 - 0.1125 + shift_singlet)

arrow = FancyArrowPatch(start, end,
                        connectionstyle="arc3,rad=-.1",
                        arrowstyle='->',
                        linewidth=1.5,
                        linestyle="--",
                        color=red,
                        mutation_scale=25,
                        zorder=3)
ax.add_patch(arrow)
ax.text(
    x=1.3,
    y=2.95 + shift / 1.5 - 0.1 - 0.18 + 0.17,
    s="$\Gamma_{A_1}$",
    color=red,
    fontsize=16,
    rotation=rot_angle,
)

# \Gamma_{E_{1,2}}
shift = f_shift(np.array([x0, x0 + 0.7]))[1]
start = (x0 + 0.45, y0 - 0.12 + shift)
end = (x0 + 1.25 + 0.225, y0 - 0.4 - 0.4 + 0.05 - 0.1125 + shift_singlet)

arrow = FancyArrowPatch(start, end,
                        connectionstyle="arc3,rad=-.1",
                        arrowstyle='->',
                        linewidth=1.5,
                        linestyle="--",
                        color=yellow,
                        mutation_scale=25,
                        zorder=3)
ax.add_patch(arrow)
ax.text(
    x=1.3,
    y=2.4 + shift / 1.5 - 0.1 - 0.2 + 0.13,
    s="$\Gamma_{E_{1,2}}$",
    color=yellow,
    fontsize=16,
    rotation=rot_angle,
)


######################
# 3A2 fine structure #
######################

x0 = 0.6
y0 = 0.2
x1 = x0 + 0.45
y1 = y0

# let's draw white background for 3E, 3A2, 1A1, 1E
xaxis = np.linspace(x0, x1, 101)
y_0 = f_shift(xaxis) + y0 - 0.125
y_1 = y_0 + 0.27
ax.fill_between(xaxis, y_0, y_1, color='#E0FFFF', alpha=0.8, zorder=3)

# plot projections #
xaxis = np.linspace(x0, x0 + 0.16, 101)
yaxis = y0 - 0.125 + projection(xaxis)
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")

x_points = []
y_points = []
x_points.append(xaxis[0])
y_points.append(yaxis[0])
x_points.append(xaxis[-1])
y_points.append(yaxis[-1])

xaxis = np.linspace(x0 + 0.45, x0 + 0.45 + 0.16, 101)
yaxis = y0 - 0.125 + f_shift(np.array([x0, x0 + 0.45]))[1] + projection(xaxis)
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")

x_points.append(xaxis[-1])
y_points.append(yaxis[-1])
x_points.append(xaxis[0])
y_points.append(yaxis[0])

# fill the projection region
ax.fill(x_points, y_points, color='darkgray', alpha=0.6, fill=False, hatch='-----', zorder=2)


# the upper line
def projection2(x):
    y = (x - x[0]) * 0.2
    return y

xaxis = np.linspace(x0, x0 + 0.15, 101)
yaxis = y0 + projection2(xaxis) + 0.155
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")

xaxis = np.linspace(x0 + 0.45, x0 + 0.45 + 0.15, 101)
yaxis = y0 + f_shift(np.array([x0, x0 + 0.45]))[1] + projection2(xaxis) + 0.155
ax.plot(xaxis, yaxis, linestyle="--", linewidth=1, color="gray")


# 0
ax.plot(
    np.linspace(x0, x1, 11),
    np.linspace(y0 - 0.125, y1 - 0.125, 11) + f_shift(np.linspace(x0, x1, 11)),
    color="k",
    linestyle="-",
    linewidth=1,
    zorder=6,
)
# +-1
ax.plot(
    np.linspace(x0, x1, 11),
    np.linspace(y0 + 0.125, y1 + 0.125, 11) - 0.03 + f_shift(np.linspace(x0, x1, 11)),
    color="k",
    linestyle="-",
    linewidth=1,
    zorder=6,
)
ax.plot(
    np.linspace(x0, x1, 11),
    np.linspace(y0 + 0.125, y1 + 0.125, 11) + 0.03 + f_shift(np.linspace(x0, x1, 11)),
    color="k",
    linestyle="-",
    linewidth=1,
    zorder=6,
)
if draw_v3e:
    t_shift_x = -0.12
    t_shift_y = 0.02
    draw_brace2(ax, x0 + 0.05, (y0 - 0.175 + 0.08, y0 + 0.175 + 0.08))
    
    plot_text_along_slope(x0 + t_shift_x, 
                          y0 + 0.17 + t_shift_y, 
                          -0.5, 
                          text=["$|$", "$\pm$", "$\\rangle$"], 
                          spacing=0.043, 
                          ax=ax, 
                          color="black", 
                          bbox=None, 
                          rot_angle=rot_angle)
    
    plot_text_along_slope(x0 + t_shift_x, 
                          y0 - 0.08 + t_shift_y, 
                          -0.5, 
                          text=["$|$", "$0$", "$\\rangle$"], 
                          spacing=0.043, 
                          ax=ax, 
                          color="black", 
                          bbox=None, 
                          rot_angle=rot_angle)

# t_{\pm}
shift = f_shift(np.array([x0, x0 + 0.65]))[1]
start = (x0 + 0.45, y0 + 0.23 + shift)
end = (x0 + 1.25 + 0.225, y0 + 0.4 + 0.05 - 0.1125 + shift_singlet)

arrow = FancyArrowPatch(start, end,
                        connectionstyle="arc3,rad=.2",
                        arrowstyle='<-',
                        linewidth=1.5,
                        linestyle="--",
                        color=purple,
                        mutation_scale=25,
                        zorder=3)
ax.add_patch(arrow)
ax.text(
    x=1.45,
    y=0.65 + shift / 1.5 - 0.27 - 0.25,
    s="$t_{\pm}$",
    color=purple,
    fontsize=16,
    rotation=rot_angle,
)

# t_{z}
shift = f_shift(np.array([x0, x0 + 0.65]))[1]
start = (x0 + 0.45, y0 - 0.03 + shift)
end = (x0 + 1.25 + 0.225, y0 + 0.4 + 0.05 - 0.1125 + shift_singlet)

arrow = FancyArrowPatch(start, end,
                        connectionstyle="arc3,rad=.2",
                        arrowstyle='<-',
                        linewidth=1.5,
                        linestyle="--",
                        color=purple,
                        mutation_scale=25,
                        zorder=3)
ax.add_patch(arrow)
ax.text(
    x=1.45,
    y=0.23 + shift / 1.5 - 0.3 - 0.2,
    s="$t_{z}$",
    color=purple,
    fontsize=16,
    rotation=rot_angle,
)

#########################
# radiative transitions #
#########################

# 1A1 -> 1E
shift = f_shift(np.array([1.95, 1.975]))[1]
ax.annotate(
    text="",
    xy=(1.975, 0.6 + shift + shift_singlet),
    xytext=(1.975, 2.0 - 0.01 + shift + shift_singlet),
    arrowprops=dict(
        arrowstyle="->",
        linestyle="-",
        linewidth=1.5,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=15,
        color="maroon",
    ),
)

# non-rad
shift = f_shift(np.array([1.95, 2.175]))[1]
ax.annotate(
    text="",
    xy=(2.175, 0.6 + shift + shift_singlet),
    xytext=(2.175, 2.0 - 0.01 + shift + shift_singlet),
    arrowprops=dict(
        arrowstyle="->",
        linestyle=(0, (4, 4)),
        linewidth=1.,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=15,
        color='maroon',
    ),
)

bbox = dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.1", alpha=0.0)
plot_text_along_slope(2.075 - 0.1 + 0.05, 1.3 + 0.1 + 0.25 + shift_singlet,
                       -0.5, text="ZPL: 1.19 eV", spacing=0.043, 
                       ax=ax, color="maroon", bbox=bbox, rot_angle=rot_angle)


# excitation from 3A2 to above 3E
ax.annotate(
    text="",
    xy=(0.65, 0.2 - 0.1 + 0.125 + 0.03 + shift),
    xytext=(0.65, 3.4 + 0.25 - 0.01 + shift),
    arrowprops=dict(
        arrowstyle="<-",
        linestyle="-",
        linewidth=2.5,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=25,
        color=green,
    ),
)


# 3E -> 3A2
# A2, A1 -> +
shift = f_shift(np.array([x0, 0.75]))[1]
ax.annotate(
    text="",
    xy=(0.75, 0.2 + 0.125 + 0.03 + shift),
    xytext=(0.75, 2.8 + 0.25 - 0.01 + shift),
    arrowprops=dict(
        arrowstyle="->",
        linestyle="-",
        linewidth=1.5,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=15,
        color=blue,
    ),
)
shift = f_shift(np.array([x0, 0.72]))[1]
ax.annotate(
    text="",
    xy=(0.72, 0.2 + 0.125 + 0.03 + shift),
    xytext=(0.72, 2.8 + 0.30 - 0.01 + shift),
    arrowprops=dict(
        arrowstyle="->",
        linestyle="-",
        linewidth=1.5,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=15,
        color=blue,
    ),
)

# Exy -> 0
shift = f_shift(np.array([x0, 0.835]))[1]
ax.annotate(
    text="",
    xy=(0.835, 0.2 - 0.125 + shift),
    xytext=(0.835, 2.8 + 0.02 - 0.01 + shift),
    arrowprops=dict(
        arrowstyle="->",
        linestyle="-",
        linewidth=1.5,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=15,
        color=blue,
    ),
)
shift = f_shift(np.array([x0, 0.865]))[1]
ax.annotate(
    text="",
    xy=(0.865, 0.2 - 0.125 + shift),
    xytext=(0.865, 2.8 - 0.01 + shift),
    arrowprops=dict(
        arrowstyle="->",
        linestyle="-",
        linewidth=1.5,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=15,
        color=blue,
    ),
)

# E12 -> -
shift = f_shift(np.array([x0, 0.95]))[1]
ax.annotate(
    text="",
    xy=(0.95, 0.2 + 0.125 - 0.03 + shift),
    xytext=(0.95, 2.8 - 0.25 + 0.02 - 0.01 + shift),
    arrowprops=dict(
        arrowstyle="->",
        linestyle="-",
        linewidth=1.5,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=15,
        color=blue,
    ),
)
shift = f_shift(np.array([x0, 0.98]))[1]
ax.annotate(
    text="",
    xy=(0.98, 0.2 + 0.125 - 0.03 + shift),
    xytext=(0.98, 2.8 - 0.25 - 0.01 + shift),
    arrowprops=dict(
        arrowstyle="->",
        linestyle="-",
        linewidth=1.5,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=15,
        color=blue,
    ),
)
bbox = dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.5", alpha=0.)
plot_text_along_slope(1.05, 1.5, -0.5, text="ZPL:\n1.945 eV", spacing=0.043, ax=ax, color=blue, bbox=bbox, rot_angle=rot_angle)


shift = f_shift(np.array([x0, 0.4]))[1]
ax.text(
    x=0.28,
    y=0.25 + shift,
    s="$^3A_2$",
    ha="center",
    va="center",
    fontsize=15,
    color="black",
    rotation=rot_angle,
)

shift = f_shift(np.array([x0, 0.35]))[1]
ax.text(
    x=2.51 - 0.1,
    y=0.6 - 0.3 + 0.6 - 1.05,
    s="$^1E$",
    ha="center",
    va="center",
    fontsize=15,
    color="black",
    rotation=rot_angle,
)

shift = f_shift(np.array([x0, 0.35]))[1]
ax.text(
    x=2.51 - 0.1,
    y=2.4 - 0.6 + 0.45 - 0.98,
    s="$^1A_1$",
    ha="center",
    va="center",
    fontsize=15,
    color="black",
    rotation=rot_angle,
)

shift = f_shift(np.array([x0, 0.4]))[1]
ax.text(
    x=0.28,
    y=2.95,
    s="$^3E$",
    ha="center",
    va="center",
    fontsize=15,
    color="black",
    rotation=rot_angle,
)


ax.tick_params(axis="both", direction="in")
ax.yaxis.set_ticks_position("both")
ax.set_xticks([])
ax.set_yticks([])

ax.axis("off")

fig.tight_layout()
plt.savefig("figure_1.pdf", dpi=300, bbox_inches="tight")
plt.show()