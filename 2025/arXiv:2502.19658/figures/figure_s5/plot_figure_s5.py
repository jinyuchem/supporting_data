#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({"font.size": 12})

#############
# Functions #
#############


def Gaussian(x, mu, sigma):
    prefix = 1 / np.sqrt(2 * np.pi * sigma**2)
    eeee = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return prefix * eeee


def sigma(freq):
    f = 6 - (6 - 2) * freq / 160
    return f


########
# Main #
########

fnames = [
    "216_cell/All-B-3A2-3E-215-rc2-15-cph-5.dat",
    "216_cell/All-B-1A1-3E-215-rc2-15-cph-5.dat",
    "13824_cell/All-B-3A2-3E.dat",
    "13824_cell/All-B-1A1-3E.dat",
]

bf = []
freq = []
for i in range(len(fnames)):
    bf.append(np.loadtxt(fnames[i], usecols=3))
    freq.append(np.loadtxt(fnames[i], usecols=6) * 1000 / 8065.544)

# deg?
wd = [
    np.zeros(freq[0].shape[0]),
    np.zeros(freq[1].shape[0]),
    np.zeros(freq[2].shape[0]),
    np.zeros(freq[3].shape[0]),
]

for i in range(len(fnames)):
    for j in range(1, freq[i].shape[0]):
        if abs(freq[i][j] - freq[i][j - 1]) < 1e-5:
            wd[i][j] = 1
            wd[i][j - 1] = 1

bf_a = [
    np.zeros(freq[0].shape[0]),
    np.zeros(freq[1].shape[0]),
    np.zeros(freq[2].shape[0]),
    np.zeros(freq[3].shape[0]),
]

bf_e = [
    np.zeros(freq[0].shape[0]),
    np.zeros(freq[1].shape[0]),
    np.zeros(freq[2].shape[0]),
    np.zeros(freq[3].shape[0]),
]

for i in range(len(fnames)):
    bf_e[i][:] = bf[i][:] * wd[i][:]
    bf_a[i][:] = bf[i][:] - bf_e[i][:]


DATA = np.zeros((4, 1000))
DATA_A = np.zeros((4, 1000))
DATA_E = np.zeros((4, 1000))

ENEAXIS = np.linspace(0, 200, 1000, endpoint=True)

for a in range(len(fnames)):
    DATA[a, :] += np.sum(
        (bf[a][:, np.newaxis] ** 2) *
        Gaussian(
            ENEAXIS[np.newaxis, :],
            freq[a][:, np.newaxis],
            sigma(freq[a][:, np.newaxis])
        ),
        axis=0
    )
    DATA_A[a, :] += np.sum(
        (bf_a[a][:, np.newaxis] ** 2) *
        Gaussian(
            ENEAXIS[np.newaxis, :],
            freq[a][:, np.newaxis],
            sigma(freq[a][:, np.newaxis])
        ),
        axis=0
    )
    DATA_E[a, :] += np.sum(
        (bf_e[a][:, np.newaxis] ** 2) *
        Gaussian(
            ENEAXIS[np.newaxis, :],
            freq[a][:, np.newaxis],
            sigma(freq[a][:, np.newaxis])
        ),
        axis=0
    )


########
# Plot #
########

fig, ax = plt.subplots(3, 2, figsize=(12, 7))

colors = ["#DB4437", "#4285F4", "#F4B400", "#0F9D58"]
linestyles = ["--", "-", "--"]
labels = ["215-atom, ", "13823-atom, "]

# fig-1 3E -> 3A2
ax[0][0].plot(
    ENEAXIS[:],
    DATA[0],
    linewidth=1.5,
    linestyle=linestyles[0],
    color=colors[0],
    label=labels[0] + "$S_{\mathrm{Total}}=$%.2f" % (sum(bf[0] ** 2)),
)
ax[0][0].plot(
    ENEAXIS[:],
    DATA[2],
    linewidth=1.5,
    linestyle=linestyles[1],
    color=colors[0],
    label=labels[1] + "$S_{\mathrm{Total}}=$%.2f" % (sum(bf[2] ** 2)),
)

# fig-2 3E -> 3A2, a1
ax[1][0].plot(
    ENEAXIS[:],
    DATA_A[0],
    linewidth=1.5,
    linestyle=linestyles[0],
    color=colors[0],
    label=labels[0] + "$S_{a_1}=$%.2f" % (sum(bf_a[0] ** 2)),
)
ax[1][0].plot(
    ENEAXIS[:],
    DATA_A[2],
    linewidth=1.5,
    linestyle=linestyles[1],
    color=colors[0],
    label=labels[1] + "$S_{a_1}=$%.2f" % (sum(bf_a[2] ** 2)),
)

# fig-3 3E -> 3A2, e
ax[2][0].plot(
    ENEAXIS[:],
    DATA_E[0],
    linewidth=1.5,
    linestyle=linestyles[0],
    color=colors[0],
    label=labels[0] + "$S_{e}=$%.2f" % (sum(bf_e[0] ** 2)),
)
ax[2][0].plot(
    ENEAXIS[:],
    DATA_E[2],
    linewidth=1.5,
    linestyle=linestyles[1],
    color=colors[0],
    label=labels[1] + "$S_{e}=$%.2f" % (sum(bf_e[2] ** 2)),
)

# fig-4 3E -> 1A1
ax[0][1].plot(
    ENEAXIS[:],
    DATA[1],
    linewidth=1.5,
    linestyle=linestyles[0],
    color=colors[1],
    label=labels[0] + "$S_{\mathrm{Total}}=$%.2f" % (sum(bf[1] ** 2)),
)
ax[0][1].plot(
    ENEAXIS[:],
    DATA[3],
    linewidth=1.5,
    linestyle=linestyles[1],
    color=colors[1],
    label=labels[1] + "$S_{\mathrm{Total}}=$%.2f" % (sum(bf[3] ** 2)),
)

# fig-5 3E -> 1A1, a1
ax[1][1].plot(
    ENEAXIS[:],
    DATA_A[1],
    linewidth=1.5,
    linestyle=linestyles[0],
    color=colors[1],
    label=labels[0] + "$S_{a_1}=$%.2f" % (sum(bf_a[1] ** 2)),
)
ax[1][1].plot(
    ENEAXIS[:],
    DATA_A[3],
    linewidth=1.5,
    linestyle=linestyles[1],
    color=colors[1],
    label=labels[1] + "$S_{a_1}=$%.2f" % (sum(bf_a[3] ** 2)),
)

# fig-6 3E -> 1A1, e
ax[2][1].plot(
    ENEAXIS[:],
    DATA_E[1],
    linewidth=1.5,
    linestyle=linestyles[0],
    color=colors[1],
    label=labels[0] + "$S_{e}=$%.2f" % (sum(bf_e[1] ** 2)),
)
ax[2][1].plot(
    ENEAXIS[:],
    DATA_E[3],
    linewidth=1.5,
    linestyle=linestyles[1],
    color=colors[1],
    label=labels[1] + "$S_{e}=$%.2f" % (sum(bf_e[3] ** 2)),
)

ax[0][0].text(
    x=0.5,
    y=1.15,
    s="$^3E \\to ^3A_2$",
    va="center",
    ha="center",
    transform=ax[0][0].transAxes,
    fontsize=14,
)
ax[0][1].text(
    x=0.5,
    y=1.15,
    s="$^3E \\to ^1A_1$",
    va="center",
    ha="center",
    transform=ax[0][1].transAxes,
    fontsize=14,
)

for i in range(3):
    for j in range(2):
        ax[i][j].legend(fontsize=10, loc="upper right", edgecolor="black")
        ax[i][j].grid(color='gray', linestyle='--', linewidth=0.5)

    ylim = 0.6
    ax[i][0].set_ylim((0, ylim / 10))
    ax[i][0].set_xlim((0, 200))
    if i != 2:
        ax[i][0].set_xticklabels([])
    ax[i][0].tick_params(axis="both", direction="in")
    ax[i][0].tick_params(which="minor", direction="in")
    ax[i][0].xaxis.set_ticks_position("both")
    ax[i][0].yaxis.set_ticks_position("both")

    ylim = 0.6
    ax[i][1].set_yticklabels([])
    ax[i][1].set_ylim((0, ylim / 10))
    ax[i][1].set_xlim((0, 200))
    if i != 2:
        ax[i][1].set_xticklabels([])
    ax[i][1].tick_params(axis="both", direction="in")
    ax[i][1].tick_params(which="minor", direction="in")
    ax[i][1].xaxis.set_ticks_position("both")
    ax[i][1].yaxis.set_ticks_position("both")


ax[0][0].text(x=6, y=0.05, s="$a_1 + e$")
ax[1][0].text(x=6, y=0.05, s="$a_1$")
ax[2][0].text(x=6, y=0.05, s="$e$")
ax[0][1].text(x=6, y=0.05, s="$a_1 + e$")
ax[1][1].text(x=6, y=0.05, s="$a_1$")
ax[2][1].text(x=6, y=0.05, s="$e$")

ax[i][0].set_xlabel("$\hbar \omega$ (meV)")
ax[i][1].set_xlabel("$\hbar \omega$ (meV)")

ax[0][0].text(x=-0.2, y=1.1, s="(a)", fontsize=14, color='k', transform=ax[0][0].transAxes)
ax[0][1].text(x=-0.1, y=1.1, s="(b)", fontsize=14, color='k', transform=ax[0][1].transAxes)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
plt.ylabel("S($\hbar \omega$) (meV$^{-1}$)", labelpad=10)
plt.subplots_adjust(wspace=0.15, hspace=0.15)

plt.savefig("figure_s5.pdf", bbox_inches="tight", dpi=300)
plt.show()
