#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})
from sklearn.metrics import auc
from scipy import constants

#########
# Input #
#########

folders = ["5K/", "50K/", "100K/", "150K/", "200K/", "250K/", "300K/", "350K/", "400K/", "450K/", "500K/", "550K/", "600K/", "650K/", "700K/"]

fnames = [
    "SC-only-a1-PL-1200-All-B-1A1-3E.dat-a1-modes.dat",
]

########
# Data #
########

eneaxis = np.loadtxt(folders[0] + fnames[0], usecols=0)

data = []
for folder in folders:
    tmp_data = []
    for fname in fnames:
        tmp_tmp_data = np.loadtxt(folder + fname, usecols=1)
        tmp_data.append(tmp_tmp_data)
    data.append(tmp_data)
data = np.array(data)

print(data.shape)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data[i, j] *= 1 / np.pi / constants.hbar * 1e-15 * constants.eV / 1000 * 0.1
        print(sum(data[i, j]))
        print(auc(eneaxis, data[i, j]))
        print()
data *= 1e4

########
# Plot #
########

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

colors = [
    "#4285F4",
    "#DB4437",
    "#F4B400",
    "#0F9D58",
    "#FF6F61",
    "#46BDC6",
    "#FF9E80",
    "#7E57C2",
    "#9CCC65",
]

cmap = plt.cm.plasma
color_values = np.linspace(0, 1, len(folders))
colors = [cmap(value) for value in color_values]


linestyles = ["-", "-", "-", "-", "-", "--", "-", "--"]
labels = [
    "$^3E \\to ^1A_1$, $a_1$, Scale $a_1$",
]
labels_2 = ["5 K", "50 K", "100 K", "150 K", "200 K", "250 K", "300 K", "350 K", "400 K", "450 K", "500 K", "550 K", "600 K", "650 K", "700 K", "750 K", "800 K"]

for i in range(len(fnames)):
    if i == 0:
        for j in range(len(folders)):
            ax.plot(
                eneaxis * 1e-3,
                data[j, i],
                color=colors[j],
                label=labels_2[j],
                linewidth=1,
                linestyle=linestyles[i],
            )
        ax.text(
            x=0.3,
            y=0.68,
            s=labels[i],
            fontsize=12,
            color="k",
            transform=ax.transAxes,
        )

print()
print(0.354)
for a in range(len(folders)):
    for i in range(eneaxis.shape[0]):
        if abs(eneaxis[i] - 354) < 1e-5:
            print(labels_2[a], data[a, 0, i])
print()
print(0.404)
for a in range(len(folders)):
    for i in range(eneaxis.shape[0]):
        if abs(eneaxis[i] - 404) < 1e-5:
            print(labels_2[a], data[a, 0, i])
print()
print(0.454)
for a in range(len(folders)):
    for i in range(eneaxis.shape[0]):
        if abs(eneaxis[i] - 454) < 1e-5:
            print(labels_2[a], data[a, 0, i])
print()
print(0.504)
for a in range(len(folders)):
    for i in range(eneaxis.shape[0]):
        if abs(eneaxis[i] - 504) < 1e-5:
            print(labels_2[a], data[a, 0, i])
print()

ax.grid(color="gray", linestyle="--", linewidth=0.5)
ax.set_xlim((-0.15, 0.68))
ax.set_ylim((0.0, 10.0))
ax.tick_params(direction="in")
ax.legend(fontsize=12, loc="upper right", edgecolor="black", ncols=2)
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
ax.set_xlabel("$\\varepsilon$ (eV)")
ax.set_ylabel("$F(\\varepsilon)$ (eV$^{-1}$)")

# fig.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
# plt.subplots_adjust(wspace=0.2, hspace=0.3)

plt.savefig("temperature_lsp.pdf", bbox_inches="tight", dpi=300)
plt.show()
