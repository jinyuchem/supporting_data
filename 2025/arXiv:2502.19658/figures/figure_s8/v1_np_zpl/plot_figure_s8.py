#!/usr/bin/env python3
import numpy as np
from scipy import constants
from scipy.special import eval_hermite
from scipy.special import comb, factorial, factorial2
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from decimal import *

plt.rcParams.update({"font.size": 12})

#############
# Functions #
#############


# fc integral
def fcf_mine(ni, nf, wi, wf, k):
    wi = wi * constants.eV  # J
    wi = wi / constants.hbar  # rad \cdot s^{-1}
    wi = wi / constants.hbar  # J^{-1} \cdot rad \cdot s^{-2}
    wf = wf * constants.eV  # J
    wf = wf / constants.hbar  # rad \cdot s^{-1}
    wf = wf / constants.hbar  # J^{-1} \cdot rad \cdot s^{-2}
    # kg^{0.5} \cdot meter
    k = k * constants.physical_constants["atomic mass constant"][0] ** 0.5 * 1e-10

    a = (wi - wf) / (wi + wf)
    b = 2 * k * np.sqrt(wi) * wf / (wi + wf)
    c = -a
    d = -2 * k * np.sqrt(wf) * wi / (wi + wf)
    e = 4 * np.sqrt(wi * wf) / (wi + wf)
    f = np.zeros((ni, nf))

    f[0, 0] = (e / 2) ** 0.5 * np.exp(b * d / 2 / e)

    f[0, 1] = 1 / np.sqrt(2) * d * f[0, 0]
    for j in range(2, nf):
        f[0, j] = (
            1 / np.sqrt(2 * j) * d * f[0, j - 1]
            + np.sqrt((j - 1) / j) * c * f[0, j - 2]
        )

    f[1, 0] = 1 / np.sqrt(2) * b * f[0, 0]
    for i in range(2, ni):
        f[i, 0] = (
            1 / np.sqrt(2 * i) * b * f[i - 1, 0]
            + np.sqrt((i - 1) / i) * a * f[i - 2, 0]
        )

    for j in range(1, nf):
        f[1, j] = (
            1 / np.sqrt(2 * 1) * b * f[0, j] + 1 / 2 * np.sqrt(j / 1) * e * f[0, j - 1]
        )

    for i in range(2, ni):
        for j in range(1, nf):
            f[i, j] = (
                1 / np.sqrt(2 * i) * b * f[i - 1, j]
                + np.sqrt((i - 1) / i) * a * f[i - 2, j]
                + 1 / 2 * np.sqrt(j / i) * e * f[i - 1, j - 1]
            )
    return f


# fc integral with Q inside
def fcf_q(ni, nf, wi, wf, k):
    fq = np.zeros((ni, nf))
    f = fcf_mine(ni, nf, wi, wf, k)
    for i in range(ni):
        for j in range(nf):
            if i == 0:
                fq[i, j] = np.sqrt(i + 1) * f[i + 1, j]
            elif i == ni - 1:
                fq[i, j] = np.sqrt(i) * f[i - 1, j]
            else:
                fq[i, j] = np.sqrt(i + 1) * f[i + 1, j] + np.sqrt(i) * f[i - 1, j]
    # change unit from eV to rad \cdot s^{-1}
    wi = wi * constants.eV / constants.hbar
    prefactor = np.sqrt(constants.hbar / 2 / wi)
    fq = fq * prefactor
    # the unit of the fq is kg^{1/2} \cdot m
    return fq


# gaussian function
def Gaussian(x, mu, sigma_r):
    sigma = (sigma_r[1] - sigma_r[0]) * (mu / 160) + sigma_r[0]
    pref = 1 / np.sqrt(2 * np.pi * sigma**2)
    expp = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return pref * expp


# lorentzian function
def Lorentzian(x, mu, gamma):
    pref = 1 / (np.pi * gamma)
    mp = gamma**2 / ((x - mu) ** 2 + gamma**2)
    return pref * mp


# BZ distribution
def BZ(ene, T):  # unit is meV and K
    return np.exp(-(ene * 1e-3 * constants.eV) / (constants.k * T))


# compute fc line shape
def bulid_fc_lsp(
    mu, eneaxis, order_a, order_b, freq_a, fc_int_v, energy_v, Temp, sigma
):
    lsp = np.zeros(eneaxis.shape[0])
    for ind_a in range(order_a):
        pref = BZ(ind_a * freq_a, Temp) * mu**2
        for ind_b in range(order_b):
            if ind_a == 0 and ind_b == 0:
                continue
            else:
                lsp[:] = lsp[:] + pref * fc_int_v[ind_b, ind_a] ** 2 * Gaussian(
                    eneaxis[:], energy_v[ind_a, ind_b], sigma
                )
    return lsp


# compute fcht line shape
def bulid_fcht_lsp(
    mu,
    DmuDq,
    eneaxis,
    order_a,
    order_b,
    freq_a,
    fc_int_v,
    fc_int_q,
    energy_v,
    Temp,
    sigma,
):
    fcht_lsp = np.zeros(eneaxis.shape[0])
    for ind_a in range(order_a):
        pref = BZ(ind_a * freq_a, Temp) * 2 * mu * DmuDq
        for ind_b in range(order_b):
            if ind_a == 0 and ind_b == 0:
                continue
            else:
                fcht_lsp[:] = fcht_lsp[:] + pref * fc_int_v[ind_b, ind_a] * fc_int_q[
                    ind_b, ind_a
                ] * Gaussian(eneaxis[:], energy_v[ind_a, ind_b], sigma)
    return fcht_lsp


# compute ht line shape
def bulid_ht_lsp(
    DmuDq, eneaxis, order_a, order_b, freq_a, fc_int_q, energy_v, Temp, sigma
):
    ht_lsp = np.zeros(eneaxis.shape[0])
    for ind_a in range(order_a):
        pref = BZ(ind_a * freq_a, Temp) * DmuDq**2
        for ind_b in range(order_b):
            if ind_a == 0 and ind_b == 0:
                continue
            else:
                ht_lsp[:] = ht_lsp[:] + pref * fc_int_q[ind_b, ind_a] ** 2 * Gaussian(
                    eneaxis[:], energy_v[ind_a, ind_b], sigma
                )
    return ht_lsp


if __name__ == "__main__":

    red = "#DB4437"
    blue = "#4285F4"
    green = "#0F9D58"

    #########
    # Input #
    #########

    # 3A2 is initial and 3E is final
    # \lambda_\perp (unitless for now)
    mu = 1.0
    # unit: 1/(amu^1/2 \AA) # \lambda_\perp increases from 3E geo to 1A1 / 3A2 geo
    DmuDq = - (54.478384245865115 - 50.26757631745079) / 54.478384245865115 / 0.476630
    # at 1A1 geo: 54.478384245865115 GHz, final
    # at 3E(a1) geo: 50.26757631745079 GHz, initial

    # initial is at 0.0
    # final is at -|Delta Q| = raw_q

    # input freq (meV)
    # from the npj paper
    freq_a = 63  # initial
    freq_b = 63  # final

    # input q (amu^1/2 \AA)
    raw_q = 0.476630 * np.sqrt((3.49 - 0.59) / 1.85)

    # order
    order_a = 40
    order_b = 40

    # energy range for plot
    ene_range = [-150, 1200]

    # resolution
    resol = 13501

    # temperature: K
    Temp = 5

    # broadening
    gamma = 0.1
    sigma = [12, 25]

    # ZPL: meV
    ZPL = 0

    ################
    # FC Integrals #
    ################

    # unit conversion, from meV to eV
    alpha_a = freq_a * 1e-3
    alpha_b = freq_b * 1e-3

    # unit conversion, from 1/(amu^{1/2} \cdot \AA) to 1 / kg^{1/2} / m
    DmuDq = (
        DmuDq
        * 1
        / (constants.physical_constants["atomic mass constant"][0] ** 0.5 * 1e-10)
    )

    # compute fc integrals
    fc_int_v = fcf_mine(order_a, order_b, alpha_a, alpha_b, raw_q)

    # compute fc_q integrals # zero point of Q axis is at the local minimum of the ES
    fc_int_q = fcf_q(order_a, order_b, alpha_a, alpha_b, raw_q)

    ##############
    # line shape #
    ##############

    lineshape = True
    if lineshape == True:

        # compute the energies corresponding to fc intergrals
        energy_v = np.zeros((order_a, order_b))
        for ind_a in range(order_a):
            for ind_b in range(order_b):
                energy_v[ind_a, ind_b] = 0 - ind_a * freq_a + ind_b * freq_b

        # line shape
        eneaxis = np.linspace(ene_range[0], ene_range[1], resol)

        fc_lsp = np.zeros(resol)
        fc_lsp = bulid_fc_lsp(
            mu, eneaxis, order_a, order_b, freq_a, fc_int_v, energy_v, Temp, sigma
        )

        fcht_lsp = np.zeros(resol)
        fcht_lsp = bulid_fcht_lsp(
            mu,
            DmuDq,
            eneaxis,
            order_a,
            order_b,
            freq_a,
            fc_int_v,
            fc_int_q,
            energy_v,
            Temp,
            sigma,
        )

        ht_lsp = np.zeros(resol)
        ht_lsp = bulid_ht_lsp(
            DmuDq, eneaxis, order_a, order_b, freq_a, fc_int_q, energy_v, Temp, sigma
        )

        # normalization with respect to the FC lsp
        all_lsp = fc_lsp + fcht_lsp + ht_lsp
        print("Sum of FC is %.5f"% (sum(abs(fc_lsp))))
        print("Sum of FCHT is %.5f"% (sum(abs(fcht_lsp))))
        print("Sum of HT is %.5f"% (sum(abs(ht_lsp))))
        print("Sum of Total is %.5f"% (sum(abs(all_lsp))))
        print()

        # sum
        print("Sum of FC is %.5f"% (sum(abs(fc_lsp))))
        print("Sum of FCHT is %.5f"% (sum(abs(fcht_lsp))))
        print("Sum of HT is %.5f"% (sum(abs(ht_lsp))))
        print("Sum of Total is %.5f"% (sum(abs(all_lsp))))

        # fraction
        print("Fraction of abs FC is %.5f" % (sum(abs(fc_lsp)) / sum(all_lsp)))
        print("Fraction of abs FCHT is %.5f" % (sum(abs(fcht_lsp)) / sum(all_lsp)))
        print("Fraction of abs HT is %.5f" % (sum(abs(ht_lsp)) / sum(all_lsp)))

        print("Fraction of FC is %.5f" % (sum(fc_lsp) / sum(all_lsp)))
        print("Fraction of FCHT is %.5f" % (sum(fcht_lsp) / sum(all_lsp)))
        print("Fraction of HT is %.5f" % (sum(ht_lsp) / sum(all_lsp)))

        # write to file
        with open("scaled-q-FC-NV-Temp-" + str(Temp) + "-lsp.dat", "w") as w:
            nline = (
                "Energy (meV)          FC          FCHT        HT        TOT",
                "\n",
            )
            w.writelines(nline)
            for i in range(fc_lsp.shape[0]):
                nline = (
                    "%.8e    %.8e    %.8e    %.8e    %.8e"
                    % (eneaxis[i], fc_lsp[i], fcht_lsp[i], ht_lsp[i], all_lsp[i]),
                    "\n",
                )
                w.writelines(nline)
            w.close()

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.plot(eneaxis * 1e-3, fc_lsp * 1e3, color=red, linewidth=1, linestyle="-", label="FC")
        ax.plot(
            eneaxis * 1e-3, fcht_lsp * 1e3, color=blue, linewidth=1, linestyle="-", label="FCHT"
        )
        ax.plot(eneaxis * 1e-3, ht_lsp * 1e3, color=green, linewidth=1, linestyle="-", label="HT")
        ax.plot(
            eneaxis * 1e-3, all_lsp * 1e3, color="black", linewidth=1, linestyle="-", label="Total"
        )

        ax.legend(fontsize=12, loc="upper right", edgecolor="black")
        ax.set_xlim((-0.05, 0.6))
        ax.set_ylim((-1.5, 6))

        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        ax.tick_params(direction="in")
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.set_xlabel("$\\varepsilon$ (eV)")
        ax.set_ylabel("$F(\\varepsilon)$ (eV$^{-1}$)")

        plt.savefig("figure_s8.pdf", dpi=300, bbox_inches="tight")
        plt.show()
