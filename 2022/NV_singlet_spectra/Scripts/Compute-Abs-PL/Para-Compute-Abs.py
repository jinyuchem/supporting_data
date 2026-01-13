#!/usr/bin/env python3

import numpy as np
from scipy.integrate import simps
from scipy import constants
import sys
import time
import multiprocessing as mp

#############
# Functions #
#############

const = constants.eV / 1000 * 1e-15 / constants.hbar

# Calculate sigma for a given freq 
def f_sigma(freq, sigma):
    all_sigma = sigma[0] - (sigma[0] - sigma[1]) \
                / (max(freq) - min(freq)) * (freq[:] - min(freq))
    return all_sigma

# Define S(t), which is the Fourier transform of S(\hbar\omega)
def Re_S(t, hrf, freq, sigma): # The unit of t is fs

    all_sigma = f_sigma(freq, sigma)

    funct = 0
    for i in range(freq.shape[0]):
        funct = funct + hrf[i] * np.exp(-t**2 * const**2 * all_sigma[i]**2/2) \
                               * np.cos(freq[i] * t * const)

    return funct

# Imaginary part of S(t)
def Im_S(t, hrf, freq, sigma): # The unit of t is fs

    all_sigma = f_sigma(freq, sigma)

    funct = 0
    for i in range(freq.shape[0]):
        funct = funct + hrf[i] * np.exp(-t**2 * const**2 * all_sigma[i]**2/2) \
                               * (-1) * np.sin(freq[i] * t * const)

    return funct

# Average occupation number of a phonon mode
def NP(temp, ene): # Unit of ene is meV
    funct = 1/(np.exp((ene * constants.eV / 1000)/(constants.Boltzmann * temp)) - 1)
    return funct

# Define C(t, T), which is the Fourier transform of C(\hbar\omega, T)
# Real part of C(t, T)
def Re_C(t, temp, hrf, freq, sigma):

    all_sigma = f_sigma(freq, sigma)

    funct = 0
    for i in range(freq.shape[0]):
        funct = funct + hrf[i] * np.exp(-t**2 * const**2 * all_sigma[i]**2/2) \
                               * np.cos(freq[i] * t * const) * NP(temp, freq[i])
    
    return funct

# Define G(t, T).
# Real part of G(t, T)
def Re_G(t, temp, hrf, freq, sigma):
    if temp > 0:
        funct = np.exp(Re_S(t, hrf, freq, sigma) \
                       - Re_S(0, hrf, freq, sigma) \
                       + 2 * Re_C(t, temp, hrf, freq, sigma) \
                       - 2 * Re_C(0, temp, hrf, freq, sigma)) \
                * np.cos(Im_S(t, hrf, freq, sigma))
    elif temp == 0:
        funct = np.exp(Re_S(t, hrf, freq, sigma) \
                       - Re_S(0, hrf, freq, sigma)) \
                * np.cos(Im_S(t, hrf, freq, sigma))
    return funct

# Imaginary part of G(t, T)
def Im_G(t, temp, hrf, freq, sigma):
    if temp > 0:
        funct = np.exp(Re_S(t, hrf, freq, sigma) \
                       - Re_S(0, hrf, freq, sigma) \
                       + 2 * Re_C(t, temp, hrf, freq, sigma) \
                       - 2 * Re_C(0, temp, hrf, freq, sigma)) \
                * np.sin(Im_S(t, hrf, freq, sigma))
    elif temp == 0:
        funct = np.exp(Re_S(t, hrf, freq, sigma) \
                       - Re_S(0, hrf, freq, sigma)) \
                * np.sin(Im_S(t, hrf, freq, sigma))
    return funct

#########
# Input #
#########

in_fname = sys.argv[2]

# Temperature in Kelvin
temp = 10

# The standard deviation for the Huang-Rhys spectrum. Unit is meV.
sigma = [6, 2]

# The damping factor determining the broadening of the ZPL. Unit is meV.
lamda = 0.1

# The Fourier transfom will be computed for [0, 20000]. The unit is fs.
time_range = [0, 100000]

# resolution of the time axis
time_reso = 100001

# The range energy offset from the ZPL.
# The PL lineshape will be calculated for the energy range from (ZPL - 550) to (ZPL + 150).
# The unit is meV.
ene_range = [-150, 550]

# The number of points on the computed PL lineshape.
ene_reso = 7001

# The filename to store the data for the PL lineshape.
out_fname = 'Abs-' + in_fname

# number of cores
num_cores = int(sys.argv[1])

#############
# Load data #
#############

# Load the frequencies of all phonon modes
# Change the unit of frequency from cm^{-1} to meV
freq = np.loadtxt(in_fname, usecols=6, dtype='f8')
freq = freq * constants.c * 100 * 2 * np.pi * constants.hbar / constants.eV * 1e3

# Load all partial Huang-Rhys factors
hrf = np.loadtxt(in_fname, usecols=3, dtype='f8')**2

########
# Main #
########

# PL line shape
lsp = np.zeros(ene_reso, dtype="f8")

# Energy axis: meV
eneaxis = np.linspace(ene_range[0], ene_range[1], ene_reso, dtype="f8")

# Time axis: fs
timeaxis = np.linspace(time_range[0], time_range[1], time_reso, dtype="f8")

if __name__ == '__main__':

    start_time = time.time()
    gr = Re_G(timeaxis, temp, hrf, freq, sigma)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    gi = Im_G(timeaxis, temp, hrf, freq, sigma)
    print("--- %s seconds ---" % (time.time() - start_time))

    ex = np.exp( - (np.abs(timeaxis) * lamda * const))
    
    start_time = time.time()
    def compute_int(ene):
        theta = timeaxis * ene * const
        integrand = (gr * np.cos(theta) - gi * np.sin(theta)) * ex
        lsp = simps(integrand, timeaxis)
        return lsp

    pool = mp.Pool(num_cores)
    lsp = pool.map(compute_int, eneaxis) 
    print("--- %s seconds ---" % (time.time() - start_time))

with open(out_fname, 'w') as w: 
   for i in range(0, ene_reso):
      w.write(f"{eneaxis[i]:.4e}   {lsp[i]:.8e}\n")
