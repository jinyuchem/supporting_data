#!/usr/bin/env python

import numpy as np
import sys
import os
import shutil
import json
import subprocess

#############
# Functions #
#############

def read_wbse_eigenvalues(fileName):
    """
    Read eigenvalues from JSON file.
    """

    with open(fileName, "r") as f:
        raw_ = json.load(f)

    return np.array(raw_["exec"]["davitr"][-1]["ev"], dtype=float)


def grad_4p(x, dis):
    gg = (-x[0] + 8*x[1] - 8*x[2] + x[3]) / (12 * dis / 0.529177249)
    return gg


def grad_2p_large(x, dis):
    gg = (x[0] - x[3]) / (4 * dis / 0.529177249)
    return gg


def grad_2p_small(x, dis):
    gg = (x[1] - x[2]) / (2 * dis / 0.529177249)
    return gg

################
# Load results #
################

dir_names = [
'xm2', 'xm1', 'xp1', 'xp2',
'ym2', 'ym1', 'yp1', 'yp2',
'zm2', 'zm1', 'zp1', 'zp2',
]

alleigs = []
for directory in dir_names:
    fname = directory + '/pwscf.wbse.save/wbse.json'
    eigs = read_wbse_eigenvalues(fname)
    alleigs.append(eigs)
alleigs = np.array(alleigs)

alleigs = np.reshape(alleigs, (3,4,4))

####################
# Compute gradient #
####################

dis = 0.0005 * 2 *  3.568
nstate = 4

for i in range(nstate):
    # x, y, z
    print('State %d'%(i+1))
    print('4p')
    for j in range(3):
        dd = alleigs[j,:,i]
        gg_4p = grad_4p(dd, dis)
        print('% .10f'%gg_4p)
    print('')

    # x, y, z
    print('2p large')
    for j in range(3):
        dd = alleigs[j,:,i]
        gg_2p_l = grad_2p_large(dd, dis)
        print('% .10f'%gg_2p_l)
    print('')

    # x, y, z
    print('2p small')
    for j in range(3):
        dd = alleigs[j,:,i]
        gg_2p_s = grad_2p_small(dd, dis)
        print('% .10f'%gg_2p_s)
    print('')
