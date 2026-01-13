#!/usr/bin/env python3

import numpy as np
import sys
import os
import shutil

# coordinates along the c.c.: 0.0 is the gs structure; 1.0 is the es structure
COD = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
# file contains the gs structure
GS_file = 'con_es_0_cod.in'
# file contains the es structure
ES_1_file = 'con_es_120_cod.in'
ES_2_file = 'con_es_240_cod.in'
# file contains other setting for pw
Pre_file = 'prefix.in'
# file contains the resulting input file for pw
Final_file = 'pw.in'

#############
GS = np.loadtxt(GS_file, usecols=(1,2,3))
ES_1 = np.loadtxt(ES_1_file, usecols=(1,2,3))
ES_2 = np.loadtxt(ES_2_file, usecols=(1,2,3))

DIFF = (ES_1 + ES_2) - GS - GS
DIFF = DIFF * 2 / 3

ATOMS = np.genfromtxt(GS_file, dtype='str', usecols=0)

#############
path_old = os.getcwd() + '/'

filename_o = path_old + Pre_file

for i in range(len(COD)):
    path_new = path_old + 'Image-' + str(i+1) + '/'
    os.mkdir(path_new)
    filename_n = path_new + Final_file
    shutil.copy2(filename_o, filename_n)
    os.chdir(path_new)

    CO_n = GS + DIFF * COD[i]

    with open('pw.in', 'a') as fn:
        for j in range(GS.shape[0]):
            nline = ( str(ATOMS[j]), '   % .12f   % .12f   % .12f'%(CO_n[j,0], CO_n[j,1], CO_n[j,2]), '\n')
            fn.writelines(nline)
        fn.close()

    with open('cod.in', 'w') as fn:
        for j in range(GS.shape[0]):
            nline = ( str(ATOMS[j]), '   % .12f   % .12f   % .12f'%(CO_n[j,0], CO_n[j,1], CO_n[j,2]), '\n')
            fn.writelines(nline)
        fn.close()

    os.chdir(path_old)
