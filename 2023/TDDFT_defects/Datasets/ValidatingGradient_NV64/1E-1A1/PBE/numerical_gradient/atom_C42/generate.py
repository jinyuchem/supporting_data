#!/usr/bin/env python3

import numpy as np
import sys
import os
import shutil

##############
# Parameters #
##############

index = 42
dis_mag = 0.0005

geo_file = 'cod.in'
pre_file = 'prefix.in'
final_file = 'pw.in'
dir_name = ['x', 'y', 'z']
dis_name = ['m2', 'm1', 'p1', 'p2']
displacements = [dis_mag * (-2), dis_mag * (-1),
                 dis_mag * 1, dis_mag * 2]

####################
# Load coordinates #
####################

geo = np.loadtxt(geo_file, usecols=(1,2,3))
atoms = np.genfromtxt(geo_file, dtype='str', usecols=0)

##################
# Write to files #
##################

path_old = os.getcwd() + '/'

filename_o = path_old + pre_file

# x, y, z
for i in range(3):
    for j in range(4):
        path_new = path_old + dir_name[i] + dis_name[j] + '/'
        os.mkdir(path_new)
        filename_n = path_new + final_file
        shutil.copy2(filename_o, filename_n)
        os.chdir(path_new)

        geo_new = np.copy(geo)
        geo_new[index - 1,i] += displacements[j]

        with open('pw.in', 'a') as fn:
            for k in range(geo.shape[0]):
                nline = (str(atoms[k]), '   % .12f   % .12f   % .12f'%(
                geo_new[k,0], geo_new[k,1], geo_new[k,2]), '\n')
                fn.writelines(nline)
            fn.close()

        with open('cod.in', 'w') as fn:
            for k in range(geo.shape[0]):
                nline = (str(atoms[k]), '   % .12f   % .12f   % .12f'%(
                geo_new[k,0], geo_new[k,1], geo_new[k,2]), '\n')
                fn.writelines(nline)
            fn.close()

        os.chdir(path_old)
