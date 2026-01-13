#!/usr/bin/env python3
import numpy as np
from copy import deepcopy
import sys
from scipy.linalg import eigh, eig
from scipy import constants
import h5py

class ATOM:

    def __init__(self, index, element):
        self.index = index
        self.element = element
        self.cart_cod = np.empty([0,3])
        self.cry_cod = np.empty([0,3])

    def set_mass(self, mass):
        self.mass = mass

    def set_cart(self, cod):
        self.cart_cod = cod[np.newaxis, :]

    def set_cry(self, cod):
        self.cry_cod = cod[np.newaxis, :]

    def cart2cry(self, LP):
        self.cry_cod = np.dot(self.cart_cod, np.linalg.inv(LP))

    def cry2cart(self, LP):
        self.cart_cod = np.dot(self.cry_cod, LP)
    
    def set_content(self):
        ctt = {
            "index": self.index,
            "element": self.element,
            "mass": self.mass,
            "cart_cod": self.cart_cod,
            "cry_cod": self.cry_cod
        }
        self.content = ctt

    def dis(self, n_cart_cod):
        dis_v = self.cart_cod - n_cart_cod
        dis_n = np.dot(dis_v, dis_v)
        return dis_n

class CELL:

    def __init__(self, LP):
        self.noa = 0
        self.LP = LP
        self.atoms = []
        self.cart_cods = np.empty([0,3])
        self.cry_cods = np.empty([0,3])
    
    def add_atom(self, atom):
        self.atoms.append(atom)
        self.noa = self.noa + 1
        self.cart_cods = np.append(self.cart_cods, atom.cart_cod, axis=0)
        self.cry_cods = np.append(self.cry_cods, atom.cry_cod, axis=0)

    def dis_atom(self, ai, aj):
        cart1 = self.atoms[ai].cart_cod
        cart2 = self.atoms[aj].cart_cod
        dis_v = cart1[0] - cart2[0]
        dis_n = np.sqrt(np.dot(dis_v, dis_v))
        return dis_n
    
    def write2qefile(self, filename, format):
        with open(filename, 'w') as w:
            for ind in range(self.noa):
                ct = self.atoms[ind].element + "   "
                if format == 'cry':
                    cod = self.atoms[ind].cry_cod[0]
                elif format == 'cart':
                    cod = self.atoms[ind].cart_cod[0]
                cod = '  % 13.9f   % 13.9f   % 13.9f'%(cod[0], cod[1], cod[2])
                ct = ct + cod
                nline = (ct, '\n')
                w.writelines(nline)
            w.close()
    
    def readfqefile(self, filename, nol, file_fmt):
        with open(filename, 'r') as f:
            for ind in range(nol):
                line = f.readline()
            
                index = ind
                element = line.split()[0]
                atom = ATOM(index, element)
            
                cry_cod = line.split()[1:]
                cry_cod = np.asarray(cry_cod, dtype=np.float64)

                if file_fmt == 'cry':
                    atom.set_cry(cry_cod)
                    atom.cry2cart(self.LP)
                elif file_fmt == 'cart':
                    atom.set_cart(cry_cod)
                    atom.cart2cry(self.LP)

                mass = 12.0107
                atom.set_mass(mass)
                atom.set_content()

                self.add_atom(atom)
            f.close()
        

def build_trial_cell(dim):
    dim = np.asarray(dim)

    unit_cell = np.zeros((8,3))
    unit_cell[0] = [0.0, 0.0, 0.0]
    unit_cell[1] = [0.0, 0.5, 0.5]
    unit_cell[2] = [0.5, 0.0, 0.5]
    unit_cell[3] = [0.5, 0.5, 0.0]
    unit_cell[4] = [0.25, 0.25, 0.25]
    unit_cell[5] = [0.75, 0.75, 0.25]
    unit_cell[6] = [0.25, 0.75, 0.75]
    unit_cell[7] = [0.75, 0.25, 0.75]

    for ic in range(3):
        unit_cell[:,ic] = unit_cell[:,ic]/dim[ic]
    
    trial_cry_cods = np.zeros((dim[0], dim[1], dim[2], 8, 3))
    for ix in range(dim[0]):
        for iy in range(dim[1]):
            for iz in range(dim[2]):
                for ind in range(8):
                        trial_cry_cods[ix,iy,iz,ind,0] = unit_cell[ind,0] + ix * 1/dim[0]
                        trial_cry_cods[ix,iy,iz,ind,1] = unit_cell[ind,1] + iy * 1/dim[1]
                        trial_cry_cods[ix,iy,iz,ind,2] = unit_cell[ind,2] + iz * 1/dim[2]
    
    trial_cry_cods = np.reshape(trial_cry_cods, (np.prod(dim)*8,3))

    return trial_cry_cods

def load_fc(filename):
    with open(filename, 'r') as f:
        noa = int(f.readline().split()[0])
        f.close()
    fc_mat = np.zeros((noa, noa, 3, 3))
    with open(filename, 'r') as f:
        f.readline()
        for ind in range(noa):
            for jnd in range(noa):
                f.readline()
                for ic in range(3):
                    line = f.readline()
                    fc_mat[ind,jnd,ic] = np.asarray(line.split()[:])
        f.close()
    return fc_mat

def write_fc(filename, fc_mat):
    noa = fc_mat.shape[0]
    with open(filename, 'w') as w:
        nline = ('% 5d    % 5d'%(noa, noa), '\n')
        w.writelines(nline)
        for ind in range(noa):
            for jnd in range(noa):
                nline = ('% 5d    % 5d'%(ind+1, jnd+1), '\n')
                w.writelines(nline)
                for ic in range(3):
                    nline = ('% 18.15f    % 18.15f    % 18.15f'%(
                        fc_mat[ind,jnd,ic,0], fc_mat[ind,jnd,ic,1], fc_mat[ind,jnd,ic,2]
                    ), '\n')
                    w.writelines(nline)
        w.close()

# Distance between pairs considering the periodic interaction
def dis_pair(cell, ind_a, ind_b):
    dis = cell.cart_cods[ind_a] - cell.cart_cods[ind_b]
    for ic in range(3):
        if dis[ic] > 0.5 * cell.LP[ic,ic]:
            dis[ic] = dis[ic] - cell.LP[ic,ic]
        elif dis[ic] < -0.5 * cell.LP[ic,ic]:
            dis[ic] = dis[ic] + cell.LP[ic,ic]
    return dis

# Minimum distance to nv center
# New version, only one sphere
def dis_nv_2(cart_cod, cell):
    nitrogen_cart_cod = cell.cart_cods[214]
    vacancy_cart_cod = np.dot([0.75, 0.58356, 0.58356], NV216cell.LP)
    center_cart_cod = 0.5 * (nitrogen_cart_cod + vacancy_cart_cod)
    dis_cen = cart_cod - center_cart_cod
    for ic in range(3):
        if dis_cen[ic] > 0.5 * cell.LP[ic,ic]:
            dis_cen[ic] = dis_cen[ic] - cell.LP[ic,ic]
        elif dis_cen[ic] < -0.5 * cell.LP[ic,ic]:
            dis_cen[ic] = dis_cen[ic] + cell.LP[ic,ic]
    dis_cen = np.linalg.norm(dis_cen)
    return dis_cen

# Separate the carbon atoms into two sets
def dec_set(cry_cod, dim):
    aux = cry_cod * dim * 4 
    aux = np.rint(aux)
    aux = np.array(aux, dtype=int)
    # +0.00 +0.00 +0.00
    if aux[0]%2==0 and aux[1]%2==0 and aux[2]%2==0:
        set_ind = 0
    # +0.25 +0.25 +0.25 case
    elif aux[0]%2==1 and aux[1]%2==1 and aux[2]%2==1:
        set_ind = 1
    return set_ind

# Find the correspondance of an atom from a larger supercell in the 512 nv cell
def f_216nv_ind(cell, shape, ind):
    ref_cry_set = cell.cry_cods[0:215]

    old_cry_cod = cell.cry_cods[ind]
    new_cry_cod = np.copy(old_cry_cod)

    # Shift x, y, z
    for ic in range(3):
        if old_cry_cod[ic] > (3/shape[ic] - 0.001):
            new_cry_cod[ic] = old_cry_cod[ic] - 3/shape[ic]

    # Compute the difference matrix between the reference atoms
    aux_dis = ref_cry_set - new_cry_cod
    aux_norm = np.linalg.norm(aux_dis, axis=1)
    n_ind = np.argmin(aux_norm)
    error = np.amin(aux_norm) * cell.LP[0,0]

    return n_ind, error

if __name__ == "__main__":

    ####################
    # Input Parameters #
    ####################

    # NV in diamond in a new cell
    # Shape of the new cell: only support cubic cell at this moment
    shape_new_cell = [12,12,12]
    N_LP = np.array([
        [3.568, 0.0, 0.0],
        [0.0, 3.568, 0.0],
        [0.0, 0.0, 3.568]
    ])
    N_LP[0] = N_LP[0] * shape_new_cell[0]
    N_LP[1] = N_LP[1] * shape_new_cell[1]
    N_LP[2] = N_LP[2] * shape_new_cell[2]
    f_name = 'new-nv-cell'

    # Threshold for pair distance: use 5 \AA in ref.
    radius_thres_1 = 5
    # Threshold for the distance from NV: use 5 \AA in ref.
    radius_thres_2 = 5
    # Newly added threshold for environment part
    radius_thres_3 = radius_thres_1

    # Correct the diagonal elements
    correct_diag = True

    #######################
    # Original supercells #
    #######################
    
    # NV in diamond in a 216 atoms cell
    LP = np.array([
        [10.704, 0.0, 0.0],
        [0.0, 10.704, 0.0],
        [0.0, 0.0, 10.704]
        ])
    nol = 215
    NV216cell = CELL(LP)
    NV216cell.readfqefile('nv-216-cod.in', nol, 'cry')
    NV216cell.atoms[-1].set_mass(14.0067)

    # Pristine diamond in a 512 atoms cell
    LP = np.array([
        [14.272, 0.0, 0.0],
        [0.0, 14.272, 0.0],
        [0.0, 0.0, 14.272]
        ])
    nol = 512
    Dia512cell = CELL(LP)
    Dia512cell.readfqefile('dia-512-cod.in', nol, 'cry')
    
    #################
    # New supercell #
    #################

    # Build new nv cell
    new_NVcell = CELL(N_LP)
    
    shift = 0.0

    # Add atoms from the 216 cell
    for ind in range(NV216cell.noa):
        new_atom = deepcopy(NV216cell.atoms[ind])
        new_atom.cart_cod = new_atom.cart_cod + shift
        new_atom.cart2cry(new_NVcell.LP)
        new_NVcell.add_atom(new_atom)

    # Fill more atoms is needed
    if shape_new_cell[0] > 3:
        # Build a trial cell with the desired shape
        trial_cell = build_trial_cell(shape_new_cell)
        count = NV216cell.noa - 1
        # Loop over all atoms in the trial cell
        for ind in range(trial_cell.shape[0]):
            if trial_cell[ind,0] > (3/shape_new_cell[0]-0.001) or \
                trial_cell[ind,1] > (3/shape_new_cell[1]-0.001) or \
                trial_cell[ind,2] > (3/shape_new_cell[2]-0.001):

                count = count + 1
                atom = ATOM(count, 'C')
                atom.set_cry(trial_cell[ind])
                atom.cry2cart(new_NVcell.LP)
                atom.set_mass(12.0107)
                atom.set_content()
                new_NVcell.add_atom(atom)

    # Write new cells in file    
    new_NVcell.write2qefile(f_name + '-cart-cod.in', 'cart')
    new_NVcell.write2qefile(f_name + '-cry-cod.in', 'cry')

    ################################
    # Load and analyze FC matrices #
    ################################

    # Load force constant matrix
    NV216_fc_mat = load_fc('nv-216-1A1-fc-0614.dat')
    Dia512_fc_mat = load_fc('dia-512-fc.dat')

    # Separate the carbon atoms into two sets
    A_Dia512_cart_cods = []
    A_Dia512_inds = []
    B_Dia512_cart_cods = []
    B_Dia512_inds = []

    for ind in range(Dia512cell.noa):
        set_ind = dec_set(Dia512cell.cry_cods[ind], 4)
        if set_ind == 0:
            A_Dia512_inds.append(ind)
            A_Dia512_cart_cods.append(Dia512cell.cart_cods[ind])
        elif set_ind == 1:
            B_Dia512_inds.append(ind)
            B_Dia512_cart_cods.append(Dia512cell.cart_cods[ind])

    A_Dia512_cart_cods = np.array(A_Dia512_cart_cods)
    A_Dia512_inds = np.array(A_Dia512_inds)
    B_Dia512_cart_cods = np.array(B_Dia512_cart_cods)
    B_Dia512_inds = np.array(B_Dia512_inds)

    A_Dia512_fc_mat = Dia512_fc_mat[A_Dia512_inds]
    B_Dia512_fc_mat = Dia512_fc_mat[B_Dia512_inds]

    A_Dia512_fc_mat = np.reshape(A_Dia512_fc_mat, (A_Dia512_inds.shape[0]*Dia512cell.noa,3,3))
    B_Dia512_fc_mat = np.reshape(B_Dia512_fc_mat, (B_Dia512_inds.shape[0]*Dia512cell.noa,3,3))

    # Analyze the atom pairs of pristine diamond
    A_Full_Pair_Matrix = np.zeros((A_Dia512_inds.shape[0], Dia512cell.noa, 3))
    B_Full_Pair_Matrix = np.zeros((B_Dia512_inds.shape[0], Dia512cell.noa, 3))

    for ind in range(A_Dia512_inds.shape[0]):
        for jnd in range(Dia512cell.noa):
            A_Full_Pair_Matrix[ind,jnd] = dis_pair(Dia512cell, A_Dia512_inds[ind], jnd)
    A_Full_Pair_Matrix = np.reshape(A_Full_Pair_Matrix, (A_Dia512_inds.shape[0]*Dia512cell.noa,3))

    for ind in range(B_Dia512_inds.shape[0]):
        for jnd in range(Dia512cell.noa):
            B_Full_Pair_Matrix[ind,jnd] = dis_pair(Dia512cell, B_Dia512_inds[ind], jnd)
    B_Full_Pair_Matrix = np.reshape(B_Full_Pair_Matrix, (B_Dia512_inds.shape[0]*Dia512cell.noa,3))

    # Find the unique pairs in the pristine diamond cell
    A_Reduced_Pair_Matrix_Indices = \
    np.unique(A_Full_Pair_Matrix.round(decimals=2), return_index=True, axis=0)[1]
    A_Reduced_Pair_Matrix_Indices = np.sort(A_Reduced_Pair_Matrix_Indices)
    A_Reduced_Pair_Matrix = A_Full_Pair_Matrix[A_Reduced_Pair_Matrix_Indices]

    B_Reduced_Pair_Matrix_Indices = \
    np.unique(B_Full_Pair_Matrix.round(decimals=2), return_index=True, axis=0)[1]
    B_Reduced_Pair_Matrix_Indices = np.sort(B_Reduced_Pair_Matrix_Indices)
    B_Reduced_Pair_Matrix = B_Full_Pair_Matrix[B_Reduced_Pair_Matrix_Indices]

    # Find the unique 3*3 force constant matrix in the pristine diamond cell
    A_Dia512_unique_fc_mat = A_Dia512_fc_mat[A_Reduced_Pair_Matrix_Indices]
    B_Dia512_unique_fc_mat = B_Dia512_fc_mat[B_Reduced_Pair_Matrix_Indices]

    #######################
    # Build new FC matrix #
    #######################

    # Build the force constant matrix for the new nv cell
    new_NV_fc_mat = np.zeros((new_NVcell.noa, new_NVcell.noa, 3, 3))

    # Part-0: The part copied from the 216 nv cell
    for ind in range(NV216cell.noa):
        for jnd in range(NV216cell.noa):
            ddd = dis_pair(new_NVcell, ind, jnd)
            # Only pairs with distance smaller than radius_thred_1 are set to non-zero
            if np.linalg.norm(ddd) < radius_thres_1:
                # If the pairs are closer to the nv, then use the defective fc matrix
                if dis_nv_2(new_NVcell.cart_cods[ind], new_NVcell) < radius_thres_2 and \
                    dis_nv_2(new_NVcell.cart_cods[jnd], new_NVcell) < radius_thres_2:
                    new_NV_fc_mat[ind,jnd] = NV216_fc_mat[ind,jnd]
                # Use the bulk fc matrix instead
                else:
                    set_ind = dec_set(new_NVcell.cry_cods[ind], shape_new_cell[0])
                    if set_ind == 0:
                        Aux_Pair_Dis = A_Reduced_Pair_Matrix - ddd
                        Aux_Pair_Norm = np.linalg.norm(Aux_Pair_Dis, axis=1)
                        min_Dis = np.amin(Aux_Pair_Norm)
                        if min_Dis < 0.5:
                            new_NV_fc_mat[ind,jnd] = A_Dia512_unique_fc_mat[np.argmin(Aux_Pair_Norm)]
                    elif set_ind == 1:
                        Aux_Pair_Dis = B_Reduced_Pair_Matrix - ddd 
                        Aux_Pair_Norm = np.linalg.norm(Aux_Pair_Dis, axis=1)
                        min_Dis = np.amin(Aux_Pair_Norm)
                        if min_Dis < 0.5:
                            new_NV_fc_mat[ind,jnd] = B_Dia512_unique_fc_mat[np.argmin(Aux_Pair_Norm)]

    # Compute the difference between the new and the old fc mat
    fc_diff = new_NV_fc_mat[0:NV216cell.noa, 0:NV216cell.noa] - NV216_fc_mat
    print('Norm of the fc diff is % .5f'%(np.linalg.norm(fc_diff)))

    fname = 'rc1-' + str(radius_thres_1) + '-rc2-' + str(radius_thres_2) \
        + '-rc3-' + str(radius_thres_3) + '-fc-diff.dat'
    with open(fname, 'w') as w:
        for ind in range(NV216cell.noa):
            for jnd in range(NV216cell.noa):
                #if np.linalg.norm(fc_diff[ind,jnd]) > 1.0e-10:
                 nline = ('%5d  %5d'%(ind+1, jnd+1), '\n')
                 w.writelines(nline)
                 nline = ('% .14f   % .14f   % .14f'%(fc_diff[ind,jnd,0,0],
                          fc_diff[ind,jnd,0,1], fc_diff[ind,jnd,0,2]), '\n')
                 w.writelines(nline)
                 nline = ('% .14f   % .14f   % .14f'%(fc_diff[ind,jnd,1,0],
                          fc_diff[ind,jnd,1,1], fc_diff[ind,jnd,1,2]), '\n')
                 w.writelines(nline)
                 nline = ('% .14f   % .14f   % .14f'%(fc_diff[ind,jnd,2,0],
                          fc_diff[ind,jnd,2,1], fc_diff[ind,jnd,2,2]), '\n')
                 w.writelines(nline)
        w.close()

    # Part-1:
    for ind in range(NV216cell.noa):
        for jnd in range(NV216cell.noa, new_NVcell.noa):
            ddd = dis_pair(new_NVcell, ind, jnd)
            # Only pairs with distance smaller than radius_thred_1 are set to non-zero
            if np.linalg.norm(ddd) < radius_thres_3:
                # If the pairs are closer to the nv, then use the defective fc matrix
                if dis_nv_2(new_NVcell.cart_cods[ind], new_NVcell) < radius_thres_2 and \
                    dis_nv_2(new_NVcell.cart_cods[jnd], new_NVcell) < radius_thres_2:

                    n_jnd, error = f_216nv_ind(new_NVcell, shape_new_cell, jnd)
                    new_NV_fc_mat[ind,jnd] = NV216_fc_mat[ind,n_jnd]
                # Use the bulk fc matrix instead
                else:
                    set_ind = dec_set(new_NVcell.cry_cods[ind], shape_new_cell[0])
                    if set_ind == 0:
                        Aux_Pair_Dis = A_Reduced_Pair_Matrix - ddd
                        Aux_Pair_Norm = np.linalg.norm(Aux_Pair_Dis, axis=1)
                        min_Dis = np.amin(Aux_Pair_Norm)
                        if min_Dis < 0.5:
                            new_NV_fc_mat[ind,jnd] = A_Dia512_unique_fc_mat[np.argmin(Aux_Pair_Norm)]
                    elif set_ind == 1:
                        Aux_Pair_Dis = B_Reduced_Pair_Matrix - ddd
                        Aux_Pair_Norm = np.linalg.norm(Aux_Pair_Dis, axis=1)
                        min_Dis = np.amin(Aux_Pair_Norm)
                        if min_Dis < 0.5:
                            new_NV_fc_mat[ind,jnd] = B_Dia512_unique_fc_mat[np.argmin(Aux_Pair_Norm)]

    # Part-2:
    for ind in range(NV216cell.noa, new_NVcell.noa):
        for jnd in range(NV216cell.noa):
            ddd = dis_pair(new_NVcell, ind, jnd)
            # Only pairs with distance smaller than radius_thred_1 are set to non-zero
            if np.linalg.norm(ddd) < radius_thres_3:
                if dis_nv_2(new_NVcell.cart_cods[ind], new_NVcell) < radius_thres_2 and \
                    dis_nv_2(new_NVcell.cart_cods[jnd], new_NVcell) < radius_thres_2:

                    n_ind, error = f_216nv_ind(new_NVcell, shape_new_cell, ind)
                    new_NV_fc_mat[ind,jnd] = NV216_fc_mat[n_ind,jnd]
                # Use the bulk fc matrix instead
                else:
                    set_ind = dec_set(new_NVcell.cry_cods[ind], shape_new_cell[0])
                    if set_ind == 0:
                        Aux_Pair_Dis = A_Reduced_Pair_Matrix - ddd
                        Aux_Pair_Norm = np.linalg.norm(Aux_Pair_Dis, axis=1)
                        min_Dis = np.amin(Aux_Pair_Norm)
                        if min_Dis < 0.5:
                            new_NV_fc_mat[ind,jnd] = A_Dia512_unique_fc_mat[np.argmin(Aux_Pair_Norm)]
                    elif set_ind == 1:
                        Aux_Pair_Dis = B_Reduced_Pair_Matrix - ddd
                        Aux_Pair_Norm = np.linalg.norm(Aux_Pair_Dis, axis=1)
                        min_Dis = np.amin(Aux_Pair_Norm)
                        if min_Dis < 0.5:
                            new_NV_fc_mat[ind,jnd] = B_Dia512_unique_fc_mat[np.argmin(Aux_Pair_Norm)]

    # Part-3:
    for ind in range(NV216cell.noa, new_NVcell.noa):
        for jnd in range(NV216cell.noa, new_NVcell.noa):
            ddd = dis_pair(new_NVcell, ind, jnd)
            # Only pairs with distance smaller than radius_thred_1 are set to non-zero
            if np.linalg.norm(ddd) < radius_thres_3:
                if dis_nv_2(new_NVcell.cart_cods[ind], new_NVcell) < radius_thres_2 and \
                    dis_nv_2(new_NVcell.cart_cods[jnd], new_NVcell) < radius_thres_2:

                    n_ind, error = f_216nv_ind(new_NVcell, shape_new_cell, ind)
                    n_jnd, error = f_216nv_ind(new_NVcell, shape_new_cell, jnd)
                    new_NV_fc_mat[ind,jnd] = NV216_fc_mat[n_ind,n_jnd]
                # Use the bulk fc matrix instead
                else:
                    set_ind = dec_set(new_NVcell.cry_cods[ind], shape_new_cell[0])
                    if set_ind == 0:
                        Aux_Pair_Dis = A_Reduced_Pair_Matrix - ddd
                        Aux_Pair_Norm = np.linalg.norm(Aux_Pair_Dis, axis=1)
                        min_Dis = np.amin(Aux_Pair_Norm)
                        if min_Dis < 0.5:
                            new_NV_fc_mat[ind,jnd] = A_Dia512_unique_fc_mat[np.argmin(Aux_Pair_Norm)]
                    elif set_ind == 1:
                        Aux_Pair_Dis = B_Reduced_Pair_Matrix - ddd
                        Aux_Pair_Norm = np.linalg.norm(Aux_Pair_Dis, axis=1)
                        min_Dis = np.amin(Aux_Pair_Norm)
                        if min_Dis < 0.5:
                            new_NV_fc_mat[ind,jnd] = B_Dia512_unique_fc_mat[np.argmin(Aux_Pair_Norm)]

    ####################
    # Modify FC matrix #
    ####################

    if correct_diag == True:
        for ind in range(new_NVcell.noa):
            for ic in range(3):
                new_NV_fc_mat[ind,ind,ic,ic] = 0.0
                aux = - np.sum(new_NV_fc_mat[ind,:,ic,ic])
                new_NV_fc_mat[ind,ind,ic,ic] = aux

    # Reshape the FC matrix
    new_NV_fc_mat = np.swapaxes(new_NV_fc_mat, 1, 2)
    new_NV_fc_mat = np.reshape(new_NV_fc_mat, (new_NVcell.noa*3, new_NVcell.noa*3))

    print('FC matrix is symmetric: ', np.allclose(new_NV_fc_mat, new_NV_fc_mat.T, rtol=1e-5, atol=1e-8))
    print(np.unravel_index(np.argmax(abs(new_NV_fc_mat - new_NV_fc_mat.T)), new_NV_fc_mat.shape))
    print((new_NV_fc_mat - new_NV_fc_mat.T)[np.unravel_index(np.argmax(abs(new_NV_fc_mat - new_NV_fc_mat.T)), new_NV_fc_mat.shape)])

    # Divide the atomic mass
    new_mass_vec = np.zeros((new_NVcell.noa, 3))
    for ind in range(new_NVcell.noa):
        new_mass_vec[ind,:] = 1/np.sqrt(new_NVcell.atoms[ind].mass)
    new_mass_vec = np.reshape(new_mass_vec, (new_NVcell.noa, 3))
    new_mass_mat = np.outer(new_mass_vec, new_mass_vec)
    new_NV_fc_mat = new_NV_fc_mat * new_mass_mat

    #####################
    # Solve for phonons #
    #####################

    w, v = eigh(new_NV_fc_mat)
    ordering = np.argsort(w)
    w = w[ordering]
    v = v[:,ordering]

    sign = np.where(w>=0, 1, -1)    

    freq = (1/2/np.pi) * (abs(w) *
           constants.physical_constants["Rydberg constant times hc in J"][0] /
           constants.physical_constants["Bohr radius"][0]**2 /
           constants.physical_constants["unified atomic mass unit"][0] )**0.5 * 1e-12

    freq = freq * sign

    # Write to the file
    for ind in range(10):
        print('Index %6d Freq % .8f THz  % .8f cm-1  % .8f meV'%(
              ind+1, freq[ind], freq[ind]*33.35641, freq[ind]*4.13567 ))

    #################
    # Write to file #
    #################

    modes = np.array([v])
    freqs = np.array([freq])

    fname = 'mesh.hdf5'
    with h5py.File(fname, 'w') as w:
        w.create_dataset('eigenvector', data=modes)
        w.create_dataset('frequency', data=freqs)
