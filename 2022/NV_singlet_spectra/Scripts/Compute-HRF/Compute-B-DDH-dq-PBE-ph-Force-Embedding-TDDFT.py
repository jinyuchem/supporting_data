#!/usr/bin/env python3
import numpy as np
from copy import deepcopy
import sys
from scipy.linalg import eigh, eig
from scipy import constants
import h5py
from ase.io import read, write
import subprocess

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

# function to load dft forces
def load_dft_forces(fname, natoms):
    force_g = subprocess.getstatusoutput("grep '%s' -A %d %s | tail -%d | awk '{print $7, $8, $9}'"%(
              'Forces', natoms+1, fname, natoms))[1]
    force_g = np.reshape(np.matrix(force_g), (natoms,3))
    return force_g

# function to load tddft forces
def load_tddft_forces(fname, natoms):
    force_e = subprocess.getstatusoutput("grep '%s' -A %d %s | tail -%d | awk '{print $7, $8, $9}'"%(
              'TDDFT Force Corrected', natoms+1, fname, natoms))[1]
    force_e = np.reshape(np.matrix(force_e), (natoms,3))
    return force_e

def load_phonon(fname, n_atoms):
    freq = np.zeros(n_atoms*3)
    modes = np.zeros((n_atoms*3, n_atoms, 3))
    with open(fname, 'r') as f:
        for i in range(n_atoms*3):
            line = f.readline()
            freq[i] = line.split()[5]
            for j in range(n_atoms):
                line = f.readline()
                for k in range(3):
                    modes[i,j,k] = line.split()[k]
        f.close()

    freq = freq * 2 * np.pi * constants.c * 1e2

    return freq, modes

def load_phonon_hdf5(fname):
    with h5py.File(fname, "r") as f:
        # phonon eigenvectors
        a_group_key = list(f.keys())[0]
        r_modes = list(f[a_group_key])[0]
        # phonon frequencies
        b_group_key = list(f.keys())[1]
        r_freqs = list(f[b_group_key])[0]

    # THz to s^{-1}; angular frequencies
    freqs = r_freqs * 1e12 * 2 * np.pi
    # eigenvectors should be real
    modes = np.reshape(np.swapaxes(r_modes, 0, 1),
                      (freqs.shape[0], int(freqs.shape[0]/3), 3)).real

    return freqs, modes

if __name__ == "__main__":

    ####################
    # Input Parameters #
    ####################

    # NV in diamond in a new cell
    # Shape of the new cell: only support cubic cell at this moment
    shape_new_cell = [12,12,12]
    N_LP = np.array([
        [3.55, 0.0, 0.0],
        [0.0, 3.55, 0.0],
        [0.0, 0.0, 3.55]
    ])
    N_LP[0] = N_LP[0] * shape_new_cell[0]
    N_LP[1] = N_LP[1] * shape_new_cell[1]
    N_LP[2] = N_LP[2] * shape_new_cell[2]

    f_name = 'new-nv-cell'

    f_phonon = '0614-13824-1A1-phonon-cph-5-mesh.hdf5'

    f_force_gs = '1A1-F-at-1E-G-DFT.dat'
    f_force_es = '1A1-F-at-1E-G-TDDFT.dat'

    # Threshold for distance to NV: use 5 \AA in ref.
    radius_thres_2 = 5

    #######################
    # Original supercells #
    #######################
    
    # NV in diamond in a 216 atoms cell
    LP = np.array([
        [10.65, 0.0, 0.0],
        [0.0, 10.65, 0.0],
        [0.0, 0.0, 10.65]
        ])
    nol = 215
    NV216cell = CELL(LP)
    NV216cell.readfqefile('nv-216-cod.in', nol, 'cry')
    NV216cell.atoms[-1].set_mass(14.0067)

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

    ###############
    # Load forces #
    ###############

    # Load forces
    es_forces = load_tddft_forces(f_force_es, NV216cell.noa)
    es_forces = es_forces * 13.6056980659 / 0.529177249

    print(es_forces[-1])

    out_gs = read(f_force_gs, index='-1', format='espresso-out')

    raw_forces = out_gs.get_forces()
    print(raw_forces[-1])
    raw_forces = np.array(raw_forces, dtype=np.float64) + es_forces
    print(raw_forces[-1])
    raw_forces = raw_forces * constants.eV / 1e-10

    #######################
    # Build new forces #
    #######################

    # Build the force constant matrix for the new nv cell
    act_forces = np.zeros((new_NVcell.noa, 3))

    # Part-0: The part copied from the 216 nv cell
    for ind in range(NV216cell.noa):
        # If the pairs are closer to the nv, then use the defective fc matrix
        if dis_nv_2(new_NVcell.cart_cods[ind], new_NVcell) < radius_thres_2:
            act_forces[ind] = raw_forces[ind]

    # Compute the difference between the new and the old fc mat
    force_diff = act_forces[0:NV216cell.noa,:] - raw_forces[:,:]
    print('Norm of the force diff is % .5f'%(np.linalg.norm(force_diff)))

    fname = 'rc2-' + str(radius_thres_2) + '-force-diff.dat'
    with open(fname, 'w') as w:
        for ind in range(NV216cell.noa):
            nline = ('%5d'%(ind+1), '\n')
            w.writelines(nline)
            nline = ('% .14f   % .14f   % .14f'%(force_diff[ind,0],
                    force_diff[ind,1], force_diff[ind,2]), '\n')
            w.writelines(nline)

    # Part-1:
    for ind in range(NV216cell.noa, new_NVcell.noa):
        # If the pairs are closer to the nv, then use the defective fc matrix
        if dis_nv_2(new_NVcell.cart_cods[ind], new_NVcell) < radius_thres_2:
            n_ind, error = f_216nv_ind(new_NVcell, shape_new_cell, ind)
            act_forces[ind] = raw_forces[n_ind]

    fname = 'rc2-' + str(radius_thres_2) + '-force.dat'
    with open(fname, 'w') as w:
        for ind in range(new_NVcell.noa):
            nline = ('%5s   % .14f   % .14f   % .14f'%(new_NVcell.atoms[ind].element,
                act_forces[ind,0], act_forces[ind,1], act_forces[ind,2]), '\n')
            w.writelines(nline)

    ################
    # load phonone #
    ################

    freq, modes = load_phonon_hdf5(f_phonon)

    #############
    # Compute B #
    #############

    # Atomic mass
    all_mass = np.ones(new_NVcell.noa)
    for i in range(new_NVcell.noa):
        all_mass[i] = new_NVcell.atoms[i].mass

    all_mass = all_mass * constants.physical_constants['atomic mass constant'][0]

    mass_forces = act_forces * np.power(all_mass, -0.5)[:, np.newaxis]
    mass_forces = np.reshape(mass_forces, new_NVcell.noa*3)

    freq_modes = np.reshape(modes, (new_NVcell.noa*3, new_NVcell.noa*3))
    freq_modes = freq_modes[3:,:] * np.power(freq[3:], -1.5)[:, np.newaxis]

    B = -np.sqrt(1/(2*constants.hbar)) * np.dot(freq_modes, mass_forces)

    B[np.isnan(B)] = 0.0

    #
    ratio = 1.039982186
    B = B * np.power(ratio, -2)
    #

    o_fname = 'All-B-PBE-ph-1A1-1E-' + str(new_NVcell.noa) + '-rc2-' + str(radius_thres_2) + '-cph-5.dat'
    with open(o_fname, 'w') as w:
        for i in range(new_NVcell.noa*3 - 3):
            nline = ('Mode      % 6d    B    % 10.8e   Freq (cm^{-1}) % 13.8f'%(
                     i+4, B[i], freq[3+i]/(constants.c * 1e2 * 2 * np.pi)), '\n')
            w.writelines(nline)
        w.close()

    # Compute total HRF
    HRF = sum(B**2)
    print('=' * 45)
    print('Total HRF is % 10.8e'%HRF)
