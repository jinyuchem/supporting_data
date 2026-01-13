#!/usr/bin/env python3
import numpy as np
import sys
import json
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns

#############
# Functions #
#############

class DJT_class():
    """ All parameters """
    def __init__( self ):
        self.prefix   = ''
        self.num_eign = 0
        self.num_ph   = 0
        self.limit    = self.num_ph 

        self.Le       = 0
        self.E_ph     = 0
        self.Ft       = 0
        self.Gt       = 0
        self.F        = 0
        self.G        = 0

def djt_solve_matrix(M):
    """ Solve the eigenvalue problem """
    Eigenvalues, Eigenvectors = eigh( M, eigvals=( 0, djt.num_eign ) )
    return Eigenvalues, Eigenvectors

def djt_write_para(djt, w):
    nline = ('num_eigen  ', str(djt.num_eign), '\n')
    w.writelines(nline)
    nline = ('num_ph     ', str(djt.num_ph), '\n')
    w.writelines(nline)
    nline = ('Le         ', str(djt.Le), '\n')
    w.writelines(nline)
    nline = ('E_ph       ', str(djt.E_ph), '\n')
    w.writelines(nline)
    nline = ('Ft         ', str(djt.Ft), '\n')
    w.writelines(nline)
    nline = ('Gt         ', str(djt.Gt), '\n')
    w.writelines(nline)
    nline = ('F          ', str(djt.F), '\n')
    w.writelines(nline)
    nline = ('G          ', str(djt.G), '\n')
    w.writelines(nline)

def djt_new_write_eva_evc(EVA, EVC, w):
    """ Write eigenvalues and eigenvectors """
    size = djt.num_ph * djt.num_ph
    for i in range(djt.num_eign):
        nline = ('The %2d-th energy is %3.12f '%(i + 1, EVA[i]), '\n')
        w.writelines(nline)
        nline = ('=' * 75, '\n')
        w.writelines(nline)

        nline = ('The norm is %3.12f '%(
                 sum(np.abs(EVC[0:size, i])**2) + sum(np.abs(EVC[size:2*size, i])**2)
                 + sum(np.abs(EVC[2*size:, i])**2)), '\n')
        w.writelines(nline)

        nline = ('The amplitude on xx  is %3.12f '%(sum(np.abs(EVC[0:size, i])**2)), '\n')
        w.writelines(nline)
        nline = ('The amplitude on xy  is %3.12f '%(sum(np.abs(EVC[size:2*size, i])**2)), '\n')
        w.writelines(nline)
        nline = ('The amplitude on yy  is %3.12f '%(sum(np.abs(EVC[2*size:3*size, i])**2)), '\n')
        w.writelines(nline)

        nline = ('=' * 75, '\n')
        w.writelines(nline)
        nline = ('This state is mainly composed of transition between', '\n')
        w.writelines(nline)
        lrange = (-abs(EVC[:,i])).argsort()[:6]
        for mmm in lrange:
            if mmm < size:
                nline = ('  xx      m index %4d    n index %4d    amplitude % .8f + % .8f * j '%(
                         mmm//djt.num_ph, mmm%djt.num_ph, np.real( EVC[mmm][i]),
                         np.imag(EVC[mmm][i])), '\n')
            elif mmm >= size and mmm < 2*size:
                nline = ('  xy      m index %4d    n index %4d    amplitude % .8f + % .8f * j '%(
                         (mmm - size)//djt.num_ph, (mmm - size)%djt.num_ph,
                         np.real(EVC[mmm][i]), np.imag(EVC[mmm][i])), '\n')
            elif mmm >= 2*size:
                nline = ('  yy      m index %4d    n index %4d    amplitude % .8f + % .8f * j '%(
                         (mmm - 2*size)//djt.num_ph, (mmm - 2*size)%djt.num_ph,
                         np.real(EVC[mmm][i]), np.imag(EVC[mmm][i])), '\n')
            w.writelines(nline)
        nline = ('=' * 75, '\n')
        w.writelines(nline)
        nline = ('All eigenvectors are listed below', '\n')
        w.writelines(nline)
        for m in range(djt.num_ph):
            for n in range(djt.num_ph):
                nline = ('  xx      m index %4d    n index %4d    amplitude % .8f + % .8f * j '%(
                         m, n, np.real(EVC[m * djt.num_ph + n][i]),
                         np.imag(EVC[m * djt.num_ph + n][i])), '\n')
                w.writelines(nline)
        for m in range(djt.num_ph):
            for n in range(djt.num_ph):
                nline = ('  xy      m index %4d    n index %4d    amplitude % .8f + % .8f * j '%(
                          m, n, np.real(EVC[m * djt.num_ph + n + size][i]),
                          np.imag(EVC[m * djt.num_ph + n + size][i])), '\n')
                w.writelines(nline)
        for m in range(djt.num_ph):
            for n in range(djt.num_ph):
                nline = ('  yy      m index %4d    n index %4d    amplitude % .8f + % .8f * j '%(
                          m, n, np.real(EVC[m * djt.num_ph + n + 2*size][i]), 
                          np.imag(EVC[m * djt.num_ph + n + 2*size][i])), '\n')
                w.writelines(nline)
        nline = ( '=' * 75, '\n' )
        w.writelines( nline )
        nline = ( '\n' )
        w.writelines( nline )
        nline = ( '\n' )
        w.writelines( nline )
    return

""" Write eigenvalues and eigenvectors """
def djt_write_eva_evc( EVA, EVC, w ):
    size = djt.num_ph * djt.num_ph
    for i in range( djt.num_eign ):
        nline = ( 'The %2d-th energy is %3.12f '%( i + 1, EVA[i] ), '\n')
        w.writelines( nline )
        nline = ( '=' * 75, '\n' )
        w.writelines( nline )

        nline = ( 'The p-factor is %3.12f '%( \
                  sum( np.abs( EVC[0:size, i] )**2 ) - sum( np.abs( EVC[size:, i] )**2 ) ), '\n')
        w.writelines( nline )

        nline = ( 'The norm is %3.12f '%( \
                  sum( np.abs( EVC[0:size, i] )**2 ) + sum( np.abs( EVC[size:, i] )**2 ) ), '\n')
        w.writelines( nline )

        nline = ( '=' * 75, '\n' )
        w.writelines( nline )
        nline = ( 'This state is mainly composed of transition between', '\n' )
        w.writelines( nline )
        lrange = ( -abs( EVC[:,i] ) ).argsort()[:6]
        for mmm in lrange:
            if mmm < size:
                nline = ( '  plus    m index %4d    n index %4d    amplitude % .8f + % .8f * j '%( \
                          mmm//djt.num_ph, mmm%djt.num_ph, np.real( EVC[mmm][i] ), \
                          np.imag( EVC[mmm][i] ) ), '\n' )
            else:
                nline = ( '  minus   m index %4d    n index %4d    amplitude % .8f + % .8f * j '%( \
                          ( mmm - size )//djt.num_ph, ( mmm - size )%djt.num_ph, \
                            np.real( EVC[mmm][i] ), np.imag( EVC[mmm][i] ) ), '\n' )
            w.writelines( nline )
        nline = ( '=' * 75, '\n' )
        w.writelines( nline )
        nline = ( 'All eigenvectors are listed below', '\n' )
        w.writelines( nline )
        for m in range( djt.num_ph ):
            for n in range( djt.num_ph ):
                nline = ( '  plus    m index %4d    n index %4d    amplitude % .8f + % .8f * j '%( \
                          m, n, np.real( EVC[ m * djt.num_ph + n ][i] ), \
                          np.imag( EVC[ m * djt.num_ph + n ][i] ) ), '\n' )
                w.writelines( nline )
        for m in range( djt.num_ph ):
            for n in range( djt.num_ph ):
                nline = ( '  minus   m index %4d    n index %4d    amplitude % .8f + % .8f * j '%( \
                          m, n, np.real( EVC[ m * djt.num_ph + n + size ][i] ), \
                          np.imag( EVC[ m * djt.num_ph + n + size ][i] ) ), '\n' )
                w.writelines( nline )
        nline = ( '=' * 75, '\n' )
        w.writelines( nline )
        nline = ( '\n' )
        w.writelines( nline )
        nline = ( '\n' )
        w.writelines( nline )
    return



def djt_new_basis_write_eva_evc(EVA, EVC, w):
    """ Write eigenvalues and eigenvectors """
    size = (djt.num_ph + 1) * (djt.num_ph + 2) / 2
    size = int(size)
    for i in range(djt.num_eign):
        nline = ('The %2d-th energy is %3.12f '%(i + 1, EVA[i]), '\n')
        w.writelines(nline)
        nline = ('=' * 75, '\n')
        w.writelines(nline)

        nline = ('The norm is %3.12f '%(
                 sum(np.abs(EVC[0:size, i])**2) + sum(np.abs(EVC[size:2*size, i])**2)
                 + sum(np.abs(EVC[2*size:, i])**2)), '\n')
        w.writelines(nline)

        nline = ('The amplitude on 1A1  is %3.12f '%(sum(np.abs(EVC[0:size, i])**2)), '\n')
        w.writelines(nline)
        nline = ('The amplitude on 1Ex  is %3.12f '%(sum(np.abs(EVC[size:2*size, i])**2)), '\n')
        w.writelines(nline)
        nline = ('The amplitude on 1Ey  is %3.12f '%(sum(np.abs(EVC[2*size:3*size, i])**2)), '\n')
        w.writelines(nline)

        nline = ('=' * 75, '\n')
        w.writelines(nline)
        nline = ('This state is mainly composed of transition between', '\n')
        w.writelines(nline)
        lrange = (-abs(EVC[:,i])).argsort()[:6]
        for mmm in lrange:
            if mmm < size:
                ind_m, ind_n = from_index_to_ij(mmm)
                nline = ('  1A1      m index %4d    n index %4d    amplitude % .8f + % .8f * j '%(
                         ind_m, ind_n, np.real(EVC[mmm][i]), np.imag(EVC[mmm][i])), '\n')
            elif mmm >= size and mmm < 2*size:
                ind_m, ind_n = from_index_to_ij(mmm - size)
                nline = ('  1Ex      m index %4d    n index %4d    amplitude % .8f + % .8f * j '%(
                         ind_m, ind_n, np.real(EVC[mmm][i]), np.imag(EVC[mmm][i])), '\n')
            elif mmm >= 2*size:
                ind_m, ind_n = from_index_to_ij(mmm - 2*size)
                nline = ('  1Ey      m index %4d    n index %4d    amplitude % .8f + % .8f * j '%(
                         ind_m, ind_n, np.real(EVC[mmm][i]), np.imag(EVC[mmm][i])), '\n')
            w.writelines(nline)
        nline = ('=' * 75, '\n')
        w.writelines(nline)
        nline = ('All eigenvectors are listed below', '\n')
        w.writelines(nline)
        for mmm in range(size):
            ind_m, ind_n = from_index_to_ij(mmm)
            nline = ('  1A1      m index %4d    n index %4d    amplitude % .8f + % .8f * j '%(
                     ind_m, ind_n, np.real(EVC[mmm][i]), np.imag(EVC[mmm][i])), '\n')
            w.writelines(nline)
        for mmm in range(size, 2*size):
            ind_m, ind_n = from_index_to_ij(mmm - size)
            nline = ('  1Ex      m index %4d    n index %4d    amplitude % .8f + % .8f * j '%(
                     ind_m, ind_n, np.real(EVC[mmm][i]), np.imag(EVC[mmm][i])), '\n')
            w.writelines(nline)
        for mmm in range(2*size, 3*size):
            ind_m, ind_n = from_index_to_ij(mmm - 2*size)
            nline = ('  1Ey      m index %4d    n index %4d    amplitude % .8f + % .8f * j '%(
                     ind_m, ind_n, np.real(EVC[mmm][i]), np.imag(EVC[mmm][i])), '\n')
            w.writelines(nline)
        nline = ( '=' * 75, '\n' )
        w.writelines( nline )
        nline = ( '\n' )
        w.writelines( nline )
        nline = ( '\n' )
        w.writelines( nline )
    return



# important
def from_index_to_ij(ind):
    """
    for a given combinatory index, return the corresponding index for phonons
    """
    # generate a list
    boundary = np.zeros(djt.num_ph + 2)
    for i in range(djt.num_ph + 2):
        boundary[i] = i * (djt.num_ph + 1) - 0.5 * i * (i - 1)
    
    for i in range(djt.num_ph + 1):
        if ind >= boundary[i] and ind < boundary[i+1]:
            ind_m = i
            ind_n = ind - boundary[i]

    return ind_m, ind_n



def build_ph_diagonal(djt):
    size = (djt.num_ph + 1) * (djt.num_ph + 2) / 2
    size = int(size)
    diag = np.zeros(size)
    
    tmp = 0
    for i in range(djt.num_ph + 1):
        for j in range(djt.num_ph + 1 - i):
            diag[tmp + j] = i + j + 1
        tmp = tmp + djt.num_ph + 1 - i

    mat = np.diag(diag)
    return mat


def build_ph_x(djt):
    size = (djt.num_ph + 1) * (djt.num_ph + 2) / 2
    size = int(size)
    mat = np.zeros((size, size))

    tmp = [0,0]
    for i in range(djt.num_ph + 1):
        for j in range(djt.num_ph + 1 - i):
            for k in range(djt.num_ph + 1):
                for l in range(djt.num_ph + 1 - k):
                    if i+1 == k and j == l:
                        mat[tmp[0] + j, tmp[1] + l] = np.sqrt(i+1) / np.sqrt(2)
                    elif i == k + 1 and j == l:
                        mat[tmp[0] + j, tmp[1] + l] = np.sqrt(i) / np.sqrt(2)
                tmp[1] = tmp[1] + djt.num_ph + 1 - k
            tmp[1] = 0
        tmp[0] = tmp[0] + djt.num_ph + 1 - i

    return mat


def build_ph_y(djt):
    size = (djt.num_ph + 1) * (djt.num_ph + 2) / 2
    size = int(size)
    mat = np.zeros((size, size))
    
    tmp = [0,0]
    for i in range(djt.num_ph + 1):
        for j in range(djt.num_ph + 1 - i):
            for k in range(djt.num_ph + 1):
                for l in range(djt.num_ph + 1 - k):
                    if i == k and j+1 == l:
                        mat[tmp[0] + j, tmp[1] + l] = np.sqrt(j+1) / np.sqrt(2)
                    elif i == k and j == l+1:
                        mat[tmp[0] + j, tmp[1] + l] = np.sqrt(j) / np.sqrt(2)
                tmp[1] = tmp[1] + djt.num_ph + 1 - k
            tmp[1] = 0
        tmp[0] = tmp[0] + djt.num_ph + 1 - i

    return mat


def build_ph_x2(djt):
    size = (djt.num_ph + 1) * (djt.num_ph + 2) / 2
    size = int(size)
    mat = np.zeros((size, size))

    tmp = [0,0]
    for i in range(djt.num_ph + 1):
        for j in range(djt.num_ph + 1 - i):
            for k in range(djt.num_ph + 1):
                for l in range(djt.num_ph + 1 - k):
                    if i == k and j == l:
                        mat[tmp[0] + j, tmp[1] + l] = (2 * i + 1) / 2
                    elif i+2 == k and j == l:
                        mat[tmp[0] + j, tmp[1] + l] = np.sqrt(i + 1) * np.sqrt(i + 2) / 2
                    elif i == k+2 and j == l:
                        mat[tmp[0] + j, tmp[1] + l] = np.sqrt(i - 1) * np.sqrt(i) / 2
                tmp[1] = tmp[1] + djt.num_ph + 1 - k
            tmp[1] = 0
        tmp[0] = tmp[0] + djt.num_ph + 1 - i

    return mat


def build_ph_y2(djt):
    size = (djt.num_ph + 1) * (djt.num_ph + 2) / 2
    size = int(size)
    mat = np.zeros((size, size))

    tmp = [0,0]
    for i in range(djt.num_ph + 1):
        for j in range(djt.num_ph + 1 - i):
            for k in range(djt.num_ph + 1):
                for l in range(djt.num_ph + 1 - k):
                    if i == k and j == l:
                        mat[tmp[0] + j, tmp[1] + l] = (2 * j + 1) / 2
                    elif i == k and j+2 == l:
                        mat[tmp[0] + j, tmp[1] + l] = np.sqrt(j + 1) * np.sqrt(j + 2) / 2
                    elif i == k and j == l+2:
                        mat[tmp[0] + j, tmp[1] + l] = np.sqrt(j - 1) * np.sqrt(j) / 2
                tmp[1] = tmp[1] + djt.num_ph + 1 - k
            tmp[1] = 0
        tmp[0] = tmp[0] + djt.num_ph + 1 - i

    return mat


def build_ph_2xy(djt):
    size = (djt.num_ph + 1) * (djt.num_ph + 2) / 2
    size = int(size)
    mat = np.zeros((size, size))

    tmp = [0,0]
    for i in range(djt.num_ph + 1):
        for j in range(djt.num_ph + 1 - i):
            for k in range(djt.num_ph + 1):
                for l in range(djt.num_ph + 1 - k):
                    if i+1 == k and j+1 == l:
                        mat[tmp[0] + j, tmp[1] + l] = np.sqrt(i+1) * np.sqrt(j+1) / 2
                    elif i+1 == k and j == l+1:
                        mat[tmp[0] + j, tmp[1] + l] = np.sqrt(i+1) * np.sqrt(j) / 2
                    elif i == k+1 and j+1 == l:
                        mat[tmp[0] + j, tmp[1] + l] = np.sqrt(i) * np.sqrt(j+1) / 2
                    elif i == k+1 and j == l+1:
                        mat[tmp[0] + j, tmp[1] + l] = np.sqrt(i) * np.sqrt(j) / 2
                tmp[1] = tmp[1] + djt.num_ph + 1 - k
            tmp[1] = 0
        tmp[0] = tmp[0] + djt.num_ph + 1 - i

    return 2 * mat


def build_pjt_djt_mat(djt):

    mat_diag = build_ph_diagonal(djt)
    mat_x = build_ph_x(djt)
    mat_y = build_ph_y(djt)
    mat_x2 = build_ph_x2(djt)
    mat_y2 = build_ph_y2(djt)
    mat_2xy = build_ph_2xy(djt)

    ct = False
    if ct == True:
        mat_diag = constraint_mat(djt, mat_diag)
        mat_x    = constraint_mat(djt, mat_x)
        mat_y    = constraint_mat(djt, mat_y)
        mat_x2   = constraint_mat(djt, mat_x2)
        mat_y2   = constraint_mat(djt, mat_y2)
        mat_2xy  = constraint_mat(djt, mat_2xy)

    mat_p1p = np.array([[1,0,1], [0,0,0], [1,0,1]]) / 2
    mat_p2p = np.array([[1,0,0], [0,1,0], [0,0,1]])
    mat_p3p = np.array([[1,0,0], [0,0,0], [0,0,-1]])
    mat_p4p = np.array([[0,1,0], [1,0,1], [0,1,0]]) / np.sqrt(2)
    mat_p5p = np.array([[1,0,-1], [0,-2,0], [-1,0,1]]) / 2
    mat_p6p = np.array([[0,1,0], [1,0,-1], [0,-1,0]]) / np.sqrt(2)

    # splitting of 1A1 and 1E
    size = (djt.num_ph + 1) * (djt.num_ph + 2) / 2
    size = int(size)
    mat_p1 = np.kron(mat_p1p, djt.Le * np.diag(np.ones(size)))

    # harmonic oscillator
    mat_p2 = np.kron(mat_p2p, mat_diag * djt.E_ph)

    # linear PJT
    mat_p3 = np.kron(mat_p3p, mat_x * djt.Ft)
    mat_p4 = np.kron(mat_p4p, mat_y * djt.Ft)

    # linear DJT
    mat_p8 = np.kron(mat_p5p, mat_x * djt.F)
    mat_p9 = np.kron(mat_p6p, mat_y * (-1) * djt.F)

    mat = mat_p1 + mat_p2 + mat_p3 + mat_p4 + mat_p8 + mat_p9

    return mat


def build_obs_pjt_djt_mat(djt):

    mat_diag = build_ph_diagonal(djt)
    mat_x = build_ph_x(djt)
    mat_y = build_ph_y(djt)
    mat_x2 = build_ph_x2(djt)
    mat_y2 = build_ph_y2(djt)
    mat_2xy = build_ph_2xy(djt)

    ct = False
    if ct == True:
        mat_diag = constraint_mat(djt, mat_diag)
        mat_x    = constraint_mat(djt, mat_x)
        mat_y    = constraint_mat(djt, mat_y)
        mat_x2   = constraint_mat(djt, mat_x2)
        mat_y2   = constraint_mat(djt, mat_y2)
        mat_2xy  = constraint_mat(djt, mat_2xy)

    mat_p1p = np.array([[1,0,0], [0,0,0], [0,0,0]])
    mat_p2p = np.array([[1,0,0], [0,1,0], [0,0,1]])
    mat_p3p = np.array([[0,1,0], [1,0,0], [0,0,0]])
    mat_p4p = np.array([[0,0,1], [0,0,0], [1,0,0]])
    mat_p5p = np.array([[0,0,0], [0,1,0], [0,0,-1]])
    mat_p6p = np.array([[0,0,0], [0,0,1], [0,1,0]])

    # splitting of 1A1 and 1E
    size = (djt.num_ph + 1) * (djt.num_ph + 2) / 2
    size = int(size)
    mat_p1 = np.kron(mat_p1p, djt.Le * np.diag(np.ones(size)))

    # harmonic oscillator
    mat_p2 = np.kron(mat_p2p, mat_diag * djt.E_ph)

    # linear PJT
    mat_p3 = np.kron(mat_p3p, mat_x * djt.Ft)
    mat_p4 = np.kron(mat_p4p, mat_y * djt.Ft)

    # linear DJT
    mat_p8 = np.kron(mat_p5p, mat_x * djt.F)
    mat_p9 = np.kron(mat_p6p, mat_y * (-1) * djt.F)

    mat = mat_p1 + mat_p2 + mat_p3 + mat_p4 + mat_p8 + mat_p9

    return mat


def build_djt_mat(djt):

    mat_diag = build_ph_diagonal(djt)
    mat_x = build_ph_x(djt)
    mat_y = build_ph_y(djt)
    mat_x2 = build_ph_x2(djt)
    mat_y2 = build_ph_y2(djt)
    mat_2xy = build_ph_2xy(djt)

    mat_p1p = np.array([[1,0], [0,1]])
    mat_p2p = np.array([[1,0], [0,-1]])
    mat_p3p = np.array([[0,1], [1,0]])

    # harmonic oscillator
    mat_p1 = np.kron(mat_p1p, mat_diag * djt.E_ph)

    # linear DJT
    mat_p2 = np.kron(mat_p2p, mat_x * djt.F)
    mat_p3 = np.kron(mat_p3p, mat_y * (-1) * djt.F)

    # quadratic DJT
    mat_p4 = np.kron(mat_p2p, mat_x2 * djt.G)
    mat_p5 = np.kron(mat_p2p, mat_y2 * (-1) * djt.G)
    mat_p6 = np.kron(mat_p3p, mat_2xy * djt.G)

    mat = mat_p1 + mat_p2 + mat_p3 + mat_p4 + mat_p5 + mat_p6

    return mat

########
# Main #
########

# Initialize
djt = DJT_class()

djt.prefix   = 'non-adiabatic'
djt.num_eign = 1487
djt.num_ph   = 30

# Our para
djt.Le       = 821
djt.E_ph = 62.9506828
djt.Ft = 133.22436286
djt.F = 62.37653058

# Build DJT matrix
DJT_M = build_obs_pjt_djt_mat(djt)

# Compute the eigenvalues and eigenvectors
EVA, EVC = djt_solve_matrix(DJT_M)

np.set_printoptions(precision=5, suppress=True)
print(EVA[:])

# Output
fname = djt.prefix + '-num-ph-' + str(djt.num_ph) + '.out'

w = open(fname, 'w')
djt_write_para(djt, w)
djt_new_basis_write_eva_evc(EVA, EVC, w)
w.close()

#########
# Debug #
#########

debug = False

if debug == True:

    eig_1 = EVC[:,0]
    eig_2 = EVC[:,1]

    theta = np.linspace(0, 2*np.pi, 121)
    size = djt.num_ph * djt.num_ph

    print('=' * 60)
    print( np.dot(eig_1.conjugate(), np.dot(DJT_M, eig_1)) )
    print( np.dot(eig_2.conjugate(), np.dot(DJT_M, eig_2)) )

    size = (djt.num_ph + 1) * (djt.num_ph + 2) / 2
    size = int(size)

    for i in range(theta.shape[0]):
        new_evc = np.cos(theta[i]) * eig_1 + np.sin(theta[i]) * eig_2
        ex_norm = np.sum(new_evc[size:2*size]**2)
        ey_norm = np.sum(new_evc[2*size:]**2)
        ene = np.dot(new_evc.conjugate(), np.dot(DJT_M, new_evc))
        print('angle is     % .3f pi  |   ex norm is    % .5f  |   ey norm is    % .5f  |   energy is    %.4f meV'%(theta[i]/np.pi, ex_norm, ey_norm, ene.real))
