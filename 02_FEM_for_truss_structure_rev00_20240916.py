#
# TITLE: FEM for truss structure
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: https://www.youtube.com/watch?v=tv1TlAebvm0&list=PLnT2pATp7adWUaMh0A8jfoN_OHGAmt_m4&index=2
#
#

import numpy as np
import math

# node list
NL = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

# element list
EL = np.array([[1, 2], [2, 3], [3,1]])

# boundary conditions for nodes
# +1: neumann boundary condition (free to move)
# -1: dirichlet boundary condition (fixed)
DorN = np.array([[-1, -1], [1, -1], [1, 1]])

# force for nodes
Fu = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, -20.0]])

# displacement for nodes
Uu = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

# Young's modulus
E = 10**6

# cross sectional area
A = 0.01

# problem dimension, # of columns = 2
PD = np.size(NL, 1)

# # of nodes = # of rows
NoN = np.size(NL, 0)

# extended node list
ENL = np.zeros([NoN, 6*PD])

ENL[:,0:PD] = NL[:,:]
ENL[:,PD:2*PD] = DorN[:,:]

#
def assign_BCs(NL, ENL):
    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)

    # initial value
    DOFs = 0
    DOCs = 0

    # local DoF
    for i in range(0, NoN):
        for j in range(0, PD):
            # dirichlet node
            if ENL[i, PD+j] == -1:
                DOCs -= 1
                ENL[i, 2*PD+j] = DOCs
            # neumann node
            else:
                DOFs += 1
                ENL[i, 2*PD+j] = DOFs

    # global DoF
    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, 2*PD+j] < 0:
                ENL[i, 3*PD+j] = abs(ENL[i, 2*PD+j]) + DOFs
            else:
                ENL[i, 3*PD+j] = abs(ENL[i, 2 * PD + j])

    #
    DOCs = sbs(DOCs)

    #
    return (ENL, DOFs, DOCs)

#
def assemble_stiffness(ENL, EL, NL, E, A):
    NoE = np.size(EL, 0)
    NPE = np.size(EL, 1)

    PD = np.size(NL,1)
    NoN = np.size(NL,0)

    K = np.zeros([NoN*PD, NoN*PD])

    for i in range(0, NoE):
        n1 = EL[i, 0:NPE]
        k = element_stiffness(n1, ENL, E, A)

        for r in range(0, NPE):
            for p in range(0, PD):
                for q in range(0, NPE):
                    for s in range(0, PD):
                        row = ENL[n1[r]-1, p+3*PD]
                        col = ENL[n1[q]-1, s+3*PD]
                        value = k[r*PD+p, q*PD+s]
                        K[int(row)-1, int(col)-1] += value
    #
    return K

#
def element_stiffness(n1, ENL, E, A):
    X1 = ENL[n1[0] - 1, 0]
    Y1 = ENL[n1[0] - 1, 1]
    X2 = ENL[n1[1] - 1, 0]
    Y2 = ENL[n1[1] - 1, 1]

    L = math.sqrt((X1-X2)**2+(Y1-Y2)**2)

    c = (X2-X1)/L
    s = (Y2-Y1)/L
    k = (E*A)/L * np.array([[c**2, c*s, -c**2, -c*s],
                            [c*s, s**2, -c*s, -s**2],
                            [-c**2, -c*s, c**2, c*s],
                            [-c*s, -s**2, c*s, s**2]])

    #
    returm k

#
def assemble_forces(ENL, NL):
    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)


#
(ENL, DOFs, DOCs) = assign_BCs(NL, ENL)

# stiffness

K = assemble_stiffness(ENL, EL, NL, E, A)

#
ENL[:,4*PD:5*PD] = Uu[:, :]
ENL[:,5*PD:6*PD] = Fu[:, :]

#
Uu = Uu.flatten()
Fu = Fu.flatten()

#
Fp = assemble_forces(ENL, NL)
Up = assemble_displacement(ENL, NL)







