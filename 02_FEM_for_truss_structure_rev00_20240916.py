#
# TITLE: FEM for truss structure
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: https://www.youtube.com/watch?v=tv1TlAebvm0&list=PLnT2pATp7adWUaMh0A8jfoN_OHGAmt_m4&index=2
#
#

import numpy as np

# nodes
NL = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

# elements
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

    DOFs = 0
    DOCs = 0

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD+j] == -1:
                DOCs -= 1
                ENL[i, 2*PD+j] = DOCs
            else:
                DOFs += 1
                ENL[i, 2*PD+j] = DOFs


#
(ENL, DOFs, DOCs) = assign_BCs(NL, ENL)







