#
# TITLE: FEM for truss structure
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: https://www.youtube.com/watch?v=tv1TlAebvm0&list=PLnT2pATp7adWUaMh0A8jfoN_OHGAmt_m4&index=2
#
# PD: problem dimension
# NoN: Number of Nodes
# NoE: Number of Elements
# NPE: Nodes per Element
# ENL: Extended Node List > size of (NoN x 6*PD)
#

import numpy as np
import math

# Node List (NoN x PD)
# coordinates (x, y)
NL = np.array([[0.0, 0.0],
               [1.0, 0.0],
               [0.5, 1.0]])

# Element List (NoE x NPE)
# (first node, second node)
EL = np.array([[1, 2],
               [2, 3],
               [3,1]])

# Boundary Conditions at Nodes
# +1: neumann boundary condition (free to move)
# -1: dirichlet boundary condition (fixed)
DorN = np.array([[-1, -1],
                 [+1, -1],
                 [+1, +1]])

# Force at nodes
Fu = np.array([[0.0, 0.0],
               [0.0, 0.0],
               [0.0, -20.0]])

# displacement at nodes
# fixed = 0.0
# free to move = 0.0 (initial)
Uu = np.array([[0.0, 0.0],
               [0.0, 0.0],
               [0.0, 0.0]])

# Young's modulus [Pa]
E = 10**6

# cross sectional area [m^2]
A = 0.01

# Problem Dimension,
# number of columns = 2
PD = np.size(NL, 1)

# Number of Nodes,
# number of rows = 3
NoN = np.size(NL, 0)

# Extended Node List
ENL = np.zeros([NoN, 6*PD])

# 0 * PD ~ 1 * PD = Nodes List
# 1 * PD ~ 2 * PD = Boundary Conditions
ENL[:,0*PD:1*PD] = NL[:,:]
ENL[:,1*PD:2*PD] = DorN[:,:]


# FUNCTION ---------------------------------------
def assign_BCs(NL, ENL):
    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)

    # initial value
    DOFs = 0
    DOCs = 0

    # local DoF
    # 2 * PD ~ 3 * PD = Local DoF
    for i in range(0, NoN):
        for j in range(0, PD):
            # dirichlet node
            if ENL[i, 1*PD+j] == -1:
                DOCs -= 1
                ENL[i, 2*PD+j] = DOCs
            # neumann node
            else:
                DOFs += 1
                ENL[i, 2*PD+j] = DOFs

    # global DoF
    # 3 * PD ~ 4 * PD = Global DoF
    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, 2*PD+j] < 0:
                ENL[i, 3*PD+j] = abs(ENL[i, 2*PD+j]) + DOFs
            else:
                ENL[i, 3*PD+j] = abs(ENL[i, 2 * PD + j])

    #
    DOCs = abs(DOCs)

    # return
    return (ENL, DOFs, DOCs)


# FUNCTION ---------------------------------------
def assemble_stiffness(ENL, EL, NL, E, A):
    NoE = np.size(EL, 0)    # row count
    NPE = np.size(EL, 1)    # column count

    PD = np.size(NL,1)      # column count
    NoN = np.size(NL,0)     # row count

    K = np.zeros([NoN*PD, NoN*PD])

    for i in range(0, NoE):
        n1 = EL[i, 0:NPE]
        k = element_stiffness(n1, ENL, E, A)

        # row loop
        for r in range(0, NPE):
            for p in range(0, PD):
                # column loop
                for q in range(0, NPE):
                    for s in range(0, PD):
                        # row, column
                        row = ENL[n1[r]-1, p+3*PD]
                        col = ENL[n1[q]-1, s+3*PD]
                        # k
                        value = k[r*PD+p, q*PD+s]
                        #
                        K[int(row)-1, int(col)-1] += value
    # return
    return K

# FUNCTION ---------------------------------------
def element_stiffness(n1, ENL, E, A):
    X1 = ENL[n1[0] - 1, 0]      # first node, x
    Y1 = ENL[n1[0] - 1, 1]      # first node, y
    X2 = ENL[n1[1] - 1, 0]      # second node, x
    Y2 = ENL[n1[1] - 1, 1]      # second node, y

    L = math.sqrt((X1-X2)**2+(Y1-Y2)**2)

    c = (X2-X1)/L
    s = (Y2-Y1)/L
    k = (E*A)/L * np.array([[c**2, c*s, -c**2, -c*s],
                            [c*s, s**2, -c*s, -s**2],
                            [-c**2, -c*s, c**2, c*s],
                            [-c*s, -s**2, c*s, s**2]])

    # return
    return k

# FUNCTION ---------------------------------------
def assemble_forces(ENL, NL):
    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)

    #
    DOF = 0

    #
    Fp = []

    #
    for i in range(0, NoN):
        for j in range(0, PD):
            # fixed node
            if ENL[i, 1*PD+j] == 1:
                DOF += 1
                Fp.append(ENL[i, 5*PD+j])

    # 1d array
    Fp = np.vstack([Fp]).reshape(-1, 1)

    # return
    return Fp

# FUNCTION ---------------------------------------
def assemble_displacement(ENL, NL):
    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)

    #
    DOC = 0

    #
    Up = []

    #
    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD + j] == -1:
                DOC += 1
                Up.append(ENL[i, 4 * PD + j])

    # 1d array
    Up = np.vstack([Up]).reshape(-1, 1)

    # return
    return Up



# 2 * PD ~ 3 * PD = Local DoF
# 3 * PD ~ 4 * PD = Global DoF
(ENL, DOFs, DOCs) = assign_BCs(NL, ENL)

# stiffness matrix
K = assemble_stiffness(ENL, EL, NL, E, A)

# 4 * PD ~ 5 * PD = Displacement
# 5 * PD ~ ^ * PD = Force
ENL[:, 4*PD:5*PD] = Uu[:, :]
ENL[:, 5*PD:6*PD] = Fu[:, :]

#
Uu = Uu.flatten()
Fu = Fu.flatten()

#
Fp = assemble_forces(ENL, NL)
Up = assemble_displacement(ENL, NL)

# debug
print(Fp)
print(Up)


