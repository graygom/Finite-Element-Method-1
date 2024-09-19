#
# TITLE: FEM for truss structure
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: https://www.youtube.com/watch?v=0zrq2N2qAVo&list=PLnT2pATp7adWUaMh0A8jfoN_OHGAmt_m4&index=1
#            https://www.youtube.com/watch?v=tv1TlAebvm0&list=PLnT2pATp7adWUaMh0A8jfoN_OHGAmt_m4&index=2
#            https://www.youtube.com/watch?v=WdCctgAscW0&list=PLnT2pATp7adWUaMh0A8jfoN_OHGAmt_m4&index=3
#
# PD: problem dimension
# NoN: Number of Nodes
# NoE: Number of Elements
# NPE: Nodes per Element
# ENL: Extended Node List > size of (NoN x 6*PD)
#


import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


#===============================================================================================
# https://www.youtube.com/watch?v=tv1TlAebvm0&list=PLnT2pATp7adWUaMh0A8jfoN_OHGAmt_m4&index=2
#===============================================================================================


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
            # dirichlet node (fixed)
            if ENL[i, 1*PD+j] == -1:
                DOCs -= 1
                ENL[i, 2*PD+j] = DOCs
            # neumann node (free to move)
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
            # +1: neumann boundary condition (free to move)
            if ENL[i, 1*PD+j] == 1:
                DOF += 1
                # force
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
            # -1: dirichlet boundary condition (fixed)
            if ENL[i, 1*PD+j] == -1:
                DOC += 1
                # displacement
                Up.append(ENL[i, 4*PD+j])

    # 1d array
    Up = np.vstack([Up]).reshape(-1, 1)

    # return
    return Up


# FUNCTION ---------------------------------------
def update_nodes(ENL, Uu, NL, Fu):
    PD = np.size(NL, 1)
    NoN = np.size(NL, 0)

    DOFs = 0
    DOCs = 0

    for i in range(0, NoN):
        for j in range(0, PD):
            # +1: neumann boundary condition (free to move)
            if ENL[i, 1*PD+j] == 1:
                DOFs += 1
                ENL[i, 4*PD+j] = Uu[DOFs-1]
            # -1: dirichlet boundary condition (fixed)
            else:
                DOCs += 1
                ENL[i, 5*PD+j] = Fu[DOCs-1]
    # return
    return ENL


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

#
# Fp  Kuu Kup  Uu
# Fu  Kpu Kpp  Up
#
Kuu = K[0*DOFs:1*DOFs, 0*DOFs:1*DOFs]
Kup = K[0*DOFs:1*DOFs, 1*DOFs:1*DOFs+1*DOCs]
Kpu = K[1*DOFs:1*DOFs+1*DOCs, 0:DOFs]
Kpp = K[1*DOFs:1*DOFs+1*DOCs, 1*DOFs:1*DOFs+1*DOCs]

#
F = Fp - np.matmul(Kup, Up)
Uu = np.matmul(np.linalg.inv(Kuu), F)
Fu = np.matmul(Kpu, Uu) + np.matmul(Kpp, Up)

#
ENL = update_nodes(ENL, Uu, NL, Fu)

# debug
print(Fp)
print(Up)
print(Fu)
print(Uu)
print(ENL)


#===============================================================================================
# https://www.youtube.com/watch?v=tv1TlAebvm0&list=PLnT2pATp7adWUaMh0A8jfoN_OHGAmt_m4&index=3
#===============================================================================================

# exaggeration
scale = 100

# initialization
coor = []
dispx_array = []

# make array
for i in range(np.size(NL, 0)):
    # displacement
    dispx = ENL[i, 8]
    dispy = ENL[i, 9]

    # position + displacement * scale
    x = ENL[i, 0] + dispx * scale
    y = ENL[i, 1] + dispy * scale

    # displacement x
    dispx_array.append(dispx)

    # coordinates (x, y)
    coor.append(np.array([x, y]))

# numpy array, vertical stack
coor = np.vstack(coor)
dispx_array = np.vstack(dispx_array)

# initialzation
x_scatter = []
y_scatter = []
color_x = []

#
for i in range(0, np.size(EL, 0)):
    # two nodes in an element
    x1 = coor[EL[i, 0] - 1, 0]
    x2 = coor[EL[i, 1] - 1, 0]
    y1 = coor[EL[i, 0] - 1, 1]
    y2 = coor[EL[i, 1] - 1, 1]

    #
    dispx_EL = np.array([dispx_array[EL[i, 0] - 1], dispx_array[EL[i, 1] - 1]])

    # two cases
    if x1 == x2:
        x = np.linspace(x1, x2, 200)
        y = np.linspace(y1, y2, 200)
    else:
        m = (y2 - y1) / (x2 - x1)
        x = np.linspace(x1, x2, 200)
        y = m * (x - x1) + y1

    #
    x_scatter.append(x)
    y_scatter.append(y)

    #
    color_x.append(np.linspace(np.abs(dispx_EL[0]), np.abs(dispx_EL[1]), 200))

#
x_scatter = np.vstack([x_scatter]).flatten()
y_scatter = np.vstack([y_scatter]).flatten()
color_x = np.vstack([color_x]).flatten()

# plot 1
dispFig = plt.figure(1)
ax_dispx = dispFig.add_subplot(111)

cmap = plt.get_cmap('jet')
ax_dispx.scatter(x_scatter, y_scatter, c=color_x, cmap=cmap, s=10, edgecolor='none')

# normalization & colormap
norm_x = Normalize(np.abs(dispx_array.min()), np.abs(dispx_array.max()))
#dispFig.colorbar(ScalarMappable(norm=norm_x, cmap=cmap))

plt.show()









