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



