#
# TITLE: FEM for truss structure
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: https://www.youtube.com/watch?v=WdCctgAscW0&list=PLnT2pATp7adWUaMh0A8jfoN_OHGAmt_m4&index=4
#
# PD: Problem Dimension
# NoN: Number of Nodes
# NoE: Number of Elements
# NPE: Nodes per Element
# ENL: Extended Node List > size of (NoN x 6*PD)
#


import numpy as np
import matplotlib.pyplot as plt


#===============================================================================================
# https://www.youtube.com/watch?v=tv1TlAebvm0&list=PLnT2pATp7adWUaMh0A8jfoN_OHGAmt_m4&index=4
#===============================================================================================

# FUNCTION ---------------------------------------
def uniform_mesh(d1, d2, p, m, element_type):
    #
    PD = 2

    # corner points
    q = np.array([[0.0, 0.0],
                  [d1, 0.0],
                  [0.0, d2],
                  [d1, d2]])

    #
    NoN = (P + 1) * (m + 1)
    NoE = p * m
    NPE = 4

    # nodes
    NL = np.zeros([NoN, PD])

    a = (q[1,0]-q[0,0]) / p         # increments in the x direction (horizontal)
    b = (q[2, 1] - q[0, 1]) / m     # increments in the y direction (vertical)

    n = 0                           # rows in NL

    for i in range(1, m+2):
        for j in range(1, p+2):
            # coordinates
            NL[n, 0] = q[0, 0] + (j - 1) * a    # x
            NL[n, 1] = q[0, 1] + (i - 1) * b    # y

            # loop
            n += 1

    # elements





# initialization
d1 = 1              # length in x direction
d2 = 1              # length in y direction
p = 4               # number of nodes in x direction
m = 3               # number of nodes in y direction

# element
element_type = 'D2QU4N'

#
NL, EL = uniform_mesh(d1, d2, p, m, element_type)





