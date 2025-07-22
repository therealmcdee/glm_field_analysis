import magpylib as magpy
import numpy as np 
from scipy.spatial.transform import Rotation as R

def helmholtz(radius, cur1, cur2, senspositions, coilpos, coilorient):
    combo_coil = magpy.Collection()
    orient = R.from_rotvec(coilorient)
    print(orient)
    shift = np.asarray([0, 0, radius/2])
    c1pos = coilpos - np.matmul(R.as_matrix(orient), shift)
    coil1 = magpy.current.Circle(c1pos, orient, 2*radius, cur1)
    combo_coil.add(coil1)

    c2pos = coilpos + np.matmul(R.as_matrix(orient), shift)
    coil2 = magpy.current.Circle(c2pos, orient, 2*radius, cur2)
    combo_coil.add(coil2)
    
    bfield = combo_coil.getB(senspositions)
    mat = np.zeros((len(senspositions), 6))
    mat[:,:3] = senspositions
    for i in range(3):
        mat[:,3+i] = bfield[:,i]
    return mat
