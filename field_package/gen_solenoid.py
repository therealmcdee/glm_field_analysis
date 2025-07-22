import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation as R

def solenoid(r, turndensity, length, appcurrent, senspositions, coilpos, coilorient):
    z = np.asarray(np.linspace(-length/2, length/2, round(turndensity*length)))

    N = turndensity*length

    a = 2*np.pi*turndensity

    x = r*np.cos(a*z)
    y = r*np.sin(a*z)

    solenoid = magpy.Collection()
    verts = np.zeros((len(z),3))
    verts[:,0] = x
    verts[:,1] = y
    verts[:,2] = z


    gpos = coilpos
    rotvec = coilorient
    orient = R.from_rotvec(rotvec)

    solenoid.add(magpy.current.Polyline(position = gpos,orientation = orient,vertices = verts,current = appcurrent))

    bfield = solenoid.getB(senspositions)
    mat = np.zeros((len(senspositions), 6))
    mat[:,:3] = senspositions
    for i in range(3):
        mat[:,3+i] = bfield[:,i]
    return mat
