import numpy as np
from scipy.optimize import minimize


from field_package.gen_harmonic_field import make_field

####--------------------------------------------------------------------####

def scalar_residual(coefs, measfield, maxl):
    reconstructed = make_field(measfield[:,:3], maxl, coefs)
    bxoff = sum((reconstructed[:,3]-measfield[:,3])**2)
    byoff = sum((reconstructed[:,4]-measfield[:,4])**2)
    bzoff = sum((reconstructed[:,5]-measfield[:,5])**2)
    result = bxoff + byoff + bzoff
    print(result)
    return np.sqrt(result)

def find_best_fit(measfield, maxl):
    initial = initial_guess(measfield, maxl)
    print(initial)
    total = 0
    for i in range(maxl):
        total += 2*i+3
    res = minimize(scalar_residual, x0 = initial, args = (measfield, maxl), tol = 1e-2)#options = {'ftol':1e-6})
    return res.x

def initial_guess(measfield, maxl):
    total = 0
    for i in range(maxl):
        total+= 2*i+3
    initguess = np.zeros(total)
    bxav = sum(measfield[:,3])/len(measfield)
    byav = sum(measfield[:,4])/len(measfield)
    bzav = sum(measfield[:,5])/len(measfield)
    upper = measfield[(measfield[:,2]>0)]
    lower = measfield[(measfield[:,2]<0)]
    left = measfield[(measfield[:,1]>0)]
    right = measfield[(measfield[:,1]<0)]
    forward = measfield[(measfield[:,0]>0)]
    backward = measfield[(measfield[:,0]<0)]
    initguess[0] = -byav
    initguess[1] = bzav
    initguess[2] = -bxav
    return np.asarray([initguess])

