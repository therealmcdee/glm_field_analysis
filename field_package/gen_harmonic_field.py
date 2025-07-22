import numpy as np 
from scipy.optimize import minimize
from sympy import exp, I, re, im, assoc_legendre, symbols, factorial, Abs, cos, sin, diff, lambdify, sqrt, pi

####------------------------------------------------------------####

def spherical_conversion(mat):
    new = np.zeros_like(mat)
    for i, coords in enumerate(mat):
        x,y,z = coords
        radius = np.linalg.norm(coords)
        polar = np.arccos(z/radius)
        azimuth = np.arctan2(y, x)
        new[i,:] = np.array([radius, polar, azimuth])
    return new

####-----------------------------------------------------------####

def glm_field(a, b):
    l, m, r, t, p, = symbols("l m r t p")
    norm = factorial(l-1)*(-2**Abs(m))/factorial(l+Abs(m))
    U = norm*(r**l)*exp(I*Abs(m)*p)*assoc_legendre(l, Abs(m), cos(t))
    if b>=0:
        Br_expr = re(diff(U.subs([(l, a), (m,b)]), r))
        Bt_expr = re((1/r)*diff(U.subs([(l,a), (m,b)]), t))
        Bp_expr = re((1/(r*sin(t)))*diff(U.subs([(l,a), (m,b)]), p))
    if b<0:
        Br_expr = im(diff(U.subs([(l, a), (m,b)]), r))
        Bt_expr = im((1/r)*diff(U.subs([(l,a), (m,b)]), t))
        Bp_expr = im((1/(r*sin(t)))*diff(U.subs([(l,a), (m,b)]), p))

    bx_expr = sin(t)*cos(p)*Br_expr + cos(t)*cos(p)*Bt_expr - sin(p)*Bp_expr
    by_expr = sin(t)*sin(p)*Br_expr + cos(t)*sin(p)*Bt_expr + cos(p)*Bp_expr
    bz_expr = cos(t)*Br_expr - sin(t)*Bt_expr

    bx = lambdify([r, t, p], bx_expr, "numpy")
    by = lambdify([r, t, p], by_expr, "numpy")
    bz = lambdify([r, t, p], bz_expr, "numpy")
    return bx, by, bz

###------------------------------------------------------------####

def basis_matrix(positions, maxl):
    npts = len(positions)
    spher_pos = spherical_conversion(positions)
    totalf = 0
    for i in range(maxl):
        totalf += 2*i+3
    xbasis = np.zeros((totalf, npts))
    ybasis = np.zeros((totalf, npts))
    zbasis = np.zeros((totalf, npts))
    cnt = 0
    for i in range(maxl):
        mvals = np.arange(-(i+1),i+2)
        for j in range(len(mvals)):
            bx, by, bz = glm_field(i+1, mvals[j])
            xbasis[cnt,:] = bx(spher_pos[:,0], spher_pos[:,1], spher_pos[:,2])
            ybasis[cnt,:] = by(spher_pos[:,0], spher_pos[:,1], spher_pos[:,2])
            zbasis[cnt,:] = bz(spher_pos[:,0], spher_pos[:,1], spher_pos[:,2])
            cnt += 1
    return xbasis, ybasis, zbasis

####-----------------------------------------------------------####

def make_field(positions, maxl, coefs):
    xb, yb, zb = basis_matrix(positions, maxl)
    fullfield = np.zeros((len(positions), 6))
    fullfield[:,:3] = positions[:,:]
    fullfield[:,3] = np.matmul(coefs, xb)
    fullfield[:,4] = np.matmul(coefs, yb)
    fullfield[:,5] = np.matmul(coefs, zb)
    return fullfield

