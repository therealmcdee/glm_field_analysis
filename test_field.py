import numpy as np
import matplotlib.pyplot as plt

import field_package




radius = 1
turndensity = 5/0.01
length = 1
applied_current = 10
coilpos = np.asarray([0,0,0])
coilorient = (0, 0, 0)

npts = 14
x = 0.2
#npts += 1
line = np.asarray(np.linspace(-x, x, npts))
grid = np.zeros((npts**3, 3))
cnt = 0
for i in range(npts):
    for j in range(npts):
        for k in range(npts):
            grid[cnt][0] = line[i]
            grid[cnt][1] = line[j]
            grid[cnt][2] = line[k]
            cnt += 1

onaxis = grid[(grid[:,0] == line[npts//2]) & (grid[:,1] == line[npts//2])]

measfield = field_package.solenoid(radius, turndensity, length, applied_current, grid, coilpos, coilorient)

#coildiff = 0.3
#measfield = field_package.helmholtz(radius, applied_current, coildiff*applied_current, grid, coilpos, coilorient)


noiseval = 30e-9
measfield[:,3:] = measfield[:,3:] + np.random.uniform(0, noiseval, np.shape(measfield[:,3:]))

magnitudes = np.sqrt(measfield[:,3]**2 + measfield[:,4]**2 + measfield[:,5]**2)
norm = max(magnitudes)

measfield[:,3:] = measfield[:,3:]/norm


maxl = 2

glms = field_package.find_best_fit(measfield, maxl)*norm

recfield = field_package.make_field(measfield[:,:3], maxl, glms)

bfield = recfield[(recfield[:,0]==line[npts//2]) & (recfield[:,1] == line[npts//2])]
bcompare = measfield[(measfield[:,0]==line[npts//2]) & (measfield[:,1]==line[npts//2])]


component = [r'$B_{x}$', r'$B_{y}$', r'$B_{z}$']

fig = plt.figure()
for i in range(3):
    plt.scatter(bcompare[:,2], bcompare[:,3+i]*norm, label = component[i])
    plt.scatter(bfield[:,2], bfield[:,3+i], label = f'rec {component[i]}')

plt.xlabel('z [m]')
plt.ylabel('B [T]')
plt.legend()

fig2 = plt.figure()
#fig2, ax2 = plt.subplots(3,1, sharex=True)
for i in range(3):
    plt.scatter(bfield[:,2], bcompare[:,3+i]*norm - bfield[:,3+i], label = component[i])
    #ax2[i].scatter(bfield[:,2], bfield[:,3+i]*norm - recfield[:,3+i], label = component[i])
    #ax2[i].legend()

plt.ylabel(r'$B_{true}-B_{recon}$')
plt.xlabel('z')
plt.legend()

fig3 = plt.figure()
nb = 30
for i in range(3):
    plt.hist(measfield[:,3+i]*norm - recfield[:,3+i], bins = nb, alpha = 0.5, label = component[i])
plt.legend()

plt.show()

