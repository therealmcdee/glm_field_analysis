import numpy as np
import matplotlib.pyplot as plt

import field_package



data = np.loadtxt('field_package/example_fields/COMSOL_B0.txt', delimiter = '\t', skiprows = 9)

print(data)
