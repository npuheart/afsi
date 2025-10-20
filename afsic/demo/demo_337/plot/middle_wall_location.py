# output the location of a line in the middle of the ventricular wall

import numpy as np
file = 'ideal_middle_wall.txt'
a = 18.5
b = 8.5  
theta = np.linspace(0, 2 * np.pi, 200)
x = a * np.cos(theta)
y = b * np.sin(theta)
mask = x < 5
x = x[mask]
y = y[mask]
mask = y < 0
x = x[mask]
y = y[mask]
np.savetxt(file, np.column_stack((x, y)))
data = np.loadtxt(file)
