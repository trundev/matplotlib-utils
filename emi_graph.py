import numpy as np
import matplotlib.pyplot as plt
import emi_calc

U = 20

#
# From emi.py
#
SOURCE_FMT = dict(color='green', label='Source')
B_FMT = dict(color='magenta', linestyle='--', label='Field')
gradB_FMT = dict(color='yellow', linestyle=':', label='Field gradient')
#
# End of emi.py
#
gradB_FMT['linestyle'] = '-'


# Magnetic field (single charge): B = y / sqrt(x^2 + y^2)^3
# y = B * sqrt(x^2 + y^2)^3
# y^(2/3) = B^(2/3) * x^2 + B^(2/3) * y^2
# B^(2/3) * x^2 = y^(2/3) - B^(2/3) * y^2
# x = +/- sqrt( (y/B)^(2/3) - y^2 )
B = 10
Ymax = B ** -.5     # Max Y in order "(y/B)^(2/3) - y^2 >= 0"
y = np.linspace(-Ymax, Ymax, U)
# Add +/-0 in the middle
y[int(y.size / 2) - 1] = -0.
y[int(y.size / 2)] = +0.

x = np.sqrt( ((y / B) ** 2) ** (1/3) - y ** 2 )
x *= np.sign(y)     # Revert sign loss by y ** 2
y = np.abs(y)

pts = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), 1)

#
# Magnetic field gradient:
#   dB/dx = -3*x*y / (x^2 + y^2) ^ (5/2)
#   dB/dy = (x^2 - 2*y^2) / (x^2 + y^2) ^ (5/2)

# The common divisor: (x^2 + y^2) ^ (5/2)
pts2 = pts * pts
div = (pts2.sum(-1)) ** (5 / 2)

# -3*x*y / (x^2 + y^2) ^ (5/2)
dB_dx = -3 * pts[:,0] * pts[:,1] / div
# (x^2 - 2*y^2) / (x^2 + y^2) ^ (5/2)
dB_dy = (pts2[:,0] - 2 * pts2[:,1]) / div

gradB_vecs = np.concatenate((dB_dx.reshape(-1, 1), dB_dy.reshape(-1, 1)), 1)

#
# Matplotlib figure draw
#
fig = plt.figure()
ax = plt.axes()

pts = pts.transpose()
gradB_vecs = gradB_vecs.transpose()

ax.plot(*pts, **B_FMT)
ax.quiver(*pts, *gradB_vecs, **gradB_FMT)

plt.show()
