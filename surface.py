from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import emi_calc

U = 20
V = 20

#
# From emi.py
#
SOURCE_FMT = dict(color='green', label='Source')
B_FMT = dict(color='magenta', linestyle='--', label='Field')
gradB_FMT = dict(color='yellow', linestyle=':', label='Field gradient')

def plot_source(ax, src_lines):
    src_pts = src_lines[...,:-1,:]
    src_dirs = src_lines[...,1:,:] - src_lines[...,:-1,:]
    return ax.quiver(*src_pts.transpose(), *src_dirs.transpose(), **SOURCE_FMT)
#
# End of emi.py
#
gradB_FMT['linestyle'] = '-'

if False:
# Sample original
    R = 2
    x = np.outer(np.linspace(-R, R, 30), np.ones(30))
    y = x.copy().T # transpose
    z = np.cos(x ** 2 + y ** 2)
    Rmax = R / 4
elif False:
# Sphere x^2 + y^2 + z^2 = R^2
    R = 4
    z = np.linspace(np.linspace(-R, -R, V), np.linspace(R, R, V), U)
    phy = np.linspace(np.linspace(-np.pi, np.pi, V), np.linspace(-np.pi, np.pi, V), U)

    r = np.sqrt(R ** 2 - z ** 2)
    x = r * np.cos(phy)
    y = r * np.sin(phy)
    Rmax = R
elif True:
# Magnetic field (single charge): B = r / sqrt(x^2 + r^2)^3
#   where: r = sqrt(y^2 + z^2)
# r = B * sqrt(x^2 + r^2)^3
# r^(2/3) = B^(2/3) * x^2 + B^(2/3) * r^2
# B^(2/3) * x^2 = r^(2/3) - B^(2/3) * r^2
# x = +/- sqrt( (r/B)^(2/3) - r^2 )
    B = 10
    Rmax = B ** -.5     # Max R in order "(r/B)^(2/3) - r^2 >= 0"
    #Rmax *= 1 - 1e-9   # ... a bit less
    r = np.linspace(np.linspace(-Rmax, -Rmax, V), np.linspace(Rmax, Rmax, V), U)
    phy = np.linspace(np.linspace(-np.pi, np.pi, V), np.linspace(-np.pi, np.pi, V), U)

    x = np.sqrt( ((r / B) ** 2) ** (1/3) - r ** 2 )
    x *= np.sign(r)     # Revert sign loss by r ** 2
    r = np.abs(r)
    y = r * np.cos(phy)
    z = r * np.sin(phy)
    Rmax /= 50
elif True:
# Magnetic field (integrated): B = x / (r * sqrt(x^2 + r^2))
#  B = x / ((x / x)^2 * r * sqrt(x^2 + r^2)) = x / (x^2 * (r/x) * sqrt(1 + (r/x)^2))
#  let r_x = r/x
#  B = 1 / (x * r_x * sqrt(1 + r_x^2))
#  x = 1 / (B * r_x * sqrt(1 + r_x^2))
# TODO:...
    pass

#
# Calculations at the points from x,y,z
#
src_line = np.array([ [0, 0, 0], [Rmax, 0, 0] ])
pts = np.concatenate((np.expand_dims(x, 2), np.expand_dims(y, 2), np.expand_dims(z, 2)), 2)

#
# Use emi_calc module to get the field generated by a src_line (integrated)
#
emi_params = emi_calc.calc_all_emis(pts, src_line)
B_vecs = emi_params['B']
gradB_vecs = emi_params['gradB']
print('gradB_vecs mean/max/min:', gradB_vecs.mean(), gradB_vecs.max(), gradB_vecs.min())
# This is sqrt(x^2 + y^2 + z^2))
B_lens = np.sqrt(np.sum(B_vecs * B_vecs, -1))
print('B_lens mean/max/min:', B_lens.mean(), B_lens.max(), B_lens.min())

# Scale down and reverse direction (will point to toward B decrease)
gradB_vecs *= -.05

if True:
# Try to find the "real" gradient-B vectors
# Magnetic field (single charge): B = r / sqrt(x^2 + r^2)^3
# Derivatives using https://www.derivative-calculator.net/:
#   - dx:
#       calculator:
#           input: r / sqrt(x^2 + r^2)^3, result: - (3*r*x) / (x^2 + r^2) ^ (5/2)
#       final:
#           dB/dx = -3*r*x / (x^2 + r^2) ^ (5/2)
#   - dr:
#       exchange 'x' and 'r' in calculator:
#           input: x / sqrt(x^2 + r^2)^3, result: - (2*x^2 - r^2) / (x^2 + r^2) ^ (5/2)
#       revert 'x' / 'r' exchange:
#           dB/dr = (x^2 - 2*r^2) / (x^2 + r^2) ^ (5/2)
    # Obtain unit vector in y/z plane (y_z) and the magnitude (r)
    y_z = pts[..., 1:3].copy()
    r = np.sqrt((y_z * y_z).sum(-1))
    y_z[..., 0] /= r
    y_z[..., 1] /= r

    # Calculate the common divisor: (x^2 + r^2) ^ (5/2)
    div = (pts[..., 0] ** 2 + r ** 2) ** (5/2)

    # Calculate dB/dx and dB/dr
    vx = -3 * r * pts[..., 0] / div
    dB = (pts[..., 0] ** 2 - 2 * r ** 2) / div

    y_z[..., 0] *= dB
    y_z[..., 1] *= dB
    gradB_vecs = np.concatenate((np.expand_dims(vx, 2), y_z), 2)
    print('gradB_vecs "real" mean/max/min:', gradB_vecs.mean(), gradB_vecs.max(), gradB_vecs.min())

    # Reverse direction (will point to toward B decrease)
    gradB_vecs *= -.0005

# Magnetic field from differential Biot–Savart law "R/sqrt(x^2 + R^2)^3"
# R -> length of y/z plane projection
pts2 = pts * pts
R = np.sqrt(pts2[..., 1:3].sum(-1))
B_vals = R / pts2.sum(-1) ** (3/2)
print('B_vals mean/max/min:', B_vals.mean(), B_vals.max(), B_vals.min())


fig = plt.figure()
ax = plt.axes(projection='3d')

plot_source(ax, src_line)
ax.quiver(*pts.transpose(), *B_vecs.transpose(), **B_FMT)
ax.quiver(*pts.transpose(), *gradB_vecs.transpose(), **gradB_FMT)

ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=.75)
ax.set_title('Surface plot')
plt.show()
