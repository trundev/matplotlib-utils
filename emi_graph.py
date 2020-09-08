import sys
import numpy as np
import matplotlib.pyplot as plt
import emi_calc

U = 20

EMI_CALC_SRC = None
EMI_CALC_SRC_DIF = None
# Source vector to validate 'emi_calc'
# (leave these None to skip this step)
if False:
    EMI_CALC_SRC_SCAL = 1e-3
    EMI_CALC_SRC = np.array([[-EMI_CALC_SRC_SCAL/2,0.,0.], [EMI_CALC_SRC_SCAL/2,0.,0.]])    # Min length along X
elif True:
    EMI_CALC_SRC_SCAL = 1.
    EMI_CALC_SRC_DIF = np.array([[0.,0.,0.], [EMI_CALC_SRC_SCAL,0.,0.]])    # Unit length along X


#
# From emi.py
#
SOURCE_FMT = dict(color='green', label='Source')
B_FMT = dict(color='magenta', linestyle='--', label='Field')
gradB_FMT = dict(color='yellow', linestyle=':', label='Field gradient')
new_gradB_FMT = dict(color='orange', linestyle=':', label='Field gradient (check)')
#
# End of emi.py
#
gradB_FMT['linestyle'] = '-'

def calc_all_emis_diff(tgt_pts, src_pairs, coef=1):
    """Modified version of emi_calc.calc_all_emis() to use calc_emi_dif()"""
    # Create empty EMI paramters structures
    emi_params = emi_calc.calc_all_emis(tgt_pts, np.zeros((0,3)))
    # Get B and Jacob-s
    src_pts = src_pairs[...,0,:].reshape((-1, 3))
    src_dirs = src_pairs[...,1,:].reshape((-1, 3))

    #
    # This is from emi_calc.calc_all_emis()
    #
    emi_it = np.nditer(emi_params, op_flags=[['readwrite']])
    for emi_pars in emi_it:
        for src_pt, src_dir in zip(src_pts, src_dirs):
            emi = emi_calc.calc_emi_dif(emi_pars['pt'], src_pt, src_dir, coef)
            # Ignore points collinear to the src-line and where jacobian failed
            if emi is not None and emi[1] is not None:
                if np.isnan(emi_pars['B']).all():
                    emi_pars['B'] = 0
                if np.isnan(emi_pars['jacob']).all():
                    emi_pars['jacob'] = 0

                emi_pars['B'] += emi[0]
                emi_pars['jacob'] += emi[1]
            else:
                print('Warning: EMI failed at %d: tgt=%s, src=%s->%s'%(emi_it.iterindex, emi_pars['pt'], src_pt, src_dir), file=sys.stderr)

    # Obtain 'gradB' from Jacobian matrix
    for emi_pars in np.nditer(emi_params, op_flags=[['readwrite']]):
        emi_pars['gradB'] = emi_calc.generate_gradB(emi_pars['B'], emi_pars['jacob'])
        emi_pars['dr_dI'] = emi_calc.generate_dr_dI(emi_pars['B'], emi_pars['jacob'])

    return emi_params

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

pts = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), -1)

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

gradB_vecs = np.concatenate((dB_dx.reshape(-1, 1), dB_dy.reshape(-1, 1)), -1)

# Validate field results from emi_calc
new_pts = None
emi_params = None
if EMI_CALC_SRC is not None:
    # Regular (integral) function
    coef = 1/EMI_CALC_SRC_SCAL
    print('Validating %d target points (integral routine), coef=%.1g'%(pts.size, coef))
    # Add Z axis with zeros
    pts3d = np.concatenate((pts, np.zeros((*pts.shape[:-1], 1))), -1)
    emi_params = emi_calc.calc_all_emis(pts3d, EMI_CALC_SRC, coef)
elif EMI_CALC_SRC_DIF is not None:
    # Differential (single partical) function
    coef = 1/EMI_CALC_SRC_SCAL
    print('Validating %d target points (differential routine), coef=%.1g'%(pts.size, coef))
    # Add Z axis with zeros
    pts3d = np.concatenate((pts, np.zeros((*pts.shape[:-1], 1))), -1)
    emi_params = calc_all_emis_diff(pts3d, EMI_CALC_SRC_DIF, coef)

if emi_params is not None:
    fail_cnt = 0
    # Check if target points are preserved
    new_pts = emi_params['pt'][...,:2]
    compare = (new_pts != pts).any(-1)
    if compare.any():
        print('Error: Target point difference:', compare, file=sys.stderr)

    # Check field
    new_B_vecs = emi_params['B']
    # Where there is failure
    failure = np.isnan(new_B_vecs).all(-1)

    # Check non-Z components, must be zero or NaN
    new_B_z = new_B_vecs[...,:2] 
    compare = (new_B_z != 0.).any(-1)
    compare[failure] = False
    if compare.any():
        cnt = compare[compare].size
        fail_cnt += cnt
        print('Error: Field includes non-Z components in %d points:'%cnt,
                compare.nonzero()[0], file=sys.stderr)
    del new_B_z

    # Check Z components, must be equal to B (allow .1% variation)
    compare = np.abs(new_B_vecs[...,2] - B) / B > 1e-3
    compare[failure] = False
    if compare.any():
        cnt = compare[compare].size
        fail_cnt += cnt
        max_var = np.abs(new_B_vecs[...,2] - B)[failure == False].max()
        print('Error: Field difference (%.1g / %.1g) in %d points:'%(max_var, B, cnt),
                compare.nonzero()[0], file=sys.stderr)
    
    # Check gradient
    new_gradB_vecs = emi_params['gradB']

    # Check non-XY components, must be zero or NaN
    new_gradB_z = new_gradB_vecs[...,2:] 
    compare = (new_gradB_z != 0.).any(-1)
    compare[failure] = False
    if compare.any():
        cnt = compare[compare].size
        fail_cnt += cnt
        print('Error: Gradient includes non-XY components in %d points:'%cnt,
                compare.nonzero()[0], file=sys.stderr)
    del new_gradB_z

    # Check XZ components, must be match dB (allow .1% variation)
    new_gradB_vecs = new_gradB_vecs[...,:2] 
    compare = np.abs(new_gradB_vecs - gradB_vecs)
    compare[gradB_vecs != 0] /= gradB_vecs[gradB_vecs != 0] # Avoid "nvalid value encountered in true_divide"
    compare = compare > 1e-3
    compare = compare.any(-1)
    compare[failure] = False
    if compare.any():
        cnt = compare[compare].size
        fail_cnt += cnt
        max_var = np.abs(new_gradB_vecs - gradB_vecs)[failure == False].max()
        print('Error: Gradient difference (%.1g) in %d points:'%(max_var, cnt),
                compare.nonzero()[0], file=sys.stderr)
    new_pts = new_pts[compare]
    new_gradB_vecs = new_gradB_vecs[compare]
    if new_pts.size == 0:
        new_pts = None
        del new_gradB_vecs

    print('Total failures', fail_cnt)

#
# Matplotlib figure draw
#
fig = plt.figure()
ax = plt.axes()

ax.plot(*pts.T, **B_FMT)
Q_gradB = ax.quiver(*pts.T, *gradB_vecs.T, **gradB_FMT)

if new_pts is not None:
    # Use same scale for new_gradB_vecs and gradB_vecs
    Q_gradB._init()     # Re-calculate scale
    ax.quiver(*new_pts.T, *new_gradB_vecs.T, scale=Q_gradB.scale, **new_gradB_FMT)

plt.show()
