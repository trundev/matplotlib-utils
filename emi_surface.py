'''Equipotential electromagnetic surface visualization
'''
import sys
import numpy
import matplotlib.pyplot as pyplot
import mpl_toolkits.mplot3d as mplot3d
import emi_calc


U_MAX = 10
V_MAX = 10
MAX_ITERATIONS = 10

#
# Source current flow
#
SRC_Z_STEP = 0  # 0.01
SOURCE_POLYLINE = [
   [0., 0., 0 * SRC_Z_STEP],
   [1., 0., 1 * SRC_Z_STEP],    # right
   [1., 1., 2 * SRC_Z_STEP],    # up
   [0., 1., 3 * SRC_Z_STEP],    # left
   [0., 0., 4 * SRC_Z_STEP],    # down
]

BASE_POINT = [0.5, -.1, .0]

#
# From emi.py
#
SOURCE_FMT = dict(color='green', label='Source')
TARGET_FMT = dict(color='blue', marker='+', label='Target')
B_FMT = dict(color='magenta', linestyle='--', label='Field')
gradB_FMT = dict(color='yellow', linestyle=':', label='Gradient')

def plot_source(ax, src_lines):
    src_pts = src_lines[...,:-1,:]
    src_dirs = src_lines[...,1:,:] - src_lines[...,:-1,:]
    return ax.quiver(*src_pts.transpose(), *src_dirs.transpose(), **SOURCE_FMT)
#
# End of emi.py
#
gradB_FMT['linestyle']='-'
FAIL_FMT = dict(color='red', marker='o', label='Failure')
WARNING_FMT = dict(color='orange', marker='o', label='Warning')
SURFACE_FMT = dict(cmap='viridis', edgecolor='none', alpha=.5)

#
# Equipotential surface approximation
#
def step_along(pt, vect, center, coef):
    center_v = pt - center
    return pt + vect / numpy.sqrt(vect.dot(vect) / center_v.dot(center_v)) * coef

def equipotential_surface(base_pt, src_lines, tolerance=.01):
    base_params = emi_calc.calc_all_emis(base_pt, src_lines)
    surface = numpy.full((V_MAX, U_MAX), numpy.nan, dtype=base_params.dtype)
    info = numpy.zeros(surface.shape, dtype=[('iter', numpy.int)])
    base_B = base_params['B']
    base_B_len = numpy.sqrt(base_B.dot(base_B))

    # Step parameters
    center_pt = src_lines.sum(0) / src_lines.shape[0]
    revolution = 2 * numpy.pi
    tan_phi = numpy.tan(revolution / 3 / U_MAX)
    tan_psi = numpy.tan(revolution / 3 / V_MAX)

    # Iterate along direction of "gradB x B" (v)
    pt = base_params['pt']
    v = 0
    while v < V_MAX:
        # Iterate along direction of "B" (u)
        base_params = None
        u = 0
        while u < U_MAX:
            print('Point', (u, v))
            # Step along direction of "gradB" to find the exact location
            fail = True
            info[v, u]['iter'] = 0
            for iter in range(MAX_ITERATIONS):
                emi_params = emi_calc.calc_all_emis(pt, src_lines)
                B = emi_params['B']
                gradB = emi_params['gradB']
                B_len = numpy.sqrt(B.dot(B))
                dB = base_B_len - B_len
                if abs(dB) < tolerance * base_B_len:
                    surface[v, u] = emi_params
                    info[v, u]['iter'] = iter + 1
                    fail = False
                    break

                # Location correction: "dB / |gradB|" along the normalized "gradB"
                pt += gradB * dB / gradB.dot(gradB)
                print('  pt', pt, 'dB', dB, '(%.1f%%)'%(dB / base_B_len * 100))

            if fail:
                print('Error: Unable to find B', B, ' around', pt, file=sys.stderr)
                surface[v, u]['pt'] = pt

            # Keep the first "u" point location for the next "v" iteration
            if base_params is None:
                base_params = emi_params

            # Do step along "u", moving toward "B", but perpendicular to "gradB"
            # (this is the vector "gradB x B x gradB")
            pt = step_along(pt, numpy.cross(numpy.cross(gradB, B), gradB), center_pt, tan_phi)
            u += 1

        # Do step along "v", moving perpendicular to "u"
        pt = base_params['pt']
        pt = step_along(pt, numpy.cross(base_params['gradB'], base_params['B']), center_pt, tan_psi)
        v += 1

    iters = info['iter']
    failed_pts = numpy.count_nonzero(iters == 0)
    max_idx = numpy.array((iters >= iters.max()).nonzero())
    max_idx = max_idx.transpose()
    print('Max iterations', iters.max(), 'at:', max_idx, ',', failed_pts, 'failed')
    return surface, info

#
# Utilities
#
def vect_dot(v0, v1):
    """Dot product of vectors from a numpy array (vectors in the last dimension)"""
    return (v0 * v1).sum(-1)

def vect_lens(v):
    """Length of vectors in a numpy array (vectors in the last dimension)"""
    return numpy.sqrt(vect_dot(v, v))

def strip_nans(a):
    """Strip NaN-s from an array, result is flatten"""
    nans = a.size
    a = numpy.extract(numpy.invert(numpy.isnan(a)), a)
    nans -= a.size
    return a, nans

#
# Data validation
#
def check_result(src_lines, surface):
    pts = surface['pt']
    B_vecs = surface['B']
    gradB_vecs = surface['gradB']
    ret = True

    print('Re-check equipotential surface:', B_vecs.shape[:-1])
    print('  B vectors')
    B_vec_lens = vect_lens(B_vecs)
    B_vec_lens, nans = strip_nans(B_vec_lens)
    print('    Lengths (mean/max/min/std):', B_vec_lens.mean(), B_vec_lens.max(), B_vec_lens.min(), B_vec_lens.std(), ',', nans, 'nans')
    if B_vec_lens.std() > 5e-2:
        print('Warning: Standard deviation is too big:', B_vec_lens.std(), file=sys.stderr)

    print('  Recalculate B vectors')
    params = emi_calc.calc_all_emis(pts, src_lines)
    B_recheck = params['B']
    B_dif = B_recheck - B_vecs
    if B_dif.min() != 0 or B_dif.max() != 0:
        print('Warning: B_recheck - B_vec (max/min):', B_dif.max(), B_dif.min(), file=sys.stderr)
        ret = False
    B_recheck_lens = vect_lens(B_recheck)
    B_recheck_lens, nans = strip_nans(B_recheck_lens)
    print('    Lengths (mean/max/min):', B_recheck_lens.mean(), B_recheck_lens.max(), B_recheck_lens.min(), ',', nans, 'nans')

    # Angle between B and grad-B (perpendicular when the source is a line)
    print('  B perpendicular to gradB (B . gradB ~ 0)')
    B_dot_gradB = vect_dot(B_vecs, gradB_vecs)
    B_dot_gradB, nans = strip_nans(B_dot_gradB)
    print('    B . gradB (max/min):', B_dot_gradB.max(), B_dot_gradB.min(), ',', nans, 'nans')
    # Angle between (B . gradB) / (|B| * |gradB|)
    gradB_vec_lens = vect_lens(gradB_vecs)
    gradB_vec_lens, _ = strip_nans(gradB_vec_lens)
    if B_dot_gradB.size == gradB_vec_lens.size and gradB_vec_lens.size == B_vec_lens.size:
        B_gradB_angle = B_dot_gradB / (B_vec_lens * gradB_vec_lens)
        B_gradB_angle = numpy.arccos(B_gradB_angle)
        B_gradB_angle *= 180 / numpy.pi
        print('    Angle between B and gradB (max/min deg):', B_gradB_angle.max(), B_gradB_angle.min())
    else:
        print('Warning: Different number of non-nan values in "B . gradB", "gradB" and "B":',
                B_dot_gradB.size, gradB_vec_lens.size, B_vec_lens.size,
                file=sys.stderr )
        ret = False
    return ret

#
# Plot
#
def main(base_pt, src_lines):
    """Main entry"""
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')

    surface, info = equipotential_surface(base_pt, src_lines)

    # Re-check results
    if True:
        check_result(src_lines, surface)

    # Source line and base target point
    plot_source(ax, src_lines)
    ax.scatter(*base_pt, **TARGET_FMT)

    pts = surface['pt']
    B_vecs = surface['B']
    gradB_vecs = surface['gradB']

    # Identify failed points
    pts_fails = numpy.where(numpy.isnan(B_vecs), pts, numpy.nan)
    ax.scatter(*pts_fails.transpose(), **FAIL_FMT)

    # Identify the points, where approximation takes more iterations
    median = int(numpy.median(info['iter']))
    print('"Warning" marker is where iterations are more than', median)
    where = info['iter'] > median
    where = numpy.stack(pts.shape[-1]*[where], axis=-1)
    pts_warns = numpy.where(where, pts, numpy.nan)
    ax.scatter(*pts_warns.transpose(), **WARNING_FMT)

    # Resize B and grad B to fit screen (ignore NaN-s)
    src_max = vect_lens(src_lines.max(0) - src_lines.min(0))
    B_max = numpy.nanmax(vect_lens(B_vecs))
    gradB_max = numpy.nanmax(vect_lens(gradB_vecs))
    # Scale field vectors to 1/20 of source lines
    B_scale = src_max / B_max / 20
    gradB_scale = src_max / gradB_max / 20
    print('Rescale B to %.3f and gradB to %f'%(B_scale, gradB_scale))
    B_vecs *= B_scale
    gradB_vecs *= gradB_scale

    pts = pts.transpose()
    B_vecs = B_vecs.transpose()
    gradB_vecs = gradB_vecs.transpose()
    ax.quiver(*pts, *B_vecs, **B_FMT)
    ax.quiver(*pts, *gradB_vecs, **gradB_FMT)
    ax.plot_surface(*pts, **SURFACE_FMT)

    ax.set_title('Equipotential surface')
    ax.legend()
    pyplot.show()
    return 0

if __name__ == '__main__':
    src_lines = numpy.array(SOURCE_POLYLINE)
    base_pt = numpy.array(BASE_POINT)
    exit(main(base_pt, src_lines))
