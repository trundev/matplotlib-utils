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
# From emi.py
#
SOURCE_FMT = dict(color='green', label='Source')
TARGET_FMT = dict(color='blue', marker='+', label='Target')
B_FMT = dict(color='magenta', linestyle='--', label='Field')
gradB_FMT = dict(color='yellow', linestyle=':', label='Field gradient')

def plot_source(ax, src_lines):
    src_lines = src_lines.transpose()
    return ax.quiver(*src_lines[:,:-1], *(src_lines[:,1:] - src_lines[:,:-1]), **SOURCE_FMT)
#
# End of emi.py
#
gradB_FMT['linestyle']='-'
FAIL_FMT = dict(color='red', marker='o', label='Failure')


def equipotential_surface(base_pt, src_lines, tolerance=.01):
    base_params = emi_calc.calc_all_emis(base_pt, src_lines)
    surface = numpy.full((V_MAX, U_MAX), numpy.nan, dtype=base_params.dtype)
    base_B = base_params['B']
    base_B_len = numpy.sqrt(base_B.dot(base_B))

    # Step parameters
    center_pt = src_lines.sum(0) / src_lines.shape[0]
    revolution = 2 * numpy.pi
    tan_phi = numpy.tan(revolution / 2 / U_MAX)
    tan_psi = numpy.tan(revolution / 4 / V_MAX)

    # Iterate along direction of "gradB x B" (v)
    failed_pts = 0
    max_iter = -1
    max_iter_u_v = None
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
            for iter in range(MAX_ITERATIONS):
                emi_params = emi_calc.calc_all_emis(pt, src_lines)
                B = emi_params['B']
                B_len = numpy.sqrt(B.dot(B))
                dB = base_B_len - B_len
                if abs(dB) < tolerance * base_B_len:
                    surface[v, u] = emi_params
                    fail = False
                    break

                # Location correction: "dB / |gradB|" along the normalized "gradB"
                gradB = emi_params['gradB']
                pt += gradB * dB / gradB.dot(gradB)
                print('  pt', pt, 'dB', dB, '(%.1f%%)'%(dB / base_B_len * 100))

            if fail:
                print('Error: Unable to find B', B, ' around', pt, file=sys.stderr)
                failed_pts += 1
                surface[v, u]['pt'] = pt
            elif max_iter < iter:
                max_iter = iter
                max_iter_u_v = u,v

            # Keep the first "u" point location for the next "v" iteration
            if base_params is None:
                base_params = emi_params

            # Do step along "u"
            center_dist = pt - center_pt
            center_dist = numpy.sqrt(center_dist.dot(center_dist))
            pt += B / B_len * center_dist * tan_phi
            u += 1

        # Do step along "v"
        pt = base_params['pt']
        center_dist = pt - center_pt
        center_dist = numpy.sqrt(center_dist.dot(center_dist))
        step_v = numpy.cross(base_params['gradB'], base_params['B'])
        pt += step_v / numpy.sqrt(step_v.dot(step_v)) * center_dist * tan_psi
        v += 1

    print('Max iterations', max_iter, 'at', max_iter_u_v, ',', failed_pts, 'failed')
    return surface

#
# Source current flow
#
SRC_Z_STEP = 0  # 0.01
SOURCE_POLYLINE = [
   [0., 0., 0 * SRC_Z_STEP],
   [1., 0., 1 * SRC_Z_STEP],    # right
#   [1., 1., 2 * SRC_Z_STEP],    # up
#   [0., 1., 3 * SRC_Z_STEP],    # left
#   [0., 0., 4 * SRC_Z_STEP],    # down
]

BASE_POINT = [0.5, -2., 1.]

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
# Plot
#
def main(base_pt, src_lines):
    """Main entry"""
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')

    surface = equipotential_surface(base_pt, src_lines)
    pts = surface['pt']
    B_vecs = surface['B']
    gradB_vecs = surface['gradB']

    # Re-check results
    if True:
        print('Re-check equipotential surface:', B_vecs.shape[:-1])
        print('  B vectors')
        B_vec_lens = vect_lens(B_vecs)
        B_vec_lens, nans = strip_nans(B_vec_lens)
        print('    Lengths (mean/max/min):', B_vec_lens.mean(), B_vec_lens.max(), B_vec_lens.min(), ',', nans, 'nans')

        print('  Recalculate B vectors')
        params = emi_calc.calc_all_emis(pts, src_lines)
        B_recheck = params['B']
        B_dif = B_recheck - B_vecs
        if B_dif.min() != 0 or B_dif.max() != 0:
            print('Warning: B_recheck - B_vec (max/min):', B_dif.max(), B_dif.min(), file=sys.stderr)
        B_recheck_lens = vect_lens(B_recheck)
        B_recheck_lens, nans = strip_nans(B_recheck_lens)
        print('    Lengths (mean/max/min):', B_recheck_lens.mean(), B_recheck_lens.max(), B_recheck_lens.min(), ',', nans, 'nans')

        # Angle between B and grad-B (perpendicular when the source is a line)
        print('  B perpendicular to gradB (B . gradB ~ 0)')
        B_dot_gradB = vect_dot(B_vecs, gradB_vecs)
        print('    B . gradB (max/min):', B_dot_gradB.max(), B_dot_gradB.min())
        # Angle between (B . gradB) / (|B| * |gradB|)
        B_gradB_angle = B_dot_gradB / (vect_lens(B_vecs) * vect_lens(gradB_vecs))
        B_gradB_angle *= 180 / numpy.pi
        print('    Angle between B and gradB (max/min deg):', B_gradB_angle.max(), B_gradB_angle.min())

    # Source line and base target point
    plot_source(ax, src_lines)
    ax.scatter(*base_pt, **TARGET_FMT)

    # Identify failed points
    pts_fails = numpy.full_like(pts, numpy.nan)
    numpy.copyto(pts_fails, pts, where=numpy.isnan(B_vecs))
    ax.scatter(*pts_fails.transpose(), **FAIL_FMT)

    pts = pts.transpose()
    B_vecs = B_vecs.transpose()
    gradB_vecs = gradB_vecs.transpose()
    ax.quiver(*pts, *B_vecs, **B_FMT)
    ax.quiver(*pts, *gradB_vecs, **gradB_FMT)
    ax.plot_surface(*pts, cmap='viridis', edgecolor='none', alpha=.75)

    ax.set_title('Equipotential surface')
    pyplot.show()
    return 0

if __name__ == '__main__':
    src_lines = numpy.array(SOURCE_POLYLINE)
    base_pt = numpy.array(BASE_POINT)
    exit(main(base_pt, src_lines))
