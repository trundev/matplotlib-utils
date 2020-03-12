'''Build surface from a vector field containing normals

Reveal a surface via specified point by traversing vector field. The vector field
must be made of normals of concentric surfaces.
'''
import sys
import numpy
import matplotlib.pyplot as pyplot
import mpl_toolkits.mplot3d as mplot3d
import emi_calc

#
# EM source current flow
#
SRC_Z_STEP = 0  # 0.01
SOURCE_POLYLINE = [
   [0., 0., 0 * SRC_Z_STEP],
   [1., 0., 1 * SRC_Z_STEP],    # right
#   [1., 1., 2 * SRC_Z_STEP],    # up
#   [0., 1., 3 * SRC_Z_STEP],    # left
#   [0., 0., 4 * SRC_Z_STEP],    # down
]

BASE_POINT = [.5, -1, .0]

STEP_SCALE = .25

SOURCE_FMT  = dict(color='green', label='Source')
BASE_PT_FMT = dict(color='black', marker='o', label='Base point')
TANGENTS_FMT = dict(color='magenta', linestyle='--', label='Tangents')
NORMALS_FMT = dict(color='blue', linestyle='-', label='Normals')
POINT_FMT   = {}    #dict(color='magenta', marker='.', label='Points')
SURFACE_FMT = dict(cmap='viridis', edgecolor='none', alpha=.5)

#
# Utilities
#
def vect_dot(v0, v1):
    """Dot product of vectors from a numpy array (x,y,z in the last dimension)"""
    return (v0 * v1).sum(-1)

def vect_lens(v):
    """Length of vectors in a numpy array (x,y,z in the last dimension)"""
    return numpy.sqrt(vect_dot(v, v))
    
def make_unit_vects(vects):
    """Make lengths of vector in a numpy array equal to 1 (x,y,z in the last dimension)"""
    vects.T[:] /= vect_lens(vects).T
    return vects

def plot_source(ax, src_lines):
    src_pts = src_lines[...,:-1,:]
    src_dirs = src_lines[...,1:,:] - src_lines[...,:-1,:]
    return ax.quiver(*src_pts.T, *src_dirs.T, **SOURCE_FMT)

#
# Surface generation
#
def surface_from_normals(normal_fn, base_pt, *params):
    """Generate surface from its normals returned by an arbitrary function"""
    def wrap_normal_fn(out, pt):
        normal, tangent = normal_fn(pt, *params)
        out['pt'] = pt
        # Ensure tangent is perpendicular to normal
        out['norm'] = make_unit_vects(normal)
        out['tang'] = make_unit_vects(numpy.cross(numpy.cross(normal, tangent), normal))
        return

    data_dtype=[
        ('pt', (base_pt.dtype, base_pt.shape[-1])),
        ('norm', (base_pt.dtype, base_pt.shape[-1])),
        ('tang', (base_pt.dtype, base_pt.shape[-1])),
        ]
    surface = numpy.empty((1, 1), dtype=data_dtype)

    wrap_normal_fn(surface[0, 0], base_pt)

    for iter in range(6):
        for axis, idx in (0, -1), (1, -1), (0, 0), (1, 0):
            # Temporarily move axis to be processed at front
            surface = numpy.moveaxis(surface, axis, 0)

            new_pts = numpy.full((1, *surface.shape[1:]), numpy.nan, dtype=surface.dtype)

            base_pts = surface[idx]
            tangents = base_pts['tang']
            if axis > 0:
                # Use bi-tangent
                tangents = make_unit_vects(numpy.cross(base_pts['norm'], tangents))
            if idx < 0:
                tangents = -tangents

            tangents *= STEP_SCALE  # TODO: Use proper step
            wrap_normal_fn(new_pts, base_pts['pt'] + tangents)

            if idx < 0:
                surface = numpy.concatenate((surface, new_pts))
            else:
                surface = numpy.concatenate((new_pts, surface))

            # Move back processed axis
            surface = numpy.moveaxis(surface, 0, axis)

    return surface['pt']

#
# Plot
#
def plot_surface(ax, extent, normal_fn, base_pt, *params):
    """Plot specific surface"""
    # Base target point
    if BASE_PT_FMT:
        ax.scatter(*base_pt, **BASE_PT_FMT)

    surface = surface_from_normals(normal_fn, base_pt, *params)

    # Re-obtain selected points
    normals, tangents = normal_fn(surface, *params)
    n_lens = vect_lens(normals)
    t_lens = vect_lens(tangents)
    print('Surface (%s)'%(surface.shape[:-1],),
          ',', numpy.count_nonzero(numpy.isnan(surface.sum(-1))), 'NaNs')
    print('Normals (mean/max/min)', numpy.nanmean(n_lens), numpy.nanmax(n_lens), numpy.nanmin(n_lens),
          ',', numpy.count_nonzero(numpy.isnan(n_lens)), 'NaNs')
    print('Tangents (mean/max/min)', numpy.nanmean(t_lens), numpy.nanmax(t_lens), numpy.nanmin(t_lens),
          ',', numpy.count_nonzero(numpy.isnan(t_lens)), 'NaNs')

    # Visualize selected normals
    # Resize vectors to fit screen (ignore NaN-s)
    normals *= extent / numpy.nanmax(n_lens) / 8
    tangents *= extent / numpy.nanmax(t_lens) / 8
    if NORMALS_FMT:
        ax.quiver(*surface.T, *normals.T, **NORMALS_FMT)
    if TANGENTS_FMT:
        ax.quiver(*surface.T, *tangents.T, **TANGENTS_FMT)
    if POINT_FMT:
        ax.scatter(*surface.T, **POINT_FMT)

    if SURFACE_FMT:
        ax.plot_surface(*surface.T, **SURFACE_FMT)
    return

def emi_gradients(pts, src_lines):
    emi_params = emi_calc.calc_all_emis(pts, src_lines)
    return emi_params['gradB'], emi_params['B']

def main(base_pt, src_lines):
    """Main entry"""
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')

    # Current source line
    if SOURCE_FMT:
        plot_source(ax, src_lines)

    extent = vect_lens(src_lines.max(0) - src_lines.min(0))
    plot_surface(ax, extent, emi_gradients, base_pt, src_lines)

    ax.set_title('Surface from normals')
    ax.legend()
    pyplot.show()
    return 0

if __name__ == '__main__':
    src_lines = numpy.array(SOURCE_POLYLINE)
    base_pt = numpy.array(BASE_POINT)
    exit(main(base_pt, src_lines))
