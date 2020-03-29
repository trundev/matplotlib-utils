'''Build surface from a vector field containing normals

Reveal a surface via specified point by traversing vector field. The vector field
must be made of normals of concentric surfaces.
'''
import sys
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.widgets as widgets
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
# Approximation precision
APPROX_TOLERANCE = 1e-3
APPROX_MAX_ITERS = 3
APPROX_ANGLE = 15 * (numpy.pi / 180)    # 1/6 of 90 degrees
APPROX_ANGLE_COS = numpy.cos(APPROX_ANGLE)
APPROX_ANGLE_SIN = numpy.sin(APPROX_ANGLE)

SOURCE_FMT  = dict(color='green', label='Source')
BASE_PT_FMT = dict(color='black', marker='o', label='Base point')
TANGENTS_FMT = dict(color='magenta', linestyle='--', label='Tangents')
NORMALS_FMT = dict(color='blue', linestyle='-', label='Normals')
POINT_FMT   = dict(color='magenta', marker='.', visible=False, label='Points')
SURFACE_FMT = dict(cmap='viridis', edgecolor='none', alpha=.5, label='_Surface')    # Strange crash w/o leading underscore

AX_MARGIN = .01
SLIDER_HOR_MARGIN = .1
BASE_AX_HEIGHT = .2
BASE_AX_WIDTH = 1
AX_BTN_HEIGHT = .2
AX_BTN_WIDTH = .2

#
# Utilities
# The functions support numpy array of vectors with the x,y,z in the last dimension
#
def vect_dot(v0, v1):
    """Dot product of vectors by vectors in a numpy arrays"""
    return (v0 * v1).sum(-1)

def vect_lens(v):
    """Length of vectors in a numpy array"""
    return numpy.sqrt(vect_dot(v, v))

def vect_scale(vects, scale):
    """Dot product of vectors and scalars in a numpy arrays"""
    return vects * scale[..., numpy.newaxis]

def unit_vects(vects):
    """Get unit vectors in a numpy array"""
    return vect_scale(vects, 1 / vect_lens(vects))

def intersect_vects(vects, points0, points1):
    """Intersection between vector and line between two points"""
    if True:    # Optimization
        # Optimize out the cross product and square root (meaningful in 3D scenarios only):
        # Scale = |AxB|^2 / [(AxB).(AxC) + (AxB).(CxB)] =
        # = (|A|^2|B|^2 - |A.B|^2) / (A - B).[A(B.C) - B(A.C)] =
        # = (|A|^2|B|^2 - |A.B|^2) / [(|A|^2 - A.B)(B.C) - (A.B - |B|^2)(A.C)]
        a_2 = vect_dot(points0, points0)
        b_2 = vect_dot(points1, points1)
        a_dot_b = vect_dot(points0, points1)
        a_dot_c = vect_dot(points0, vects)
        b_dot_c = vect_dot(points1, vects)
        scale = a_2 * b_2 - a_dot_b * a_dot_b
        scale /= (a_2 - a_dot_b) * b_dot_c - (a_dot_b - b_2) * a_dot_c
    else:
        # Scale = |AxB| / (|AxC + CxB|)
        def _len(v):
            # With 2D vectors numpy.cross() returns scalar
            return vect_lens(v) if v.shape else v
        scale = _len(numpy.cross(points0, points1)) / _len(numpy.cross(points0, vects) + numpy.cross(vects, points1))
    res = vect_scale(vects, scale)
    # Ensure result is 'inf' when the lines are parallel (scale is +infinite)
    mask = numpy.broadcast_to(numpy.isposinf(scale)[..., numpy.newaxis], res.shape)
    numpy.place(res, mask, numpy.inf)
    return res

def plot_source(ax, src_lines):
    src_pts = src_lines[...,:-1,:]
    src_dirs = src_lines[...,1:,:] - src_lines[...,:-1,:]
    return ax.quiver(*src_pts.T, *src_dirs.T, **SOURCE_FMT)

#
# Surface generation
#
def adjust_points(pt, v, pt0, v0):
    """Adjust pt in order both points to be equidistant from the intersection of vectors"""
    rel_pt = pt0 - pt
    v = intersect_vects(v, rel_pt, v0 + rel_pt)
    #
    # The intersection point at "pt + v" (or "pt0 + v - rel_pt") is the
    # center of rotation of "pt0" toward "pt".
    #
    v0 = v - rel_pt
    if False:
        # Regular adjustment: move "pt" toward intersection
        #
        v_adj = vect_scale(v, 1 - vect_lens(v0) / vect_lens(v))
    else:
        # Extra adjustment: move "pt" in order the rotation to be at fixed angle
        #
        # Vector with the length of "v0", but pependicular to it toward "rel_pt"
        perp = numpy.cross(v0, numpy.cross(v0, rel_pt))
        perp = vect_scale(unit_vects(perp), vect_lens(v0))
        # Rotate "v0" at APPROX_ANGLE around the center
        rot_v0 = perp * APPROX_ANGLE_SIN - v0 * APPROX_ANGLE_COS
        # Here vect_lens(rot_v0) == vect_lens(v0)
        v_adj = rot_v0 + v

    # No adjustment when the lines are parallel
    numpy.place(v_adj, numpy.isposinf(v), 0)

    # Check for negligible adjustments
    adj_max2 = vect_dot(v_adj, v_adj).max()
    if adj_max2 < APPROX_TOLERANCE ** 2:
        v_adj = None
    return v_adj

def surface_from_normals(extent_uv, normal_fn, base_pt, *params):
    """Generate surface from its normals returned by an arbitrary function"""
    def wrap_normal_fn(out, pt):
        normal, tangent = normal_fn(pt, *params)
        out['pt'] = pt
        # Ensure tangent is perpendicular to normal and both are unit vectors
        out['norm'] = unit_vects(normal)
        out['tang'] = unit_vects(numpy.cross(numpy.cross(normal, tangent), normal))
        return

    data_dtype=[
        ('pt', (base_pt.dtype, base_pt.shape[-1])),
        ('norm', (base_pt.dtype, base_pt.shape[-1])),
        ('tang', (base_pt.dtype, base_pt.shape[-1])),
        ]
    surface = numpy.empty((1, 1), dtype=data_dtype)

    wrap_normal_fn(surface[0, 0], base_pt)
    # Set initial tangent length
    surface[0, 0]['tang'] *= STEP_SCALE

    for extent in range(max(extent_uv)):
        # Select in which directions to expand
        if extent < extent_uv[0] and extent < extent_uv[1]:
            axis_idxs = [(0, -1), (1, -1), (0, 0), (1, 0)]
        elif extent < extent_uv[0]:
            axis_idxs = [(0, -1), (0, 0)]
        else:
            axis_idxs = [(1, -1), (1, 0)]

        for axis, idx in axis_idxs:
            # Temporarily move axis to be processed at front
            surface = numpy.moveaxis(surface, axis, 0)

            new_pts = numpy.full(surface.shape[1:], numpy.nan, dtype=surface.dtype)

            base_pts = surface[idx]
            tangents = base_pts['tang']
            if axis > 0:
                # Use bi-tangent (the 'norm' is unit vector perpendicular to 'tang')
                tangents = numpy.cross(base_pts['norm'], tangents)
            if idx < 0:
                tangents = -tangents

            # Iterations to increase the approximation precision
            pts = base_pts['pt'].copy()
            pts_adj = tangents
            for _ in range(APPROX_MAX_ITERS):
                pts += pts_adj
                wrap_normal_fn(new_pts, pts)
                pts_adj = adjust_points(pts, new_pts['norm'], base_pts['pt'], base_pts['norm'])
                # Abort approximation when all adjustments are negligible
                if pts_adj is None:
                    break
            if pts_adj is not None:
                print('Warning: Approximation failed at extent:', extent, ', dir:', axis, idx, ', pts_adj:', pts_adj)

            # The next step will be toward 'tang', but at same distance as that one
            new_pts['tang'] = vect_scale(new_pts['tang'], vect_lens(pts - base_pts['pt']))

            new_pts = numpy.expand_dims(new_pts, 0)
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
def plot_surface(ax, extent_uv, extent, normal_fn, base_pt, *params):
    """Plot specific surface"""
    colls = []

    # Base target point
    if BASE_PT_FMT:
        colls.append( ax.scatter(*base_pt, **BASE_PT_FMT))

    surface = surface_from_normals(extent_uv, normal_fn, base_pt, *params)

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
        colls.append( ax.quiver(*surface.T, *normals.T, **NORMALS_FMT))
    if TANGENTS_FMT:
        colls.append( ax.quiver(*surface.T, *tangents.T, **TANGENTS_FMT))
    if POINT_FMT:
        colls.append( ax.scatter(*surface.T, **POINT_FMT))

    if SURFACE_FMT:
        colls.append( ax.plot_surface(*surface.T, **SURFACE_FMT))
    return colls

#
# Widget utilities
#
def deflate_rect(rect, hor_margin=AX_MARGIN, vert_margin=AX_MARGIN):
    """Deflate matplotlib rectangle [left, bottom, right, top]"""
    rect[0] += hor_margin
    rect[1] += vert_margin
    rect[2] -= 2 * hor_margin
    rect[3] -= 2 * vert_margin
    return rect

def split_rect(rect, split_pos=.5, hor_split=False):
    """Split matplotlib rectangle [left, bottom, right, top]"""
    rect1 = rect.copy()
    rect2 = rect.copy()
    if hor_split:
        rect1[3] = split_pos
        rect2[3] -= split_pos
        rect2[1] += rect1[3]
    else:
        rect1[2] = split_pos
        rect2[2] -= split_pos
        rect2[0] += rect1[2]
    return rect1, rect2

class main_class:
    def __init__(self, ax, colls, plot_cb, *plot_params):
        self.ax = ax
        self.colls = colls
        self.plot_cb = plot_cb
        self.plot_params = plot_params
        self.extent_uv = [1, 1]

    def find_collection(self, label):
        for idx, coll in enumerate(self.colls):
            if coll.get_label() == label:
                return coll, idx
        return None, None

    def replace_collection(self, coll):
        """Replace collection by keeping its visibility"""
        old, idx = self.find_collection(coll.get_label())
        if old:
            visible = old.get_visible()
            old.remove()
            coll.set_visible(visible)
            self.colls[idx] = coll
        else:
            self.colls.append(coll)

    def replace_collections(self, colls):
        for coll in colls:
            self.replace_collection(coll)

    def do_redraw(self):
        colls = self.plot_cb(self.ax, self.extent_uv, *self.plot_params)
        self.replace_collections(colls)
        pyplot.draw()

    def do_showhide(self, label):
        coll, _ = self.find_collection(label)
        if coll:
            coll.set_visible(not coll.get_visible())
            self.ax.legend()
            pyplot.draw()

    def slider_u_changed(self, val):
        self.extent_uv[0] = int(val)
        self.do_redraw()

    def slider_v_changed(self, val):
        self.extent_uv[1] = int(val)
        self.do_redraw()

def emi_gradients(pts, src_lines):
    emi_params = emi_calc.calc_all_emis(pts, src_lines)
    return emi_params['gradB'], emi_params['B']

def main(base_pt, src_lines):
    """Main entry"""
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')

    colls = []

    # Current source line
    if SOURCE_FMT:
        colls.append( plot_source(ax, src_lines))

    extent = vect_lens(src_lines.max(0) - src_lines.min(0))
    main_data = main_class(ax, colls, plot_surface, extent, emi_gradients, base_pt, src_lines)
    main_data.do_redraw()

    # Base widget rectangle
    rect1, rect2 = split_rect([0, 0, BASE_AX_WIDTH, BASE_AX_HEIGHT], AX_BTN_WIDTH)

    # Check boxes to show/hide individual elements
    rax = fig.add_axes(deflate_rect(rect1))
    labels = [coll.get_label() for coll in colls]
    visibility = [coll.get_visible() for coll in colls]
    check = widgets.CheckButtons(rax, labels, visibility)
    check.on_clicked(main_data.do_showhide)

    # Sliders
    _, rect21 = split_rect(rect2, AX_BTN_HEIGHT / 3)
    rect214, rect213 = split_rect(rect21, AX_BTN_HEIGHT / 4, True)
    rect213, _ = split_rect(rect213, AX_BTN_HEIGHT / 4, True)
    rax = fig.add_axes(deflate_rect(rect213, SLIDER_HOR_MARGIN), fc='lightgray')
    slider_u = widgets.Slider(rax, 'U extent', 0, 10, main_data.extent_uv[0],
            dragging=False, valstep=1)
    slider_u.on_changed(main_data.slider_u_changed)
    rax = fig.add_axes(deflate_rect(rect214, SLIDER_HOR_MARGIN), fc='lightgray')
    slider_v = widgets.Slider(rax, 'V extent', 0, 8, main_data.extent_uv[1],
            dragging=False, valstep=1)
    slider_v.on_changed(main_data.slider_v_changed)

    ax.set_title('Surface from normals')
    ax.legend()
    pyplot.show()
    return 0

if __name__ == '__main__':
    src_lines = numpy.array(SOURCE_POLYLINE)
    base_pt = numpy.array(BASE_POINT)
    exit(main(base_pt, src_lines))
