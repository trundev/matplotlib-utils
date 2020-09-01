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
import coils

#
# EM source current flow
#
# Helmholtz coils (two parallel circles at distance of R)
rad = 1.
SOURCE_POLYLINE = coils.helmholtz_coil(rad)

#
# Initial surface point
#
SEED_POINT = [2*rad, 0, 0]
del rad

INITIAL_STEP_SCALE = .25
# Approximation precision
APPROX_TOLERANCE = 1e-3
APPROX_MAX_ITERS = 3
APPROX_ANGLE = 15 * (numpy.pi / 180)    # 1/6 of 90 degrees
APPROX_ANGLE_COS = numpy.cos(APPROX_ANGLE)
APPROX_ANGLE_SIN = numpy.sin(APPROX_ANGLE)

SOURCE_FMT  = {'color': 'green', 'label': 'Source'}
SEED_PT_FMT = {'color': 'black', 'marker': 'o', 'label': 'Seed point'}
TANGENTS_FMT = {'color': 'magenta', 'linestyle': '--', 'label': 'Tangents'}
NORMALS_FMT = {'color': 'blue', 'linestyle': '-', 'label': 'Normals'}
POINT_FMT   = {'color': 'magenta', 'marker': '.', 'visible': False, 'label': 'Points'}
# Note: Axes3D.plot_surface() crashes when label does start with underscore!?!
SURFACE_FMT = {'cmap': 'viridis', 'color':'greenyellow', 'edgecolor': 'none', 'alpha': .5, 'label': '_Surface'}

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
def adjust_points(pt, v, v0):
    """Adjust pt in order both points to be equidistant from the intersection of vectors"""
    v0 = intersect_vects(v0, pt, v + pt)
    #
    # The intersection point at "v0" is the center of rotation of a point at
    # the origin toward "pt"".
    #
    if False:
        # Regular adjustment: move "pt" toward intersection, so as |v| == |v0|
        #
        v = v0 - pt
        pt_adj = v0 - vect_scale(v, vect_lens(v0) / vect_lens(v))
    else:
        # Extra adjustment: move "pt" in order the rotation to be at fixed angle
        #
        # Vector with the length of "v0", but pependicular to it toward "pt"
        perp = numpy.cross(numpy.cross(v0, pt), v0)
        perp = vect_scale(unit_vects(perp), vect_lens(v0))
        # Rotate "v0" at APPROX_ANGLE around the center
        rot_v0 = perp * APPROX_ANGLE_SIN - v0 * APPROX_ANGLE_COS
        # Here vect_lens(rot_v0) == vect_lens(v0)
        pt_adj = rot_v0 + v0

    # Fix the NaN values, when the lines are parallel
    numpy.place(pt_adj, numpy.isposinf(v0), numpy.inf)
    return pt_adj

def calc_adjustment(tangents, new_pts, base_pts, bitang_sl):
    """Adjust point by using normal and bi-tangent curvature"""
    tang_adj = adjust_points(tangents, new_pts['norm'], base_pts['norm'])
    tang_adj2 = adjust_points(tangents, new_pts['tangs'][bitang_sl], base_pts['tangs'][bitang_sl])

    # Select the point closer to the base-point
    mask = vect_dot(tang_adj, tang_adj) > vect_dot(tang_adj2, tang_adj2)
    mask = numpy.broadcast_to(mask[..., numpy.newaxis], tang_adj.shape)
    numpy.copyto(tang_adj, tang_adj2, where=mask)

    # No adjustment when both tangents are parallel
    numpy.place(tang_adj, numpy.isposinf(tang_adj), 0)

    # Check for negligible adjustments
    adj = tang_adj - tangents
    adj_max2 = vect_dot(adj, adj).max()
    if adj_max2 < APPROX_TOLERANCE ** 2:
        tang_adj = None
    return tang_adj

def surface_from_normals(extent_uv, normal_fn, seed_pt, *params, **kw_params):
    """Generate surface from its normals returned by an arbitrary function"""
    def wrap_normal_fn(out, pt):
        normal, tangent = normal_fn(pt, *params, **kw_params)
        out['pt'] = pt
        # Ensure both tangents are perpendicular to normal and all are unit vectors
        out['norm'] = unit_vects(normal)
        tangs = unit_vects(numpy.cross(numpy.cross(normal, tangent), normal))
        out['tangs'][..., 0, :] = tangs
        out['tangs'][..., 1, :] = numpy.cross(out['norm'], tangs)
        return

    # Internal context for each surface point
    data_dtype=[
        # The surface point
        ('pt', (seed_pt.dtype, seed_pt.shape[-1])),
        # Normal vector to surface
        ('norm', (seed_pt.dtype, seed_pt.shape[-1])),
        # Tangent and bi-tangent
        ('tangs', (seed_pt.dtype, (2, seed_pt.shape[-1]))),
        ]
    surface = numpy.empty((1, 1), dtype=data_dtype)

    wrap_normal_fn(surface[0, 0], seed_pt)
    # Set initial tangent length
    surface[0, 0]['tangs'][0] *= INITIAL_STEP_SCALE
    surface[0, 0]['tangs'][1] *= INITIAL_STEP_SCALE

    for extent in range(max(extent_uv)):
        # Select in which directions to expand
        if extent < extent_uv[0] and extent < extent_uv[1]:
            axis_idxs = [(0, -1), (1, -1), (0, 0), (1, 0)]
        elif extent < extent_uv[0]:
            axis_idxs = [(0, -1), (0, 0)]
        else:
            axis_idxs = [(1, -1), (1, 0)]

        for axis, idx in axis_idxs:
            # Duplicating the edge (first or last row/column) to expand the surface
            pad_width = numpy.zeros((surface.ndim, 2), dtype=int)
            pad_width[axis, idx] = 1
            surface = numpy.pad(surface, pad_width, 'edge')

            # Reference to the newly added edge
            sl = surface.ndim * [slice(None)]
            sl[axis] = idx
            new_pts = surface[tuple(sl)]

            # Reference to the original (base) edge -- 1 or -2
            sl[axis] = idx + (-1 if idx < 0 else 1)
            base_pts = surface[tuple(sl)]

            # Select the tangents and bi-tangents for the particular axis
            tang_sl = numpy.s_[..., axis, :]
            bitang_sl = numpy.s_[...,  1 - axis, :]
            tangents = base_pts['tangs'][tang_sl]
            if idx < 0:
                tangents = -tangents

            # Iterations to increase the approximation precision
            for _ in range(APPROX_MAX_ITERS):
                wrap_normal_fn(new_pts, base_pts['pt'] + tangents)
                tangents = calc_adjustment(tangents, new_pts, base_pts, bitang_sl)
                # Abort approximation when all adjustments are negligible
                if tangents is None:
                    break
            if tangents is not None:
                print('Warning: Approximation failed at extent:', extent, ', dir:', axis, idx, ', tangents:', tangents)

            # The next step toward current tangent will be at the distance and the direction of that one
            new_tangs = new_pts['tangs']
            this_step = new_pts['pt'] - base_pts['pt']
            if idx < 0:
                this_step = -this_step
            new_tangs[tang_sl] = vect_scale(new_tangs[tang_sl], numpy.negative(
                    # Preserve distance
                    vect_lens(this_step),
                    # Peserve direction
                    where=vect_dot(this_step, new_tangs[tang_sl]) < 0
                ))

            # The next step toward the "other" tangent will be at the distance from the base point
            new_tangs[bitang_sl] = vect_scale(new_tangs[bitang_sl], vect_lens(base_pts['tangs'][bitang_sl]))

    return surface['pt']

#
# Plot
#
def plot_geometry(ax, seed_pt, extent_uv, scale, normal_fn, *params, **kw_params):
    """Plot specific surface"""
    print('')
    print('Generating surface seed %s, extents %d/%d'%(seed_pt, *extent_uv))

    colls = []

    # The seed target point
    if SEED_PT_FMT:
        colls.append( ax.scatter(*seed_pt, **SEED_PT_FMT))

    surface = surface_from_normals(extent_uv, normal_fn, seed_pt, *params, **kw_params)

    # Re-obtain selected points
    normals, tangents = normal_fn(surface, *params, **kw_params)
    n_lens = vect_lens(normals)
    t_lens = vect_lens(tangents)
    print('Surface %s'%(surface.shape[:-1],),
          ',', numpy.count_nonzero(numpy.isnan(surface.sum(-1))), 'NaNs')
    print('Normals (mean/max/min)', numpy.nanmean(n_lens), numpy.nanmax(n_lens), numpy.nanmin(n_lens),
          ',', numpy.count_nonzero(numpy.isnan(n_lens)), 'NaNs')
    print('Tangents (mean/max/min)', numpy.nanmean(t_lens), numpy.nanmax(t_lens), numpy.nanmin(t_lens),
          ',', numpy.count_nonzero(numpy.isnan(t_lens)), 'NaNs')

    # Visualize selected normals
    # Resize vectors to fit screen (ignore NaN-s)
    normals *= scale / numpy.nanmax(n_lens) / 8
    tangents *= scale / numpy.nanmax(t_lens) / 8
    if NORMALS_FMT:
        colls.append( ax.quiver(*surface.T, *normals.T, **NORMALS_FMT))
    if TANGENTS_FMT:
        colls.append( ax.quiver(*surface.T, *tangents.T, **TANGENTS_FMT))
    if POINT_FMT:
        colls.append( ax.scatter(*surface.T, **POINT_FMT))

    if SURFACE_FMT:
        # Plot surface or polyline
        if min(surface.shape[:2]) > 1:
            colls.append( ax.plot_surface(*surface.T, **SURFACE_FMT))
        else:
            fmt = SURFACE_FMT.copy()
            del fmt['edgecolor'], fmt['cmap']
            pts = surface.reshape(-1, surface.shape[-1])
            colls.append( ax.plot(*pts.T, **fmt)[0])
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

#
# Main user interaction context
#
class main_class:
    def __init__(self, ax, seed_pt):
        self.ax = ax
        self.colls = []
        self.seed_pt = seed_pt
        self.plot_cb = None
        self.plot_params = None
        self.plot_kw_params = None
        self.extent_uv = [1, 1]

    def on_redraw(self, plot_cb, *plot_params, **plot_kw_params):
        self.plot_cb = plot_cb
        self.plot_params = plot_params
        self.plot_kw_params = plot_kw_params

    def _find_collection(self, label):
        for idx, coll in enumerate(self.colls):
            if coll.get_label() is label:
                return coll, idx
        return None, None

    def update_collection(self, coll):
        """Replace collection by keeping its visibility"""
        old, idx = self._find_collection(coll.get_label())
        if old:
            visible = old.get_visible()
            old.remove()
            coll.set_visible(visible)
            self.colls[idx] = coll
        else:
            self.colls.append(coll)

    def update_collections(self, colls):
        for coll in colls:
            self.update_collection(coll)

    def get_collections(self):
        return self.colls

    def do_redraw(self):
        colls = self.plot_cb(self.ax, self.seed_pt, self.extent_uv, *self.plot_params, **self.plot_kw_params)
        self.update_collections(colls)
        self.ax.figure.canvas.draw_idle()

    def do_showhide(self, label):
        coll, _ = self._find_collection(label)
        if coll:
            coll.set_visible(not coll.get_visible())
            self.ax.legend()
            self.ax.figure.canvas.draw_idle()

    def slider_u_changed(self, val):
        self.extent_uv[0] = int(val)
        self.do_redraw()

    def slider_v_changed(self, val):
        self.extent_uv[1] = int(val)
        self.do_redraw()

def emi_gradients(pts, src_lines):
    emi_params = emi_calc.calc_all_emis(pts, src_lines)
    return emi_params['gradB'], emi_params['B']

def main(seed_pt, src_lines):
    """Main entry"""
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')

    flat_src_lines = src_lines.reshape((-1, src_lines.shape[-1]))
    scale = vect_lens(flat_src_lines.max(0) - flat_src_lines.min(0))
    main_data = main_class(ax, seed_pt)
    main_data.on_redraw(plot_geometry, scale, emi_gradients, src_lines)

    # Current source line
    if SOURCE_FMT:
        main_data.update_collection( plot_source(ax, src_lines))

    main_data.do_redraw()

    # Base widget rectangle
    rect1, rect2 = split_rect([0, 0, BASE_AX_WIDTH, BASE_AX_HEIGHT], AX_BTN_WIDTH)

    # Check boxes to show/hide individual elements
    rax = fig.add_axes(deflate_rect(rect1))
    labels = []
    visibility = []
    for coll in main_data.get_collections():
        labels.append(coll.get_label())
        visibility.append(coll.get_visible())
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
    slider_v = widgets.Slider(rax, 'V extent', 0, 10, main_data.extent_uv[1],
            dragging=False, valstep=1)
    slider_v.on_changed(main_data.slider_v_changed)

    ax.set_title('Surface from normals')
    ax.legend()
    pyplot.show()
    return 0

if __name__ == '__main__':
    src_lines = numpy.array(SOURCE_POLYLINE)
    seed_pt = numpy.array(SEED_POINT)
    exit(main(seed_pt, src_lines))
