'''Electromagnetic induction model visualization

Tested on python 3.8.3. Requires: numpy, matplotlib, emi_calc
'''
import sys
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.widgets as widgets
import mpl_toolkits.mplot3d as mplot3d
import emi_calc
import coils

# Source current flow, see coils.gen_coil() arguments: (rad0, rad1, turns, len, separate, segs, center)
GEN_COIL_PARAMS = (1., 1., 2, .04, False, 4)
COIL_PARAM_CHANGE_XFORMS = (None, None, (1,0), (1,0))   # Zero based scale transform on turns and len

SOURCE_POLYLINE = coils.gen_coil(*GEN_COIL_PARAMS)

# Points to calculate induction vectors
TGT_YZ_POS = 0, 0
TGT_ROW_STEPS = 8
TARGET_POINTS = numpy.linspace([-1, *TGT_YZ_POS], [1, *TGT_YZ_POS], num=TGT_ROW_STEPS, endpoint=True)

# Slider origin/direction parameters
SOURCE_SLIDER_ORG = [0, 0, 0]
TARGET_SLIDER_DIR = [0, 0, 1]
SOURCE_SLIDER_DIR = [1, 1, 1]

FIELD_SCALE = .2
EMF_SCALE = .2

AX_MARGIN = .02
AX_BTN_HEIGHT = .2
AX_BTN_WIDTH = .25
AX_TEXT_WIDTH = .08
AX_NUM_SLIDERS = 2

SOURCE_FMT = dict(color='green', label='Source')
TARGET_FMT = dict(color='blue', marker='+', label='Target')
B_FMT = dict(color='magenta', linestyle='--', label='Field', visible=False)
GRADB_FMT = dict(color='yellow', linestyle=':', label='Field gradient', visible=False)
DR_DI_FMT = dict(color='orange', linestyle=':', label='I.dr/dI', visible=True)
EMF_FMT = dict(color='red', linestyle='-.', label='EM Force')

# From https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = numpy.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = numpy.mean(limits, axis=1)
    radius = 0.5 * numpy.max(numpy.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def plot_source(ax, src_lines):
    src_pts = src_lines[...,:-1,:]
    src_dirs = src_lines[...,1:,:] - src_lines[...,:-1,:]
    return ax.quiver(*src_pts.transpose(), *src_dirs.transpose(), **SOURCE_FMT)

def replace_collection(colls, key, new):
    """Replace collection in a dictionary by keeping its visibility"""
    old = colls.get(key)
    if old is not None:
        visible = old.get_visible()
        old.remove()
        new.set_visible(visible)
    colls[key] = new

def rotate_points(pts, axis, angle):
    # Shift the axes, to move the 'axis' to Z
    axis += 1
    pts = numpy.roll(pts, pts.shape[-1] - axis, axis=-1)

    # Perform 2D rotation on (pt[0], pt[1])
    x = pts[...,0] * numpy.cos(angle) - pts[...,1] * numpy.sin(angle)
    y = pts[...,0] * numpy.sin(angle) + pts[...,1] * numpy.cos(angle)
    pts[...,0], pts[...,1] = x, y

    # Shift-back axes
    return numpy.roll(pts, axis, axis=-1)

def deflate_rect(rect, hor_margin=AX_MARGIN, vert_margin=AX_MARGIN):
    """Deflate matplotlib rectangle [left, bottom, right, top]"""
    rect[0] += hor_margin
    rect[1] += vert_margin
    rect[2] -= 2 * hor_margin
    rect[3] -= 2 * vert_margin
    return rect

class main_data:
    src_lines = None
    tgt_pts = None
    # Matplotlib collections
    colls = {}

    def __init__(self, ax):
        self.ax = ax

    def redraw(self, src_lines, tgt_pts, coef=1):
        # EMI source lines
        if src_lines is None:
            src_lines = self.src_lines
        else:
            replace_collection(self.colls, 'src_coll', plot_source(self.ax, src_lines))
            self.src_lines = src_lines

        # Target points
        if tgt_pts is None:
            tgt_pts = self.tgt_pts
        else:
            replace_collection(self.colls, 'tgt_coll', self.ax.scatter(*tgt_pts.transpose(), **TARGET_FMT))
            self.tgt_pts = tgt_pts

        # Calculate EMI parameters B and dB for each target point
        emi_params = emi_calc.calc_all_emis(tgt_pts, src_lines, coef)
        if True:
            print("Calculated EMI parameters for %d target points:"%tgt_pts[...,0].size)
            pts = emi_params['pt']
            B_vecs = emi_params['B']
            gradB_vecs = emi_params['gradB']
            dr_dI_vecs = emi_params['dr_dI']
            emf_vecs = numpy.cross(dr_dI_vecs, B_vecs)
            print("\t[x, y, z]: B-len / gradB-len [x, y, z] / dr_dI-len / emf-len")
            print("\t------------------------------------------------------------")
            for pt, bl, gb, gbl, drl, emfl in zip(
                    pts,
                    numpy.sqrt((B_vecs * B_vecs).sum(-1)),
                    gradB_vecs,
                    numpy.sqrt((gradB_vecs * gradB_vecs).sum(-1)),
                    numpy.sqrt((dr_dI_vecs * dr_dI_vecs).sum(-1)),
                    numpy.sqrt((emf_vecs * emf_vecs).sum(-1)),
                    ):
                print('\t[%+.3f, %+.3f, %+.3f]: %.3f / %.3f [%+.3f, %+.3f, %+.3f] / %.3f / %.3f'%(
                        *pt, bl, gbl, *gb, drl, emfl))

        pts = emi_params['pt'].transpose()

        # Magnetic field
        if B_FMT:
            B = emi_params['B'].dot(FIELD_SCALE).transpose()
            replace_collection(self.colls, 'B_coll', self.ax.quiver(*pts, *B, **B_FMT))

        # The field-gradient, i.e. the direction of increase of the field magnitude
        if GRADB_FMT:
            dB = emi_params['gradB'].dot(FIELD_SCALE).transpose()
            replace_collection(self.colls, 'gradB_coll', self.ax.quiver(*pts, *dB, **GRADB_FMT))

        # The field-gradient, i.e. the "movement" of the field because of current increase
        if DR_DI_FMT:
            dr_dI = emi_params['dr_dI'].dot(FIELD_SCALE).transpose()
            replace_collection(self.colls, 'dr_dI_coll', self.ax.quiver(*pts, *dr_dI, **DR_DI_FMT))

        # The EMF induced because of field change
        if EMF_FMT:
            emf = numpy.cross(emi_params['dr_dI'], emi_params['B'])
            emf = emf.dot(EMF_SCALE).transpose()
            replace_collection(self.colls, 'emf_coll', self.ax.quiver(*pts, *emf, **EMF_FMT))
        return True

    def get_collections(self):
        return self.colls.values()

class on_clicked:
    def __init__(self, data):
        self.data = data

    def __call__(self, label):
        for coll in self.data.get_collections():
            if coll.get_label() == label:
                coll.set_visible(not coll.get_visible())
                break
        pyplot.draw()

class src_changed:
    def __init__(self, data, func):
        self.data = data
        self.func = func

    def __call__(self, pos):
        if self.func is None:
            self.data.redraw(None, None, pos)
        else:
            pts = numpy.array(SOURCE_POLYLINE)
            pts = self.func(pts, pos)
            self.data.redraw(pts, None)
        pyplot.draw()

class tgt_changed:
    def __init__(self, data, func):
        self.data = data
        self.func = func

    def __call__(self, pos):
        pts = numpy.array(TARGET_POINTS)
        pts = self.func(pts, pos)
        self.data.redraw(None, pts)
        pyplot.draw()

class move_changed:
    def __init__(self, move_dir):
        self.move_dir = numpy.array(move_dir)

    def __call__(self, pts, pos):
        return pts + self.move_dir * pos

class rotate_changed:
    def __init__(self, rot_axis, origin):
        self.origin = numpy.array(origin)
        self.rot_axis = rot_axis

    def __call__(self, pts, pos):
        return rotate_points(pts - self.origin, self.rot_axis, pos) + self.origin

class scale_changed:
    def __init__(self, scale_dir, origin):
        self.scale_dir = numpy.array(scale_dir)
        self.origin = numpy.array(origin)

    def __call__(self, pts, pos):
        pos = numpy.ones(3) + self.scale_dir * (pos - 1)
        return (pts - self.origin) * pos + self.origin

class coil_param_changed:
    def __init__(self, args, xforms):
        self.args = [*args]
        self.xforms = xforms

    def __call__(self, pts, pos):
        args = self.args.copy()
        for idx, xform in enumerate(self.xforms):
            if xform is not None:
                if xform[0]:
                    # Scale
                    args[idx] = xform[0] * (args[idx] - xform[1]) * pos  + xform[1]
                else:
                    # Translate
                    args[idx] += xform[1] * pos
                # The values specified as int, will be kept integer (i.e. "2" vs. "2.")
                if isinstance(self.args[idx], int):
                    args[idx] = round(args[idx])
        print('Generated coil:',
                ', '.join('%s=%s'%(n,v) for n,v in
                    zip(('rad0', 'rad1', 'turns', 'len', 'separate', 'segs', 'center'), args)) )
        return coils.gen_coil(*args)

def main(argv):
    """Main entry"""
    fig = pyplot.figure()
    rect = [0, AX_BTN_HEIGHT, 1, 1 - AX_BTN_HEIGHT]
    ax = fig.add_axes(deflate_rect(rect), projection='3d', adjustable='box')

    # Initial drawing
    data = main_data(ax)
    data.redraw(numpy.array(SOURCE_POLYLINE), numpy.array(TARGET_POINTS))

    set_axes_equal(ax)

    # Check boxes to show/hide individual elements
    rect = [0, 0, AX_BTN_WIDTH, AX_BTN_HEIGHT]
    rax = fig.add_axes(deflate_rect(rect))
    colls = data.get_collections()
    labels = [coll.get_label() for coll in colls]
    visibility = [coll.get_visible() for coll in colls]
    check = widgets.CheckButtons(rax, labels, visibility)
    check.on_clicked(on_clicked(data))

    # Slider to scale source lines (slider 1)
    rect = [AX_BTN_WIDTH, 1 * AX_BTN_HEIGHT / AX_NUM_SLIDERS,
            1 - AX_BTN_WIDTH, AX_BTN_HEIGHT / AX_NUM_SLIDERS]
    rax = pyplot.axes(deflate_rect(rect, AX_TEXT_WIDTH + AX_MARGIN))
    src_slider = widgets.Slider(rax, data.colls['src_coll'].get_label(), 0, 2, 1)
    src_slider.on_changed(src_changed(data, coil_param_changed(GEN_COIL_PARAMS, COIL_PARAM_CHANGE_XFORMS)))

    # Slider to move target points (slider 2)
    rect = [AX_BTN_WIDTH, 0 * AX_BTN_HEIGHT / AX_NUM_SLIDERS,
            1 - AX_BTN_WIDTH, AX_BTN_HEIGHT / AX_NUM_SLIDERS]
    rax = pyplot.axes(deflate_rect(rect, AX_TEXT_WIDTH + AX_MARGIN))
    tgt_slider = widgets.Slider(rax, data.colls['tgt_coll'].get_label(), -2, 2, 0)
    tgt_slider.on_changed(tgt_changed(data, move_changed(TARGET_SLIDER_DIR)))


    ax.legend()
    pyplot.show()
    return 0

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
