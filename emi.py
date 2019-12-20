'''Electromagnetic induction model visualization

Tested on python 3.7.5. Requires: numpy, matplotlib, emi_calc
'''
import sys
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.widgets as widgets
import mpl_toolkits.mplot3d as mplot3d
import emi_calc

# Source current flow
SRC_Z_STEP = 0.01
SOURCE_POLYLINE = [
   [0., 0., 0 * SRC_Z_STEP],
   [1., 0., 1 * SRC_Z_STEP],    # right
   [1., 2., 2 * SRC_Z_STEP],    # up
   [0., 2., 3 * SRC_Z_STEP],    # left
   [0., 0., 4 * SRC_Z_STEP],    # down
]
# Points to calculate induction vectors
TGT_Y_POS = 1
TGT_ROW_STEPS = 8
TARGET_POINTS = [
    # Top row
    *(
        [x, TGT_Y_POS, 1] for x in numpy.linspace(-.5, 1.5, num=TGT_ROW_STEPS, endpoint=False)
     ),
    # Right row
    *(
        [1.5, TGT_Y_POS, z] for z in numpy.linspace(1, -1, num=TGT_ROW_STEPS, endpoint=False)
     ),
    # Bottom row
    *(
        [x, TGT_Y_POS, -1] for x in numpy.linspace(1.5, -.5, num=TGT_ROW_STEPS, endpoint=False)
     ),
    # Left row
    *(
        [-.5, TGT_Y_POS, z] for z in numpy.linspace(-1, 1, num=TGT_ROW_STEPS, endpoint=False)
     ),
]
# Planar grid between [-.5, TGT_Y_POS, -1] and [1.5, TGT_Y_POS, 1]
#TARGET_POINTS = numpy.linspace(
#    numpy.linspace([-.5, TGT_Y_POS, -1], [1.5, TGT_Y_POS, -1], num=TGT_ROW_STEPS),
#    numpy.linspace([-.5, TGT_Y_POS,  1], [1.5, TGT_Y_POS,  1], num=TGT_ROW_STEPS),
#    num=TGT_ROW_STEPS
#).reshape((-1,3))

# Transformation parameters
TRANSFORM_ORG = [0.5, 1, 0]

FIELD_SCALE = .2
EMF_SCALE = .1

AX_MARGIN = .02
AX_BTN_HEIGHT = .3
AX_BTN_WIDTH = .25
AX_TEXT_WIDTH = .05
AX_NUM_SLIDERS = 3

LABEL_TRANSLATE = 'Translate'
LABEL_ROTATE = 'Rotate'
LABEL_SCALE = 'Scale'
TRANSLATE_PARAMS = [-2, 2, 0]
ROTATE_PARAMS = [-360, 360, 0]
SCALE_PARAMS = [0, 2, 1]

SOURCE_FMT = dict(color='green', label='Source')
TARGET_FMT = dict(color='blue', marker='+', label='Target')
B_FMT = dict(color='magenta', linestyle='--', label='Field')
dB_FMT = dict(color='yellow', linestyle=':', label='Field change')
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
    src_lines = src_lines.transpose()
    ##ax.plot(*src_lines.transpose(), **SOURCE_FMT)
    return ax.quiver(*src_lines[:,:-1], *(src_lines[:,1:] - src_lines[:,:-1]), **SOURCE_FMT)

def replace_collection(old, new):
    """Replace collection by keeping its visibility"""
    if old is not None:
        visible = old.get_visible()
        old.remove()
        new.set_visible(visible)
    return new

def deflate_rect(rect, hor_margin=AX_MARGIN, vert_margin=AX_MARGIN):
    """Deflate matplotlib rectangle [left, bottom, right, top]"""
    rect[0] += hor_margin
    rect[1] += vert_margin
    rect[2] -= 2 * hor_margin
    rect[3] -= 2 * vert_margin
    return rect

def rotate_point(pt, axis, pos):
    # Shift axes
    pt = numpy.array([*pt[axis + 1:], *pt][:pt.size])

    # Perform 2D rotation on (pt[0], pt[1])
    x = pt[0] * numpy.cos(pos) - pt[1] * numpy.sin(pos)
    y = pt[0] * numpy.sin(pos) + pt[1] * numpy.cos(pos)
    pt[0], pt[1] = x, y

    # Shift-back axes
    pt = numpy.array([*pt[pt.size - axis - 1:], *pt][:pt.size])
    return pt

class main_data:
    src_lines = None
    tgt_pts = None
    # Matplotlib collections
    src_coll = None
    tgt_coll = None
    B_coll= None
    dB_coll= None
    emf_coll= None

    def __init__(self, ax):
        self.ax = ax

    def redraw(self, src_lines, tgt_pts):
        # EMI source lines
        if src_lines is None:
            src_lines = self.src_lines
        else:
            self.src_coll = replace_collection(self.src_coll, plot_source(self.ax, src_lines))
            self.src_lines = src_lines

        # Target points
        if tgt_pts is None:
            tgt_pts = self.tgt_pts
        else:
            self.tgt_coll = replace_collection(self.tgt_coll, self.ax.scatter(*tgt_pts.transpose(), **TARGET_FMT))
            self.tgt_pts = tgt_pts

        # Calculate EMI parameters B and dB for each target point
        emi_params = emi_calc.calc_all_emis(tgt_pts, src_lines)

        pts = numpy.array([emi_pars['pt'] for emi_pars in emi_params]).transpose()

        # Magnetic field
        if B_FMT:
            B = [emi_pars['B'].dot(FIELD_SCALE) for emi_pars in emi_params]
            B = numpy.array(B).transpose()
            self.B_coll = replace_collection(self.B_coll, self.ax.quiver(*pts, *B, **B_FMT))

        # The direction of "movement" of the field because of current increase
        if dB_FMT:
            dB = [emi_pars['dB'].dot(FIELD_SCALE) for emi_pars in emi_params]
            dB = numpy.array(dB).transpose()
            self.dB_coll = replace_collection(self.dB_coll, self.ax.quiver(*pts, *dB, **dB_FMT))

        # The EMF induced because of field change
        if EMF_FMT:
            emf = [numpy.cross(emi_pars['B'], emi_pars['dB']).dot(EMF_SCALE) for emi_pars in emi_params]
            emf = numpy.array(emf).transpose()
            self.emf_coll = replace_collection(self.emf_coll, self.ax.quiver(*pts, *emf, **EMF_FMT))
        return True

    def get_collections(self):
        return [self.src_coll, self.tgt_coll, self.B_coll, self.dB_coll, self.emf_coll]

class visible_clicked:
    def __init__(self, data):
        self.data = data

    def __call__(self, label):
        for coll in self.data.get_collections():
            if coll.get_label() == label:
                coll.set_visible(not coll.get_visible())
                break
        pyplot.draw()

class transform:
    xform_src = None    # True: source, False: target
    xform_type = None   # LABEL_TRANSLATE, LABEL_ROTATE, LABEL_SCALE
    sliders = [None, None, None]

    def __init__(self, data, origin):
        self.data = data
        self.origin = origin

    def set_slider(self, idx, slider):
        self.sliders[idx] = slider

    def src_type_changed(self):
        active = not(self.xform_src is None or self.xform_type is None)
        slider_params = None
        if active:
            if self.xform_type == LABEL_TRANSLATE:
                slider_params = TRANSLATE_PARAMS
            elif self.xform_type == LABEL_ROTATE:
                slider_params = ROTATE_PARAMS
            elif self.xform_type == LABEL_SCALE:
                slider_params = SCALE_PARAMS
        for s in self.sliders:
            if s is not None:
                s.set_active(active)
                s.reset()
                #TODO: Change slider range
                #if slider_params:
                #
                #    s.valmin, s.valval = slider_params[:2]
                #    s.set_val(slider_params[2])

    def src_changed(self, src):
        self.xform_src = src
        self.src_type_changed()

    def type_changed(self, xform_type):
        self.xform_type = xform_type
        self.src_type_changed()

    def pos_changed(self, slider_idx, pos):
        if self.xform_src is True:
            org_pts = numpy.array(SOURCE_POLYLINE)
            pts = self.data.src_lines
        elif self.xform_src is False:
            org_pts = numpy.array(TARGET_POINTS)
            pts = self.data.tgt_pts
        else:
            return

        origin = self.origin[slider_idx]
        if self.xform_type == LABEL_TRANSLATE:
            for idx, pt in enumerate(pts):
                pt[slider_idx] = org_pts[idx][slider_idx] + pos
        elif self.xform_type == LABEL_ROTATE:
            for idx, pt in enumerate(pts):
                pts[idx] = rotate_point(org_pts[idx] - self.origin, slider_idx, pos) + self.origin
        elif self.xform_type == LABEL_SCALE:
            for idx, pt in enumerate(pts):
                pt[slider_idx] = (org_pts[idx][slider_idx] - origin) * pos + origin
        else:
            return

        if self.xform_src == True:
            self.data.redraw(pts, None)
        elif self.xform_src == False:
            self.data.redraw(None, pts)
        return

class transform_what:
    def __init__(self, xform):
        self.xform = xform

    def __call__(self, label):
        if label == self.xform.data.src_coll.get_label():
            self.xform.src_changed(True)
        elif label == self.xform.data.tgt_coll.get_label():
            self.xform.src_changed(False)
        else:
            self.xform.src_changed(None)

class transform_type:
    def __init__(self, xform):
        self.xform = xform

    def __call__(self, label):
        self.xform.type_changed(label)

class xform_changed:
    def __init__(self, xform, idx):
        self.xform = xform
        self.idx = idx

    def __call__(self, pos):
        self.xform.pos_changed(self.idx, pos)
        pyplot.draw()

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
    check.on_clicked(visible_clicked(data))

    # Transformation selection context
    xform = transform(data, numpy.array(TRANSFORM_ORG))

    # Radio buttons to select what to control
    rect = [AX_BTN_WIDTH, 5 * AX_BTN_HEIGHT / 9, AX_BTN_WIDTH, 4 * AX_BTN_HEIGHT / 9]
    rax = fig.add_axes(deflate_rect(rect))
    radio = widgets.RadioButtons(rax, [data.src_coll.get_label(), data.tgt_coll.get_label()], None)
    radio.on_clicked(transform_what(xform))

    # Radio buttons to select how to control it
    rect = [AX_BTN_WIDTH, 0, AX_BTN_WIDTH, 5 * AX_BTN_HEIGHT / 9]
    rax = fig.add_axes(deflate_rect(rect))
    radio2 = widgets.RadioButtons(rax, [LABEL_TRANSLATE, LABEL_ROTATE, LABEL_SCALE], None)
    radio2.on_clicked(transform_type(xform))

    # Sliders to transform X/Y/Z
    for idx in range(AX_NUM_SLIDERS):
        rect = [2 * AX_BTN_WIDTH, (AX_NUM_SLIDERS - 1 - idx) * AX_BTN_HEIGHT / AX_NUM_SLIDERS,
                1 - 2 * AX_BTN_WIDTH, AX_BTN_HEIGHT / AX_NUM_SLIDERS]
        rax = pyplot.axes(deflate_rect(rect, AX_TEXT_WIDTH + AX_MARGIN))
        name = bytearray(['X'.encode()[0] + idx]).decode()
        slider = widgets.Slider(rax, name, -2, 2, 0)
        slider.on_changed(xform_changed(xform, idx))
        xform.set_slider(idx, slider)


    ax.legend()
    pyplot.show()
    return 0

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
