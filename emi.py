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

# Slider origin/direction parameters
SOURCE_SLIDER_ORG = [0.5, 1, 0]
TARGET_SLIDER_DIR = [0, 1, 0]

FIELD_SCALE = .2
EMF_SCALE = .1

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
    def __init__(self, data, origin):
        self.data = data
        self.origin = origin

    def __call__(self, pos):
        src_lines = numpy.array(SOURCE_POLYLINE)
        for idx, pt in enumerate(src_lines):
            src_lines[idx] = (pt - self.origin) * pos + self.origin

        self.data.redraw(src_lines, None)
        pyplot.draw()

class tgt_changed:
    def __init__(self, data, move_dir):
        self.data = data
        self.move_dir = move_dir

    def __call__(self, pos):
        tgt_pts = numpy.array(TARGET_POINTS)
        for idx, _ in enumerate(tgt_pts):
            tgt_pts[idx] += self.move_dir * pos

        self.data.redraw(None, tgt_pts)
        pyplot.draw()

def main(argv):
    """Main entry"""
    fig = pyplot.figure()
    ax = fig.gca(projection='3d', adjustable='box')#mplot3d.axes3d.Axes3D(fig)

    # Initial drawing
    data = main_data(ax)
    data.redraw(numpy.array(SOURCE_POLYLINE), numpy.array(TARGET_POINTS))

    # Check boxes to show/hide individual elements
    rax = pyplot.axes([0.02, 0.02, 0.2, 0.2])
    colls = data.get_collections()
    labels = [coll.get_label() for coll in colls]
    visibility = [coll.get_visible() for coll in colls]
    check = widgets.CheckButtons(rax, labels, visibility)
    check.on_clicked(on_clicked(data))

    # Slider to scale source lines
    rax = pyplot.axes([0.35, 0.10, 0.5, 0.04])
    src_slider = widgets.Slider(rax, data.src_coll.get_label(), 0, 2, 1)
    src_slider.on_changed(src_changed(data, numpy.array(SOURCE_SLIDER_ORG)))

    # Slider to move target points
    rax = pyplot.axes([0.35, 0.02, 0.5, 0.04])
    tgt_slider = widgets.Slider(rax, data.tgt_coll.get_label(), -2, 2, 0)
    tgt_slider.on_changed(tgt_changed(data, numpy.array(TARGET_SLIDER_DIR)))

    set_axes_equal(ax)

    ax.legend()
    pyplot.show()
    return 0

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
