'''Electromagnetic induction model

Tested on python 3.7.5. Requires: numpy, matplotlib
'''
import sys
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.widgets as widgets
import mpl_toolkits.mplot3d as mplot3d

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


def calculate_emi(pt, line, coef=1):
    """Calculate the magnetic field at specific point, induced by electric current flowing along
    a line segment. Also, the direction where this field would shift, when the current is increased.
    """
    emi_params = numpy.zeros((2, pt.shape[0]), dtype=numpy.float64)

    # Start and end 'r' vectors
    r0 = pt - line[0]
    r1 = pt - line[1]

    # Calculate the integral from Biot–Savart law (https://en.wikipedia.org/wiki/Biot–Savart_law):
    #   dl x r / sqrt(l^2 + R^2)^3
    #
    # The "l" origin is selected at the closest point to the target to simplify calculations.
    # Thus "r = l^2 + R^2" and "|dl x r| = |dl|.R", where R is distance between the target and origin.
    #
    # Use integral calculator https://www.integral-calculator.com/ (substitute l with x):
    #   int[ R/sqrt(x^2 + R^2)^3 dx ] = x / (R * sqrt(x^2 + R^2)) + C
    delta = line[1] - line[0]
    len2 = delta.dot(delta)
    if not len2:
        return emi_params   # Zero length, return zero EMI params

    # Normalized vector between start and end, useful for subsequent calculations
    delta_n = delta / numpy.sqrt(len2)

    # The '-' is to base the vector at the origin, instead of at line[0]
    l0 = -delta_n.dot(delta_n.dot(r0))
    l1 = l0 + delta
    R = l0 + r0

    #
    # Integral at the start of interval
    #
    # |l0 x r0| = |l0|.|R|
    vect0 = numpy.cross(l0, r0)

    # Divide by 'r0'
    divider = numpy.sqrt(r0.dot(r0))
    if not divider:
        return None     # Target point coincides with "line[0]"
    vect0 /= divider

    #
    # Integral at the end of interval
    #
    # |l1 x r1| = |l1|.|R|
    vect1 = numpy.cross(l1, r1)

    # Divide by 'r1'
    divider = numpy.sqrt(r1.dot(r1))
    if not divider:
        return None     # Target point coincides with "line[1]"
    vect1 /= divider

    #
    # Combine both integrals
    #
    # Divide by 'R^2'
    divider = R.dot(R)
    if not divider:
        return None     # Target point lies on the "line"

    B = (vect1 - vect0) / divider

    emi_params[0] = B
    # The direction of "movement" of B, when the current is increased, use the same magnitude as B
    #
    # Important:
    #   The sum of B-s is NOT perpendicular to the sum of their perpendiculars. Thus, the direction
    # of "movement" of the summary B can not be calculated from itself. These vectors must be
    # summed separately.
    emi_params[1] = numpy.cross(B, delta_n)
    return emi_params

def plot_source(ax, src_lines):
    src_lines = src_lines.transpose()
    ##ax.plot(*src_lines.transpose(), **SOURCE_FMT)
    return ax.quiver(*src_lines[:,:-1], *(src_lines[:,1:] - src_lines[:,:-1]), **SOURCE_FMT)

class on_clicked:
    def __init__(self, lines):
        self.lines = lines

    def __call__(self, label):
        for line in self.lines:
            if line.get_label() == label:
                line.set_visible(not line.get_visible())
                break
        pyplot.draw()

def main(argv):
    """Main entry"""
    fig = pyplot.figure()
    ax = fig.gca(projection='3d', adjustable='box')#mplot3d.axes3d.Axes3D(fig)

    # EMI source lines
    src_lines = numpy.array(SOURCE_POLYLINE)
    src_col = plot_source(ax, src_lines)

    # Target points
    tgts =  numpy.array(TARGET_POINTS).transpose()
    tgt_col = ax.scatter(*tgts, **TARGET_FMT)

    # Calculate EMI parameters B and dB for each target point
    emi_params = []
    for pt in TARGET_POINTS:
        pt = numpy.array(pt)
        emi_pars = None
        for idx in range(len(src_lines) - 1):
            emi = calculate_emi(pt, src_lines[idx:idx+2])
            if emi is not None:     # Ignore collinear target points
                if emi_pars is None:
                    emi_pars = emi
                else:
                    emi_pars += emi
        if emi_pars is not None:
            emi_params.append({'pt': pt, 'B': emi_pars[0], 'dB': emi_pars[1]})

    pts = numpy.array([emi_pars['pt'] for emi_pars in emi_params]).transpose()

    # Magnetic field
    if B_FMT:
        B = [emi_pars['B'].dot(FIELD_SCALE) for emi_pars in emi_params]
        B = numpy.array(B).transpose()
        B_col = ax.quiver(*pts, *B, **B_FMT)

    # The direction of "movement" of the field because of current increase
    if dB_FMT:
        dB = [emi_pars['dB'].dot(FIELD_SCALE) for emi_pars in emi_params]
        dB = numpy.array(dB).transpose()
        dB_col = ax.quiver(*pts, *dB, **dB_FMT)

    # The EMF induced because of field change
    if EMF_FMT:
        emf = [numpy.cross(emi_pars['B'], emi_pars['dB']).dot(EMF_SCALE) for emi_pars in emi_params]
        emf = numpy.array(emf).transpose()
        emf_col = ax.quiver(*pts, *emf, **EMF_FMT)

    # Check boxes to show/hide individual elements
    lines = [src_col, tgt_col, B_col, dB_col, emf_col]
    rax = pyplot.axes([0.02, 0.02, 0.2, 0.2])
    labels = [line.get_label() for line in lines]
    visibility = [line.get_visible() for line in lines]
    check = widgets.CheckButtons(rax, labels, visibility)
    check.on_clicked(on_clicked(lines))

    set_axes_equal(ax)

    ax.legend()
    pyplot.show()
    return 0

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
