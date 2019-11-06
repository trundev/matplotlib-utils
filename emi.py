'''Electromagnetic induction model

Tested on python 3.7.5. Requires: numpy, matplotlib
'''
import sys
import numpy
import matplotlib.pyplot as pyplot
import mpl_toolkits.mplot3d as mplot3d

# Source current flow
SOURCE_POLYLINE = [
   [0., 0., .0],
   [1., 0., .0],   # right
   [1., 2., .0],   # up
   [0., 2., .0],   # left
   [0., 0., .0],   # down
]
CURRENT = 1
# Points to calculate induction vectors
TARGET_POINTS = [
    [.5, 1., .02],  # Center
    [.5, 1., 1],    # Center up
    [0, 1., 1],    # Center up/left
    [1, 1., 1],    # Center up/right
    [-1, 1., .1],  # Center left
    [ 2, 1., .1],  # Center right
]
TARGET_POINTS = [
    *(
        [x, 1, 1] for x in numpy.linspace(-.5, 1.5, num=8)
     ),
    *(
        [-.5, 1, z] for z in numpy.linspace(-1, 1, num=8)
     ),
    *(
        [x, 1, -1] for x in numpy.linspace(-.5, 1.5, num=8)
     ),
    *(
        [1.5, 1, z] for z in numpy.linspace(-1, 1, num=8)
     ),
]

SOURCE_FMT = 'g'
TARGET_FMT = '+b'
B_FMT = '--m'
dB_FMT = ':y'   #None   #
EMF_FMT = '-.r' #None   #
FIELD_SCALE = .2

# Vacuum permeability: H/m (Henry per meter) or N/A^2 (Newton per square ampere)
M0 =  4 * numpy.pi * 1e-7
M0_4PI = 1e-7   # Or M0 / 4 * numpy.pi

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


def calculate_integral(l_n, r):
    """Calculate the integral from Biot–Savart law: "dl x r / sqrt(l^2 + R^2)^3"
    (https://en.wikipedia.org/wiki/Biot–Savart_law)
    Calculations are simplified by choosing the "l" origin to be the closest point to the target.
    Thus "r = l^2 + R^2" and "|dl x r| = |dl|.R", where R is distance between the target and origin.

    Arguments:
        l_n - normalized vector along "l"
        r - vector from the point being integrated to the target
    """
    # Use integral calculator: https://www.integral-calculator.com/:
    # R/sqrt(x^2 + R^2)^3 dx => x / (R * sqrt(x^2 + R^2))
    if False:
        # Non-optimized calculation
        # Integral = (l_n x r) * |l| / (|l_n x r| * |r| * |R|)
        vect = numpy.cross(l_n, r)

        # Normalize result vector
        divider = numpy.sqrt(vect.dot(vect))

        # Multiply by 'l' (l_n is in the opposite direction)
        l = -l_n.dot(r)
        vect *= l

        # Divide by 'r'
        divider *= numpy.sqrt(r.dot(r))

        # Divide by 'R' ('l' is already flipped)
        R = r + l * l_n
        divider *= numpy.sqrt(R.dot(R))

        if not divider:
            return None
        vect /= divider

    else:
        # Some optimizations -- avoid cross product normalization
        # |l x r| = |l|.|r|.sin phi = |l|.|R|
        # Integral = (l_n x r) * |l| / (R^2 * r)
        vect = numpy.cross(l_n, r)

        # Multiply by 'l' (l_n is in the opposite direction)
        l = -l_n.dot(r)
        vect *= l

        # Divide by 'R^2' ('l' is already flipped)
        R = r + l * l_n
        divider = R.dot(R)

        # Divide by 'r'
        divider *= numpy.sqrt(r.dot(r))
        if not divider:
            return None
        vect /= divider

    return vect

def calculate_emi(pt, line, coef=1):
    emi_params = numpy.zeros((2, pt.shape[0]), dtype=numpy.float64)

    # Start and end 'r' vectors
    r0 = pt - line[0]
    r1 = pt - line[1]

    delta = line[1] - line[0]
    len2 = delta.dot(delta)
    if len2:
        # Normalized vector between start and end (it is useful)
        delta_n = delta / numpy.sqrt(len2)

        # Integral between start and end
        B = calculate_integral(delta_n, r1) \
            - calculate_integral(delta_n, r0)

        emi_params[0] = B
        emi_params[1] = numpy.cross(B, delta_n)

    else:
        # Unable to calculate EMI
        delta_n = None

    return emi_params

def main(argv):
    """Main entry"""
    fig = pyplot.figure()
    ax = fig.gca(projection='3d', adjustable='box')#mplot3d.axes3d.Axes3D(fig)

    # EMI source lines
    src_lines = numpy.array(SOURCE_POLYLINE)
    ax.plot(*src_lines.transpose(), SOURCE_FMT)

    # Target points
    for pt in TARGET_POINTS:
        ax.plot(*([v] for v in pt), TARGET_FMT)

    # Calculate EMI parameters B and dB for each target point
    emi_params = []
    for pt in TARGET_POINTS:
        pt = numpy.array(pt)
        emi_pars = None
        for idx in range(len(src_lines) - 1):
            emi = calculate_emi(pt, src_lines[idx:idx+2])
            if emi_pars is None:
                emi_pars = emi
            else:
                emi_pars += emi
        emi_params.append({'pt': pt, 'B': emi_pars[0], 'dB': emi_pars[1]})

    for emi_pars in emi_params:
        pt = emi_pars['pt']

        # Magnetic field
        B = emi_pars['B'].dot(FIELD_SCALE) + pt 
        ax.plot([pt[0], B[0]], [pt[1], B[1]], [pt[2], B[2]], B_FMT)

        # The direction of "movement" of the field because of current increase
        if dB_FMT:
            dB = emi_pars['dB'].dot(FIELD_SCALE) + pt 
            ax.plot([pt[0], dB[0]], [pt[1], dB[1]], [pt[2], dB[2]], dB_FMT)

        # The EMF inducted because of field change
        if EMF_FMT:
            emf = numpy.cross(emi_pars['B'], emi_pars['dB'])
            emf = emf.dot(FIELD_SCALE) + pt 
            ax.plot([pt[0], emf[0]], [pt[1], emf[1]], [pt[2], emf[2]], EMF_FMT)

    set_axes_equal(ax)

    pyplot.legend()
    pyplot.show()
    return 0

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
