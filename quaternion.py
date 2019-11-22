"""Quaternion multiplication (3D vector rotation) experiments

  https://www.mathworks.com/help/aeroblks/quaternionmultiplication.html
  https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
  https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternion_as_rotations
"""
import sys
import math
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.widgets as widgets
import mpl_toolkits.mplot3d as mplot3d

QUAT_DT=numpy.float64

SOURCE_FMT = dict(color='green', label='Source')
RESULT_FMT = dict(color='blue', label='Result')
ROTATED_FMT = dict(color='cyan', label='Rotated')

def normalize(quat):
    """Normalize (convert to unit vector/quaternion)

    Returns None for the zero vectors
    """
    q = numpy.array(quat)
    len2 = q.dot(q)
    return q / len2 ** .5 if len2 else None

def quaternion_multiply(quat1, quat2):
    """Hamilton product between quaternions"""
    if True:
        # Optimization using numpy cross-product
        # prod = (w1, V1)(w2, V2) = (w1*w2 - V1.V2, w1*V2 + w2*V1 + V1xV2)
        w1 = quat1[0]
        V1 = quat1[1:]
        w2 = quat2[0]
        V2 = quat2[1:]
        prod = numpy.zeros_like(quat1)
        prod[0] = w1*w2 - V1.dot(V2)
        prod[1:] = w1*V2 + w2*V1 + numpy.cross(V1, V2)
        return prod

    # Basic Hamilton formula
    a1, b1, c1, d1 = quat1
    a2, b2, c2, d2 = quat2
    return numpy.array([a1*a2 - b1*b2 - c1*c2 - d1*d2,
                        a1*b2 + b1*a2 + c1*d2 - d1*c2,
                        a1*c2 - b1*d2 + c1*a2 + d1*b2,
                        a1*d2 + b1*c2 - c1*b2 + d1*a2],
                       dtype=QUAT_DT)

def rotate_vector(quat, vector):
    """Rotate vector using qaternion"""
    quat = normalize(quat)
    res = quaternion_multiply(quat, pad_vector_to_quaternion(vector))
    quat[1:] = -quat[1:]
    res = quaternion_multiply(res, quat)
    return res[1:]

def quaternion_rot_angle(quat):
    """Angle of rotation: 2*atan2(|(x,y,z)|, w)

    Result in range 0-2*pi (0-360 degrees)"""
    q = quat[1:]
    return 2 * math.atan2(q.dot(q)**.5, quat[0])

#
# Utils
#
def pad_vector_to_quaternion(vector):
    """Pad with zeros to make quaternion"""
    return numpy.array([ *[0]*(4 - len(vector)), *vector ])

def vect_to_str(quat, fmt='%.3f'):
    """Compact string representation"""
    if quat is None:
        return '<none>'
    return '[' + ', '.join( [fmt%x for x in quat]) + ']'

def plot_vectors(ax, vects, **kw):
    """Draw vectors (also vector parts of queternions)"""
    # Extract the vector part of the quaternions inside the list
    vects = [v[1:] if len(v) > 3 else v for v in vects]
    vects = numpy.array(vects).transpose()
    return ax.quiver(*numpy.zeros_like(vects), *vects, **kw)

#
# Show/hide check buttons in matplotlib axis view
#
class check_buttons:
    """Handle show/hide check buttons"""
    def __init__(self, lines=[]):
        self.lines = lines

    def __call__(self, label):
        for line in self.lines:
            if line.get_label() == label:
                line.set_visible(not line.get_visible())
                break
        pyplot.draw()

    def add_line(self, line):
        self.lines.append(line)

    def create(self, rax):
        labels = [line.get_label() for line in self.lines]
        visibility = [line.get_visible() for line in self.lines]
        return widgets.CheckButtons(rax, labels, visibility)

    def reg(self, chk):
        chk.on_clicked(self)

def main(argv):
    quats = []
    for arg in argv:
        quats.append( [float(x) for x in arg.split(',')] )
    print('Arguments:\n  ' + '\n  '.join( [str(x) for x in quats]))

    fig = pyplot.figure()
    ax = fig.gca(projection='3d')

    # Plot two points to autoscale
    ax.scatter([-1,1], [-1,1], [-1,1])

    # Show source
    plt_src = plot_vectors(ax, quats, **SOURCE_FMT)

    print('Result:')
    result = None
    res_quats = []
    for quat in quats:
        # Check if this is quaternion
        is_quat = True
        if len(quat) < 4:
            is_quat = False
            # Pad with zeros from start
            print('  Pad vector:', vect_to_str(quat), ':')
            quat = pad_vector_to_quaternion(quat)
            print('    ', vect_to_str(quat))

        if is_quat:
            print('  Normalizing quaternion:', vect_to_str(quat), ':')
            quat = normalize(quat)
            print('    ', vect_to_str(quat))
            print('    * rotation angle (+/-):', math.degrees(quaternion_rot_angle(quat)))
            res_quats.append(quat)

        if result is None:
            result = quat
        else:
            print('  Multiplying:', vect_to_str(result), vect_to_str(quat), ':')
            result = quaternion_multiply(result, quat)
            print('    ', vect_to_str(result))
            res_quats.append(result)


    btns = check_buttons([plt_src])

    if res_quats:
        plt_res = plot_vectors(ax, res_quats, **RESULT_FMT)
        btns.add_line(plt_res)

    # Quaternion and 3D vector(s): Do rotation
    if len(quats) >= 2 and len(quats[0]) == 4 and numpy.all([len(q) == 3 for q in quats[1:]]):
        vects = []
        for vect in quats[1:]:
            print('Rotated vector:', vect_to_str(vect))
            vect = rotate_vector(quats[0], vect)
            print('    ', vect_to_str(vect))
            vects.append(vect)

        plt_rot = plot_vectors(ax, vects, **ROTATED_FMT)
        btns.add_line(plt_rot)

    check = btns.create(pyplot.axes([0.02, 0.02, 0.2, 0.2]))
    btns.reg(check)

    ax.legend()
    pyplot.show()

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
