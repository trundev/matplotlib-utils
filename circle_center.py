'''Find the center of circle specified by three points or four points (three lines)

The points are obtained interactively via matplotlib.pyplot.ginput().
The number of points selects the circle type:
- three points -- circumscribed circle (via the points)
- four points -- inscribed circle (tangent to the lines)

Tested on python 3.7.2. Reuires: numpy, matplotlib
'''
import numpy

import matplotlib.pyplot as pyplot
import matplotlib.patches as patches

CHORD_FMT = '.--g'
BISECT_FMT = 'y'
RADIUS_FMT = 'b'
CIRCLE_STYLE = dict(alpha=.5, fill=False, linestyle=':', color='blue')
POINT_FMT = '+r'
Z_VECTOR = numpy.array([0,0,1])

ALLOW_OPTIMIZATION = 1

#
# Basic vector manipulations
#
def perpendicular(vect):
    return numpy.cross(vect, Z_VECTOR)[:2]

def vect_len(vect):
    len2 = vect.dot(vect)
    return len2 ** .5

def normalize(vect):
    return vect / vect_len(vect)

def intersect(vect, point0, point1):
    """Intersection between vector and line between two points"""
    if ALLOW_OPTIMIZATION:
        # Optimize out the cross product and square root (meaningful in 3D scenarios only):
        # Scale = |AxB|^2 / [(AxB).(AxC) + (AxB).(CxB)] =
        # = (|A|^2|B|^2 - |A.B|^2) / (A - B).[A(B.C) - B(A.C)]
        scale = point0.dot(point0) * point1.dot(point1) - point0.dot(point1) ** 2
        scale /= (point0 - point1).dot( point0 * point1.dot(vect) - point1 * point0.dot(vect))
    else:
        # Scale = |AxB| / (|AxC + CxB|)
        def _len(v):
            # With 2D vectors numpy.cross() returns scalar
            return vect_len(v) if v.shape else v
        scale = _len(numpy.cross(point0, point1)) / _len(numpy.cross(point0, vect) + numpy.cross(vect, point1))
    return vect * scale

def distance_to_line(point, dir):
    """Shortest vector to line defined by point and direction (origin based)"""
    vect = numpy.cross(point, dir)
    if not vect.shape:
        # Make Z-only 3D vector
        vect = numpy.array([0,0, vect])
    vect = numpy.cross(dir, vect)
    vect /= dir.dot(dir) 
    return vect if point.size > 2 else vect[:2]

#
# pyplot helpers
#
def plot_line(ax, start, end,  *args, **kwargs):
    return ax.plot([start[0], end[0]], [start[1], end[1]],  *args, **kwargs)

def plot_vector(ax, start, vect,  *args, **kwargs):
    return plot_line(ax, start, vect + start,  *args, **kwargs)

def do_ginput(fig, num_points):
    input = fig.ginput(num_points)
    print('Input point', input)
    input = numpy.array(input)

    # Draw input vectors
    fig.gca().plot(input[:,0], input[:,1], CHORD_FMT, label='Chords')
    return input

def get_bisect_lines(vectors, ax, angle_bisect=False):
    bisect_lines = []
    line = None
    if angle_bisect:
        # Bisect-lines: Angle bisectors between circle tangents
        for idx in range(len(vectors) - 2):
            start = vectors[idx + 1]
            side0 = vectors[idx] - start
            side1 = vectors[idx + 2] - start
            bisect = normalize(side0) + normalize(side1)
            line, = plot_vector(ax, start, bisect, BISECT_FMT)
            bisect_lines.append({'start': start, 'end': bisect + start})
    else:
        # Bisect-lines: Perpendicular vectors starting at the middle of circle chords
        for idx in range(len(vectors) - 1):
            chord = vectors[idx + 1] - vectors[idx]
            perp = perpendicular(chord)
            start = chord * .5 + vectors[idx]
            line, = plot_vector(ax, start, perp, BISECT_FMT)
            bisect_lines.append({'start': start, 'end': perp + start})

    if line:
        line.set_label('Bisects')
    return bisect_lines

#
# main
#
def main():
    fig = pyplot.gcf()
    ax = fig.gca()
    ax.set_autoscale_on(False)
    ax.set_aspect('equal')

    # Circumscribed circle: 3 points
    # Inscribed circle: 3 lines (4 points)
    pyplot.title('Enter 4 or 3 circle points (<ESC> to stop)')
    vectors = do_ginput(fig, 4)

    # Inscribed or circumscribed circle
    if len(vectors) == 4:
        do_inscribed = True
    else:
        do_inscribed = False
    pyplot.title(('Inscribed' if do_inscribed else 'Circumscribed') + ' circle')

    bisect_lines = get_bisect_lines(vectors, ax, do_inscribed)

    # Center of the circle is the intersection of the bisect lines
    if len(bisect_lines) > 1:
        start = bisect_lines[1]['start']
        vect = intersect(bisect_lines[1]['end'] - start, bisect_lines[0]['end'] - start, bisect_lines[0]['start'] - start)
        center = vect + start
        print('Center', center)

        # Select raduis vectors
        if do_inscribed:
            radius_vects = []
            for idx in range(len(vectors) - 1):
                vect = distance_to_line(vectors[idx] - center, vectors[idx + 1] - vectors[idx])
                radius_vects.append(vect)
        else:
            radius_vects = vectors - center

        # Draw center and the circle
        ax.plot(*center[:2], POINT_FMT)
        ax.add_artist(patches.Circle(center[:2], vect_len(radius_vects[0]), **CIRCLE_STYLE))

        for vect in radius_vects:
            print('\tRadius len', vect_len(vect))
            line, = plot_vector(ax, center, vect, RADIUS_FMT)
        if line:
            line.set_label('Radiuses')
    else:
        pyplot.title('Insufficient number of points')

    pyplot.legend()
    return pyplot.show()

if __name__ == '__main__':
    exit(main())
