'''Find the center of circle specified by three points

The points are obtained interactively via matplotlib.pyplot.ginput()

Tested on python 3.7.2. Reuires: numpy, matplotlib
'''
import numpy

import matplotlib.pyplot as pyplot
import matplotlib.patches as patches

CHORD_FMT = '.--g'
BISECT_FMT = 'y'
RADIUS_FMT = 'b'
CIRCLE_STYLE = {'alpha':.5, 'fill':False, 'linestyle':':', 'color':'blue'}
POINT_FMT = '+r'
Z_VECTOR = numpy.array([0,0,1])

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
    def _len(v):
        # With 2D vectors numpy.cross() returns scalar
        return vect_len(v) if v.shape else v
    scale = _len(numpy.cross(point0, point1)) / _len(numpy.cross(point0, vect) + numpy.cross(vect, point1))
    return vect * scale

#
# pyplot helpers
#
def plot_line(ax, start, end,  *args, **kwargs):
    return ax.plot([start[0], end[0]], [start[1], end[1]],  *args, **kwargs)

def plot_vector(ax, start, vect,  *args, **kwargs):
    return plot_line(ax, start, vect + start,  *args, **kwargs)

def do_ginput(fig):
    pyplot.title('Enter 3 circle points')
    input = fig.ginput(3)
    pyplot.title('Circle')
    print('Input point', input)
    input = numpy.array(input)

    # Draw input vectors
    fig.gca().plot(input[:,0], input[:,1], CHORD_FMT, label='Chords')
    return input

def get_bisect_lines(vectors, ax):
    # Bisect-lines: Perpendicular vectors starting at the middle of input chords
    bisect_lines = []
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

    vectors = do_ginput(fig)
    bisect_lines = get_bisect_lines(vectors, ax)

    # Intersection of bisect lines
    start = bisect_lines[1]['start']
    vect = intersect(bisect_lines[1]['end'] - start, bisect_lines[0]['end'] - start, bisect_lines[0]['start'] - start)
    center = vect + start
    print('Center', center)

    # Draw center and the circle
    ax.plot(*center[:2], POINT_FMT)
    ax.add_artist(patches.Circle(center[:2], vect_len(vectors[0] - center), **CIRCLE_STYLE))

    for vect in vectors:
        print('\tRadius len', vect_len(vect - center))
        line, = plot_line(ax, vect, center, RADIUS_FMT)
    if line:
        line.set_label('Radiuses')

    pyplot.legend()
    return pyplot.show()

if __name__ == '__main__':
    exit(main())
