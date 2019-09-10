'''Find the center of circle specified by three points

The points are obtained interactively via matplotlib.pyplot.ginput()

Tested on python 3.7.2. Reuires: numpy, matplotlib
'''
import numpy

import matplotlib.pyplot as pyplot

CHORD_COLOR = 'green'
BISECT_COLOR = 'yellow'
RADIUS_COLOR = 'blue'
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
def plot_line(start, end, color):
    return pyplot.plot([start[0], end[0]], [start[1], end[1]], color=color)

def plot_vector(start, vect, color):
    return plot_line(start, vect + start, color=color)

def do_ginput():
    pyplot.title('Enter 3 circle points')
    input = pyplot.ginput(3)
    print('Input point', input)
    input = numpy.array(input)

    # Display input vectors
    pyplot.plot(input[:,0], input[:,1], color=CHORD_COLOR, label='Chords')
    return input

def get_bisect_lines(vectors):
    # Bisect-lines: Perpendicular vectors starting at the middle of input chords
    bisect_lines = []
    for idx in range(len(vectors) - 1):
        chord = vectors[idx + 1] - vectors[idx]
        perp = perpendicular(chord)
        start = chord * .5 + vectors[idx]
        line, = plot_vector(start, perp, BISECT_COLOR)
        bisect_lines.append({'start': start, 'end': perp + start})

    if line:
        line.set_label('Bisects')
    return bisect_lines

#
# main
#
def main():
    pyplot.gca().set_autoscale_on(False)
    pyplot.gca().set_aspect('equal')

    vectors = do_ginput()
    bisect_lines = get_bisect_lines(vectors)

    # Intersection of bisect lines
    start = bisect_lines[1]['start']
    vect = intersect(bisect_lines[1]['end'] - start, bisect_lines[0]['end'] - start, bisect_lines[0]['start'] - start)
    center = vect + start
    print('Center', center)

    for vect in vectors:
        print('\tRadius len', vect_len(vect - center))
        line, = plot_line(vect, center, RADIUS_COLOR)
    if line:
        line.set_label('Radiuses')

    pyplot.legend()
    return pyplot.show()

if __name__ == '__main__':
    exit(main())
