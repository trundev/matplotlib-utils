'''Find the center of circle specified by three points

The points are obtained interactively via matplotlib.pyplot.ginput()

Tested on python 3.7.2. Reuires: numpy, matplotlib
'''
import numpy

import matplotlib.pyplot as pyplot
#import matplotlib.patches as patches

TANGENT_STYLE = dict(marker='.', color='green', label='Tangents')
POINT_STYLE = dict(marker='x', color='red', label='Points')
INFLINE_STYLE = dict(alpha=.2, color='blue')
CHORD_STYLE = dict(alpha=.5, color='orange', linestyle=':')
RADIAL_STYLE = dict(alpha=.5, color='yellow', linestyle=':')
Z_VECTOR = numpy.array([0,0,1])

NUM_CIRCLES = 5

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
    return ax.plot([start[0], end[0]], [start[1], end[1]], *args, **kwargs)

def plot_vector(ax, start, vect,  *args, **kwargs):
    return plot_line(ax, start, vect + start, *args, **kwargs)

def plot_inf_line(ax, point, dir, *args, **kwargs):
    xbound = ax.get_xbound()
    ybound = ax.get_ybound()
    if dir[0] * dir[1] < 0:
        # top-left or bottom-right orientation
        pt0 = numpy.array([xbound[1], ybound[0]])
        pt1 = numpy.array([xbound[0], ybound[1]])
    else:
        # top-right or bottom-left orientation
        pt0 = numpy.array([xbound[0], ybound[0]])
        pt1 = numpy.array([xbound[1], ybound[1]])

    # dir(dir.V) / dir^2
    pt0 = dir * dir.dot(pt0 - point) / dir.dot(dir) + point
    pt1 = dir * dir.dot(pt1 - point) / dir.dot(dir) + point
    return plot_line(ax, pt0, pt1, *args, **kwargs)

def do_ginput(fig):
    pyplot.title('Enter circle tangent and a point')
    input = fig.ginput(3)
    pyplot.title('Circle')
    print('Input points', input)
    input = numpy.array(input)

    # Draw the first line (tangent)
    ax = fig.gca()
    ax.plot(input[:2,0], input[:2,1], **TANGENT_STYLE)

    # Draw the point
    sz = ax.get_xlim()
    sz = (sz[1] - sz[0]) / 100
    ax.plot(*input[2,:2], **POINT_STYLE)

    return input

def tangent_circle(ax, point, tangent_pt, tangent_dir):
    # Chord
    plot_line(ax, point, tangent_pt, **CHORD_STYLE)

    # Radial from the tangent
    radial0_dir = perpendicular(tangent_dir)
    plot_vector(ax, tangent_pt, radial0_dir, **RADIAL_STYLE)

    # Radial from the chord (bisect)
    radial1_pt = (tangent_pt + point) / 2
    radial1_dir = perpendicular(tangent_pt - point)
    plot_vector(ax, radial1_pt, radial1_dir, **RADIAL_STYLE)

    # Intersect both radials
    center = intersect(radial0_dir, radial1_pt - tangent_pt, radial1_pt + radial1_dir - tangent_pt)
    center = center[:2] + tangent_pt
    ax.plot(*center, 'x')
    return center

#
# main
#
def main():
    fig = pyplot.gcf()
    ax = fig.gca()
    ax.set_autoscale_on(False)
    ax.set_aspect('equal')

    vectors = do_ginput(fig)

    tangent_pt = vectors[0]
    tangent_dir = vectors[1] - vectors[0]
    plot_inf_line(ax, tangent_pt, tangent_dir, **INFLINE_STYLE)

    # Draw NUM_CIRCLES circles via points on the line
    step = 1 / (NUM_CIRCLES - 1) if NUM_CIRCLES > 1 else 1
    for t in numpy.arange(0, 1 + step, step):
        line_pt = tangent_pt + tangent_dir * t
        tangent_circle(ax, vectors[2], line_pt, tangent_dir)

    pyplot.legend()
    return pyplot.show()

if __name__ == '__main__':
    exit(main())
