'''Various coil configurations used when generating EM fields
'''
import numpy

# Rotate the polygon approximation of circle, to make first segment parallel to Y axis
ADJUST_SEGS = True

def helmholtz_coil(rad, num=2, len=None, segs=8):
    """Helmholtz coil: parallel circles, usually two at distance of R"""
    if len is None:     # Use the proper distance of R
        len = rad * (num - 1)
    return gen_coil(rad, rad, num, len, True, segs)

def helix_coil(rad, turns, len, segs=8):
    """Helix coil: regular cylindrical inductor coil"""
    return gen_coil(rad, rad, turns, len, False, segs)

def spiral_coil(rad0, rad1, turns, segs=8):
    """Spiral coil: spiral inductor coil"""
    return gen_coil(rad0, rad1, turns, 0, False, segs)

def gen_coil(rad0, rad1, turns, len, separate=False, segs=8, center=None):
    """Generic coil: conical spiral or separate circles"""
    num = segs
    angle = 2*numpy.pi
    if not separate:
        num = round(num * turns)    # Spiral can have a fractional turn
        angle *= turns
    phi = numpy.linspace(0, angle, num=num + 1, endpoint=True)
    if ADJUST_SEGS:     # Rotate the polygon, to make first segment parallel to Y axis
        phi -= numpy.pi / segs
    if rad1 is None:
        rad1 = rad0

    if separate:
        polyline = [
            *numpy.linspace(
                [rad0 * numpy.cos(phi), rad0 * numpy.sin(phi), phi.size * [-len / 2]],
                [rad1 * numpy.cos(phi), rad1 * numpy.sin(phi), phi.size * [len / 2]],
                num=turns, endpoint=True)
        ]
        polyline = numpy.transpose(polyline, (0,2,1))
    else:
        rad = numpy.linspace(rad0, rad1, num=phi.size)
        polyline = [
            rad * numpy.cos(phi), rad * numpy.sin(phi), numpy.linspace(-len / 2, len / 2, num=phi.size),
        ]
        polyline = numpy.transpose(polyline)

    if center is not None:
        polyline += center

    return polyline
