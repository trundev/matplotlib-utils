'''Electromagnetic induction model

Requires: numpy
'''
import sys
import numpy

def calc_emi(pt, line, coef=1):
    """Calculate the magnetic field at specific point, induced by electric current flowing along
    a line segment.

    Returns: [
            <B-vect>,   # Magnetic field at the point
            <dB-vect>,  # The direction where this field would shift, because of a unit current increase
        ]
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

    # Scale by a coefficient, like current or magnetic constant
    B *= coef

    emi_params[0] = B
    # The direction of "movement" of B, when the current is increased, use the same magnitude as B
    #
    # Important:
    #   The sum of B-s is NOT perpendicular to the sum of their perpendiculars. Thus, the direction
    # of "movement" of the summary B can not be calculated from itself. These vectors must be
    # summed separately.
    emi_params[1] = numpy.cross(B, delta_n)
    return emi_params

def calc_all_emis(tgt_pts, src_lines):
    """Calculate EMI parameters B and dB at each of the target points

    Requirements:
        tgt_pts and src_lines must be numpy.array
    Returns: {
            'pt': <pt-vect>,    # Point from tgt_pts, where the field is calculated
            'B': <B-vect>,      # Magnetic field vector
            'dB': <dB-vect>     # Direction of magnetic field shift, if the current increases
        }
    """
    emi_params = []
    for pt in tgt_pts:
        emi_pars = None
        for idx in range(len(src_lines) - 1):
            emi = calc_emi(pt, src_lines[idx:idx+2])
            if emi is not None:     # Ignore point collinear to the src-line
                if emi_pars is None:
                    emi_pars = emi
                else:
                    emi_pars += emi
        if emi_pars is not None:
            emi_params.append({'pt': pt, 'B': emi_pars[0], 'dB': emi_pars[1]})
    return emi_params
