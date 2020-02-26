'''Electromagnetic induction model

Requires: numpy
'''
import sys
import numpy

def calc_emi(pt, line, coef=1):
    """Calculate the magnetic field at specific point, induced by electric current flowing along
    a line segment.

    Returns: [
            <B-vect>,       # Magnetic field vector at the point
            <gradB-vect>,   # Gradient vector of the magnetic field (calculated from the magnitude)
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
    delta_len2 = delta.dot(delta)
    if not delta_len2:
        return emi_params   # Zero length, return zero EMI params

    # Vector projections of "r0" and "r1" in the direction of "delta"
    # The '-' is to set the origin at the projected point, instead of at line[0]
    l0 = -delta.dot(delta.dot(r0) / delta_len2)
    l1 = l0 + delta
    R = l0 + r0

    #
    # Integral at the start of interval
    #
    # Start with l0 x R to get a direction vector with length of |l0|.|R|
    vect0 = numpy.cross(l0, R)

    # Divide by 'r0'
    r0_len = numpy.sqrt(r0.dot(r0))
    if not r0_len:
        return None     # Target point coincides with "line[0]"
    vect0 /= r0_len

    #
    # Integral at the end of interval
    #
    # Start with l1 x R to get a direction vector with length of |l1|.|R|
    vect1 = numpy.cross(l1, R)

    # Divide by 'r1'
    r1_len = numpy.sqrt(r1.dot(r1))
    if not r1_len:
        return None     # Target point coincides with "line[1]"
    vect1 /= r1_len

    #
    # Combine both integrals
    #
    # Divide result by 'R^2', resulting:
    # |l|.|R| / |r| / |R|^2 = |l| / (|R|.|r|)
    R_len2 = R.dot(R)
    if not R_len2:
        return None     # Target point lies on the "line"

    B = (vect1 - vect0) / R_len2

    # Scale by a coefficient, like current or magnetic constant
    B *= coef

    emi_params[0] = B
    # The direction of "movement" of B, when the current is increased, use the same magnitude as B
    #
    # Important:
    #   The sum of B-s is NOT perpendicular to the sum of their perpendiculars. Thus, the direction
    # of "movement" of the summary B can not be calculated from itself. These vectors must be
    # summed separately.
    emi_params[1] = numpy.cross(B, delta / numpy.sqrt(delta_len2))
    return emi_params

def calc_all_emis(tgt_pts, src_lines):
    """Calculate EMI parameters B and dB at each of the target points

    Requirements:
        tgt_pts and src_lines must be numpy.array
    Returns: {
            'pt': <pt-vect>,        # Point from tgt_pts, where the field is calculated
            'B': <B-vect>,          # Magnetic field vector
            'gradB': <gradB-vect>   # Gradient of the magnetic field magnitude
        }
    """
    # The result is of the same shape as tgt_pts, but the last dimention is moved into 'pt' field
    emi_pars_dt = numpy.dtype([
        ('pt', tgt_pts.dtype, (3,)),
        ('B', tgt_pts.dtype, (3,)),
        ('gradB', tgt_pts.dtype, (3,)),
        ])
    emi_params = numpy.empty(tgt_pts.shape[:-1], emi_pars_dt)

    emi_it = numpy.nditer(emi_params, flags=['multi_index'], op_flags=[['writeonly']])
    while not emi_it.finished:
        pt = tgt_pts[emi_it.multi_index]

        emi_pars = None
        for idx in range(len(src_lines) - 1):
            emi = calc_emi(pt, src_lines[idx:idx+2])
            if emi is not None:     # Ignore point collinear to the src-line
                if emi_pars is None:
                    emi_pars = emi
                else:
                    emi_pars += emi

        emi_it[0]['pt'] = pt
        emi_it[0]['B'] = emi_pars[0] if emi_pars is not None else None
        emi_it[0]['gradB'] = emi_pars[1] if emi_pars is not None else None
        emi_it.iternext()

    return emi_params
