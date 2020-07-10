'''Electromagnetic induction model

Requires: numpy
'''
import sys
import numpy

def build_jacobian(l_comp, R_comp, l_vect, R_vect, B_vect):
    """Combine gradient components along 'l' and 'R' into a Jacobian matrix"""
    l_len = numpy.sqrt((l_vect * l_vect).sum(-1))
    R_len = numpy.sqrt((R_vect * R_vect).sum(-1))
    B_len = numpy.sqrt((B_vect * B_vect).sum(-1))
    # Empty 3x3 jacobian matrix
    jacob = numpy.zeros((B_vect.shape[-1], B_vect.shape[-1]), B_vect.dtype)

    # This is in the space with a standard basis along the "l", "R" and "B" axes
    jacob[1, 2] = -B_len / R_len
    jacob[2, 0] = l_comp
    jacob[2, 1] = R_comp

    # Transform the Jacobian to main space
    xform = numpy.stack((
            l_vect / l_len,
            R_vect / R_len,
            B_vect / B_len
        )).T
    xform_inv = numpy.linalg.inv(xform)
    return numpy.matmul(xform, numpy.matmul(jacob.T, xform_inv)).T

def calc_emi_dif(tgt_pt, src_pt, src_dir, coef=1):
    """Calculate the magnetic field at specific point, induced by movement of a charged particle.

    Returns: [
            <B-vect>,       # Magnetic field vector at the point
            <B-jacob>,      # Jacobian matrix of the magnetic field
        ]
    """
    emi_params = [
            numpy.zeros(tgt_pt.shape[-1], tgt_pt.dtype),
            numpy.zeros((tgt_pt.shape[-1], tgt_pt.shape[-1]), tgt_pt.dtype)
        ]

    # 'r' vector
    r = tgt_pt - src_pt

    src_dir_len2 = src_dir.dot(src_dir)
    if not src_dir_len2:
        return emi_params   # Zero length, return zero EMI params

    # Vector projections of "r" in the direction of "src_dir"
    l = src_dir.dot(src_dir.dot(r) / src_dir_len2)
    R = r - l

    r_len = numpy.sqrt(r.dot(r))
    if not r_len:
        return None     # Target point coincides with "src_pt"

    # Calculate the differential Biot–Savart law (https://en.wikipedia.org/wiki/Biot–Savart_law):
    # dl x r / r^3
    B = numpy.cross(src_dir, r) / r_len ** 3

    # Scale by a coefficient, like current, magnetic constant and 1/(4*pi)
    B *= coef

    emi_params[0] = B

    # Calculate the partial derivatives from Biot–Savart law "R/sqrt(l^2 + R^2)^3" (see calc_emi())
    # along "l" and "R" axes.

    # Gradient component along 'l':
    # Use derivative calculator https://www.derivative-calculator.net/ (substitute l with x):
    #   input: R / sqrt(x^2 + R^2)^3, result: -3Rx / (x^2 + R^2)^(5/2)
    # Substitute back x to l, then sqrt(l^2 + R^2) to r:
    #   result: -3 * R * l / r^5
    R_len2 = R.dot(R)
    l_len2 = l.dot(l)
    R_len = numpy.sqrt(R_len2)
    l_len = numpy.sqrt(l_len2)
    if l.dot(src_dir) < 0:
        l_len = -l_len

    l_comp = -3 * R_len * l_len / r_len ** 5

    # Gradient component along 'R':
    # Use derivative calculator https://www.derivative-calculator.net/ (substitute R with x):
    #   input: x / sqrt(x^2 + l^2)^3, result: - (2x^2 - l^2) / (x^2 + l^2)^(5/2)
    # Substitute back x to R, then sqrt(l^2 + R^2) to r:
    #   result: (l^2 - 2R^2) / r^5

    R_comp = (l_len2 - 2 * R_len2) / r_len ** 5

    l_comp *= coef
    R_comp *= coef

    # Combine l_comp and R_comp into a Jacobian matrix
    emi_params[1] = build_jacobian(l_comp, R_comp, src_dir, R, B)

    return emi_params

def calc_emi(tgt_pt, src_pt, src_dir, coef=1):
    """Calculate the magnetic field at specific point, induced by electric current flowing along
    a line segment.

    Returns: [
            <B-vect>,       # Magnetic field vector at the point
            <B-jacob>,      # Jacobian matrix of the magnetic field
        ]
    """
    emi_params = [
            numpy.zeros(tgt_pt.shape[-1], tgt_pt.dtype),
            numpy.zeros((tgt_pt.shape[-1], tgt_pt.shape[-1]), tgt_pt.dtype)
        ]

    # Start and end 'r' vectors
    r0 = tgt_pt - src_pt
    r1 = r0 - src_dir

    # Calculate the integral from Biot–Savart law (https://en.wikipedia.org/wiki/Biot–Savart_law):
    #   dl x r / sqrt(l^2 + R^2)^3
    #
    # The "l" origin is selected at the closest point to the target to simplify calculations.
    # Thus "r = l^2 + R^2" and "|dl x r| = |dl|.R", where R is distance between the target and origin.
    #
    # Use integral calculator https://www.integral-calculator.com/ (substitute l with x):
    #   int[ R/sqrt(x^2 + R^2)^3 dx ] = x / (R * sqrt(x^2 + R^2)) + C
    src_dir_len2 = src_dir.dot(src_dir)
    if not src_dir_len2:
        return emi_params   # Zero length, return zero EMI params

    # Vector projections of "r0" and "r1" in the direction of "src_dir"
    # The '-' is to set the origin at the projected point, instead of at src_pt
    l0 = -src_dir.dot(src_dir.dot(r0) / src_dir_len2)
    l1 = l0 + src_dir
    R = l0 + r0

    #
    # Integral at the start of interval
    #
    # Start with l0 x R to get a direction vector with length of |l0|.|R|
    vect0 = numpy.cross(l0, R)

    # Divide by 'r0'
    r0_len = numpy.sqrt(r0.dot(r0))
    if not r0_len:
        return None     # Target point coincides with "src_pt"
    vect0 /= r0_len

    #
    # Integral at the end of interval
    #
    # Start with l1 x R to get a direction vector with length of |l1|.|R|
    vect1 = numpy.cross(l1, R)

    # Divide by 'r1'
    r1_len = numpy.sqrt(r1.dot(r1))
    if not r1_len:
        return None     # Target point coincides with "src_pt + src_dir"
    vect1 /= r1_len

    #
    # Combine both integrals
    #
    # Divide result by 'R^2', resulting:
    # |l|.|R| / |r| / |R|^2 = |l| / (|R|.|r|)
    R_len2 = R.dot(R)
    if not R_len2:
        return None     # Target point lies on the source line

    B = (vect1 - vect0) / R_len2

    # Scale by a coefficient, like current, magnetic constant and 1/(4*pi)
    B *= coef

    emi_params[0] = B

    # Calculate the partial derivatives from Biot–Savart law "R/sqrt(l^2 + R^2)^3" (see above)
    # along "l" and "R" axes, then integrate each of them along 'l'.
    #
    # The individual gradient vector components are the values of these integrals. The 'l'
    # component is along the 'src_dir' direction and 'R' component is to the direction of its
    # perpendicular through 'tgt_pt'.

    # Gradient component along 'l' (substitute l with x):
    #   int[ dF(x)/dx dx] = F(x) => gradBx = R/sqrt(x^2 + R^2)^3 - R/sqrt(x^2 + R^2)^3 + C
    # Finally:
    #   R * (1/r1^3 - 1/r0^3)
    R_len = numpy.sqrt(R_len2)

    l_comp = R_len * ( 1 / r1_len ** 3 - 1 / r0_len ** 3)

    # Gradient component along 'R':
    # Use derivative calculator https://www.derivative-calculator.net/ (substitute R with x):
    #   input: x / sqrt(x^2 + l^2)^3, result: - (2x^2 - l^2) / (x^2 + l^2)^(5/2)
    # Substitute back x to R, then l with x:
    #   result: (x^2 - 2R^2) / sqrt(x^2 + R^2)^5
    # Use integral calculator https://www.integral-calculator.com/ (back R and x):
    #   input: (x^2 - 2R^2) / sqrt(x^2 + R^2)^5, result: - (x^3 + 2xR^2) / ( R^2(x^2 + R^2)^(3/2) ) + C
    # Simplify (substitute back x to l):
    #   - (l^3 + 2*l*R^2) / ( R^2(l^2 + R^2)^(3/2) ) = - l(l^2 + R^2 + R^2) / ( R^2 * r^3 ) =
    #   = - l(r^2 + R^2) / ( R^2 * r^3 )
    # Finally:
    #   - l1(r1^2 + R^2) / ( R^2 * r1^3 ) + l1(r1^2 + R^2) / ( R^2 * r0^3 )
    l0_len = numpy.sqrt(l0.dot(l0))
    if l0.dot(src_dir) < 0:
        l0_len = -l0_len
    l1_len = numpy.sqrt(l1.dot(l1))
    if l1.dot(src_dir) < 0:
        l1_len = -l1_len

    R_comp = -l1_len*(r1_len ** 2 + R_len2) / (R_len2 * r1_len ** 3)
    R_comp -= -l0_len*(r0_len ** 2 + R_len2) / (R_len2 * r0_len ** 3)

    # The '-' is to flip direction to point toward field magnitude increase
    l_comp *= -coef
    R_comp *= coef

    # Combine l_comp and R_comp into a Jacobian matrix
    emi_params[1] = build_jacobian(l_comp, R_comp, src_dir, R, B)

    return emi_params

def generate_dr_dI(B, jacob):
    """Calculate I.dr/dI (a vector to be multiplied by dI/I)

    The dr/dI is an imaginary change in the point position (dr), that would provoke the same EMI effect as a
    change in the electric current (dI). The movement is toward the gradient of the magnitude of the field,
    i.e. the closest point with the same |B| as at the current point, but after the current is changed.
    The cross product between this vector and B is proportional to the EMF caused by the dI.
    """
    # dB/dI = gradB.dr/dI = |B|/I  =>  I.dr/dI = |B|.gradB/gradB^2 = |B|.B/|B|.Jacob / (B/|B|.Jacob)^2
    # =>  I.dr/dI = |B|^2.B.Jacob / (B.Jacob)^2
    B_jacob = numpy.matmul(B, jacob)
    dr_dI = B_jacob / (B_jacob * B_jacob).sum(-1)
    dr_dI *= (B * B).sum(-1)
    return dr_dI

def generate_gradB(B, jacob):
    """Calculate gradient of the magnitude of 'B' by using Jacobian matrix"""
    Blen = numpy.sqrt((B * B).sum(-1))
    gradB = numpy.matmul(B / Blen, jacob)
    return gradB

def calc_all_emis(tgt_pts, src_lines, coef=1):
    """Calculate EMI parameters B and dB at each of the target points

    Requirements:
        tgt_pts and src_lines must be numpy.array
    Returns: {
            'pt': <pt-vect>,        # Point from tgt_pts, where the field is calculated
            'B': <B-vect>,          # Magnetic field vector
            'jacob': <matrix>,      # Jacobian matrix
            'gradB': <gradB-vect>   # Gradient of the magnetic field magnitude (derived from 'jacob')
        }
    """
    if not tgt_pts.dtype.fields:
        # Option 1: tgt_pts contains points
        # The result is of the same shape as tgt_pts, but the last dimention is moved into 'pt' field
        emi_pars_dt = numpy.dtype([
            ('pt', tgt_pts.dtype, (3,)),
            ('B', tgt_pts.dtype, (3,)),
            ('jacob', tgt_pts.dtype, (3,3)),
            ('gradB', tgt_pts.dtype, (3,)),
            ])
        emi_params = numpy.empty(tgt_pts.shape[:-1], emi_pars_dt)
        emi_params['pt'] = tgt_pts
        emi_params['B'] = numpy.nan
        emi_params['jacob'] = numpy.nan
        emi_params['gradB'] = numpy.nan
    else:
        # Option 2: tgt_pts contains EMI params structure
        # The result is added to existing field (superposition principle)
        emi_params = tgt_pts

    # Allow multiple source lines
    src_pts = src_lines[...,:-1,:]
    src_dirs = src_lines[...,1:,:] - src_lines[...,:-1,:]
    src_pts = src_pts.reshape((-1, src_pts.shape[-1]))
    src_dirs = src_dirs.reshape((-1, src_dirs.shape[-1]))

    emi_it = numpy.nditer(emi_params, op_flags=[['readwrite']])
    for emi_pars in emi_it:
        for src_pt, src_dir in zip(src_pts, src_dirs):
            emi = calc_emi(emi_pars['pt'], src_pt, src_dir, coef)
            if emi is not None:     # Ignore point collinear to the src-line
                if numpy.isnan(emi_pars['B']).all():
                    emi_pars['B'] = 0
                if numpy.isnan(emi_pars['jacob']).all():
                    emi_pars['jacob'] = 0

                emi_pars['B'] += emi[0]
                emi_pars['jacob'] += emi[1]

    # Obtain 'gradB' from Jacobian matrix
    emi_it = numpy.nditer(emi_params, op_flags=[['readwrite']])
    for emi_pars in emi_it:
        emi_pars['gradB'] = generate_gradB(emi_pars['B'], emi_pars['jacob'])

    return emi_params
