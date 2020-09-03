'''Electromagnetic self- and mutual- inductance calculator

Requires: numpy
'''
import sys
import numpy
import emi_calc

NUM_SPLITS = 0

def split_lines(lines):
    """Split each line in 2 half-lines"""
    # Middle points
    mpoints = (lines[...,1:,:] + lines[...,:-1,:]) / 2
    # Allocate result storage
    shape = [ *lines.shape ]
    shape[-2] += mpoints.shape[-2]
    res = numpy.empty(shape, dtype=lines.dtype)
    # Interleave 'lines' and 'mpoints'
    for lineno in range(0, lines.shape[-2]):
        res[...,lineno * 2,:] = lines[...,lineno,:]
    for lineno in range(0, mpoints.shape[-2]):
        res[...,lineno * 2 + 1,:] = mpoints[...,lineno,:]
    return res

def adjust_colinears(pts, lines, shift=sys.float_info.epsilon):
    """Shift all points that are colinear to any of the source lines"""
    idx_list = []
    ln_dirs = lines[...,1:,:] - lines[...,:-1,:]
    for idx, pt in enumerate(pts.reshape(-1, pts.shape[-1])):
        pt_dirs = pt - lines[...,:-1,:]

        # Locate points collinear to any of the lines
        # Hint: '== .0' could be replaced by 'abs(...) < sys.float_info.epsilon'
        # See also 'float_info.min' vs. 'float_info.epsilon'
        collinears = (numpy.cross(pt_dirs, ln_dirs) == 0.).all(-1)
        if collinears.any():
            # Extract collinear directions and select a perpendicular vector
            col_dirs = ln_dirs[collinears]
            # Try in YZ plane
            adj_dirs = numpy.cross(col_dirs, [1,0,0])
            # Try in ZX plane (where col_dirs is parallel to X)
            mask = (adj_dirs == 0.).all(-1)
            adj_dirs[mask] = numpy.cross(col_dirs[mask], [0,1,0])

            # Scale 'shift' to ensure the float addition will actually change the 'pt'
            # components. Note that "n + epsilon == n", when "n > 1."
            max_comp = max(pt.max(), -pt.min())
            abs_shift = shift * max_comp if max_comp > sys.float_info.epsilon else shift

            # Set all lengths to 'shift'
            adj_dirs *= (abs_shift / numpy.sqrt((adj_dirs * adj_dirs).sum(-1)))[:,numpy.newaxis]

            # Adjust with the sum of all directions
            pt += adj_dirs.sum(0)
            idx_list.append(idx)

    return pts, idx_list

def inductance(src_lines, tgt_lines=None, tgt_coef=0):
    """Inductance calculator. tgt_lines=None: self-inductance"""
    if tgt_lines is None:
        # Self-inductance
        tgt_lines = src_lines

    # Split lines to increase precision (TODO: Must be dynamic)
    for _ in range(NUM_SPLITS):
        tgt_lines = split_lines(tgt_lines)

    # Target points at the middle of each line and the direction vectors
    tgt_pts = (tgt_lines[...,1:,:] + tgt_lines[...,:-1,:]) / 2
    tgt_dirs = tgt_lines[...,1:,:] - tgt_lines[...,:-1,:]

    #HACK: Self-inductance: Avoid EMI failures by shifting the points collinear
    # to any of the source lines
    for _ in range(100):
        tgt_pts, idxs = adjust_colinears(tgt_pts, src_lines)
        if not idxs:
            break
        print('Points adjusted to avoid EMI failure:', idxs)

    # Get EMI parameters and EMFs
    emi_params = emi_calc.calc_all_emis(tgt_pts, src_lines)

    # EMF because of a change in the field intensity
    emf_vecs = numpy.cross(emi_params['dr_dI'], emi_params['B'])

    # EMF because of a current accross magnetic field (also Hall effect)
    if tgt_coef:
        # In self-inductance case, the dot-product below makes this pointless
        emf_vecs += numpy.cross(tgt_dirs, emi_params['B']) * tgt_coef

    # Dot product (integral along target line)
    inductance = (emf_vecs * tgt_dirs).sum(-1)
    return inductance, emi_params, emf_vecs
