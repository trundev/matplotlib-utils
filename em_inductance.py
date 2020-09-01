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
