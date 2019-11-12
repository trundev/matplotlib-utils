"""Quaternion multiplication (3D vector rotation) experiments

  https://www.mathworks.com/help/aeroblks/quaternionmultiplication.html
  https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
  https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternion_as_rotations
"""
import sys
import math
import numpy

QUAT_DT=numpy.float64

def normalize(quat):
    """Normalize (convert to unit vector/quaternion)

    Returns None for the zero vectors
    """
    q = numpy.array(quat)
    len2 = q.dot(q)
    return q / len2 ** .5 if len2 else None

def quaternion_multiply(quat1, quat2):
    """Hamilton product between quaternions"""
    if True:
        # Optimization using numpy cross-product
        # prod = (w1, V1)(w2, V2) = (w1*w2 - V1.V2, w1*V2 + w2*V1 + V1xV2)
        w1 = quat1[0]
        V1 = numpy.array(quat1[1:], quat1.dtype)
        w2 = quat2[0]
        V2 = numpy.array(quat2[1:], quat2.dtype)
        prod = numpy.zeros_like(quat1)
        prod[0] = w1*w2 - V1.dot(V2)
        prod[1:] = w1*V2 + w2*V1 + numpy.cross(V1, V2)
        return prod

    # Basic Hamilton formula
    a1, b1, c1, d1 = quat1
    a2, b2, c2, d2 = quat2
    return numpy.array([a1*a2 - b1*b2 - c1*c2 - d1*d2,
                        a1*b2 + b1*a2 + c1*d2 - d1*c2,
                        a1*c2 - b1*d2 + c1*a2 + d1*b2,
                        a1*d2 + b1*c2 - c1*b2 + d1*a2],
                       dtype=QUAT_DT)

def quaternion_to_str(quat):
    """Compact string representation"""
    if quat is None:
        return '<none>'
    return '[' + ', '.join( ['%.3f'%x for x in quat]) + ']'

def rotate_vector(quat, vector):
    """Rotate vector using qaternion"""
    quat = normalize(quat)
    res = quaternion_multiply(quat, pad_vector_to_quaternion(vector))
    quat[1:] = -quat[1:]
    res = quaternion_multiply(res, quat)
    return res[1:]

def quaternion_rot_angle(quat):
    """Angle of rotation: 2*atan2(|(x,y,z)|, w)

    Result in range 0-2*pi (0-360 degrees)"""
    q = numpy.array(quat[1:], dtype=quat.dtype)
    return 2 * math.atan2(q.dot(q)**.5, quat[0])

def pad_vector_to_quaternion(vector):
    """Pad with zeros to make quaternion"""
    return numpy.array([ *[0]*(4 - len(vector)), *vector ])

def main(argv):
    quats = []
    for arg in argv:
        quats.append( [float(x) for x in arg.split(',')] )
    print('Arguments:\n  ' + '\n  '.join( [str(x) for x in quats]))

    print('Result:')
    result = None
    for quat in quats:
        # Check if this is quaternion
        is_quat = True
        if len(quat) < 4:
            is_quat = False
            # Pad with zeros from start
            quat = pad_vector_to_quaternion(quat)

        if is_quat:
            print('  Normalizing quaternion:', quaternion_to_str(quat), ':')
            quat = normalize(quat)
            print('    * rotation angle (+/-):', math.degrees(quaternion_rot_angle(quat)))

        if result is None:
            result = quat
        else:
            print('  Multiplying:', quaternion_to_str(result), quaternion_to_str(quat), ':')
            result = quaternion_multiply(result, quat)

        print('    ', quaternion_to_str(result))

    # Quaternion and 3D vector: Do rotation
    if len(quats) == 2 and len(quats[0]) == 4 and len(quats[1]) == 3:
        vect = rotate_vector(quats[0], quats[1])
        print('Rotated vector:', quaternion_to_str(vect))

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
