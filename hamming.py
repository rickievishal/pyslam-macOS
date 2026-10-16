"""Small NumPy fallback for the optional native hamming extension."""

import numpy as np


def hamming_distance(a, b):
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


def hamming_distances(a, b):
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    if a.ndim == 1:
        return np.unpackbits(np.bitwise_xor(b, a), axis=1).sum(axis=1).astype(np.uint16)
    return np.unpackbits(np.bitwise_xor(a, b), axis=1).sum(axis=1).astype(np.uint16)
