import os

import tdpy


def retr_pathgaia(pathbase=None):
    """Return normalized base/data/image paths for Troia Gaia side scripts."""

    if pathbase is None:
        pathbase = tdpy.retr_pathbase('troia')
    else:
        pathbase = tdpy.ensr_path(pathbase)

    return {
        'pathbase': pathbase,
        'pathdata': tdpy.ensr_path(os.path.join(pathbase, 'data')),
        'pathimag': tdpy.ensr_path(os.path.join(pathbase, 'imag')),
    }