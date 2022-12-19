# This file is part of colorcheck
# Copyright (C) 2022 Andreas Kvas
# See LICENSE for detailed licensing information.

import matplotlib.colors
import numpy as np
import os


def writecpt(file_name, cmap, header=None):
    """
    Write a colormap to a cpt file, for use with GMT.

    Parameters
    ----------
    file_name : str
        Name of the file to write to.
    cmap : colormap_like
        Instance of a colormap-like class.
    header : str or None
        If given, write header lines to the file.
    """
    if isinstance(cmap, matplotlib.colors.Colormap):
        x = np.linspace(0, 1, cmap.N)
        rgba = cmap(x)
        cmap = rgba[:, 0:3]

    cmap = (cmap * 255).astype(int)

    with open(file_name, 'w+') as f:
        if header is not None:
            for line in header.split(os.linesep):
                f.write('# ' + line + os.linesep)
        f.write('# COLOR_MODEL = RGB' + os.linesep)
        for k in range(x.size - 1):
            f.write('{0:7.5e} {1:3d} {2:3d} {3:3d} '.format(x[k], cmap[k, 0], cmap[k, 1], cmap[k, 2]))
            f.write('{0:7.5e} {1:3d} {2:3d} {3:3d}'.format(x[k + 1], cmap[k + 1, 0], cmap[k + 1, 1], cmap[k + 1, 2]) + os.linesep)
