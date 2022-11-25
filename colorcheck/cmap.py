# This file is part of colorcheck
# Copyright (C) 2022 Andreas Kvas
# See LICENSE for detailed licensing information.

import numpy as np
import colorspacious
import matplotlib


def is_colormap_like(cmap):
    """Return whether an object is colormap-like."""
    if isinstance(cmap, matplotlib.colors.Colormap):
        return True

    try:
        cmap(0.5)
    except TypeError:
        return False
    else: 
        rgba = cmap((0.0, 0.5, 1.0)) 
        try:
            return np.all(rgba >= 0) and np.all(rgba <= 1)
        except TypeError:
            return False 
    

def cmap_rgba(cmap, width=640, height=80, orientation='horizontal'):
    """
    Create an image-like rgb(a) array from a Colormap instance.

    Parameters
    ----------
    cmap : colormap-like
        Colormap-like object. 
    width : int
        Width of the image array in pixels.
    height : int
        Height of the image array in pixels.
    orientation : str
        Orientiation of the colormap gradient. One of 'horizonal' (default) or 'vertical'.

    Returns
    -------
    img_array : ndarray(width, height, color.dimension)
        Array representation of the colormap of the given size and orientation.    
    """
    x = np.linspace(0, 1, width)
    arr = np.tile(cmap(x)[:, np.newaxis, :], (1, height, 1))
    if orientation == 'horizontal':
        return np.swapaxes(arr, 0, 1)

    return arr


def lightness(cmap, samples=256):
    """
    Compute the lightness for each color in the colormap. The computed lightness is based on 
    the CIECAM02-UCS color model. 

    Parameters
    ----------
    cmap : colormap-like
        Colormap-like object.
    samples : int
        Number of colors in the color gradient.

    Returns
    -------
    lightness : ndarray(samples)
        Lightness values for each color in the colormap.
    colors : ndarray(samples, color dimension)
        RGB(A) values of the used colors. This is primarily useful for visualization purposes.
    """
    x = np.linspace(0, 1, samples)
    colors = cmap(x)
    rgb = colors[np.newaxis, :, :3]
    lab = colorspacious.cspace_converter("sRGB1", "CAM02-UCS")(rgb)[0, :, :]

    return lab[:, 0], colors


def lightness_gradient(cmap, samples=256):
    """
    Compute the lightness gradient :math:`\Delta E_\text{CIEDE2000}` between consecutive colors of the colormap.

    Parameters
    ----------
    cmap : colormap-like
        Colormap-like object.
    samples : int
        Number of colors in the color gradient.

    Returns
    -------
    lightness_gradient : ndarray(samples)
        Lightness gradient values between consecutive colors in the colormap.
    lightness_gradient_cumulative : ndarray(samples)
       Cumulative lightness gradient values between consecutive colors in the colormap.
    """
    x = np.linspace(0, 1, samples)
    colors = cmap(x)
    rgb = colors[np.newaxis, :, :3]
    lab = colorspacious.cspace_converter("sRGB1", "CAM02-UCS")(rgb)[0, :, :]

    dE = np.sum(np.gradient(lab, axis=0)**2, axis=1)

    return dE, np.cumsum(dE) 
