# This file is part of colorcheck
# Copyright (C) 2023 - 2024 Andreas Kvas
# See LICENSE for detailed licensing information.

import numpy as np
import colorcheck


def flip_lightness(img, preserve_colors=False, reduce_darkness=0.1, darkness_threshold=6):
    """Change the lightness of an image with, or without preseving colors. The function can
    be used to convertd between dark and light themes for figures.

    The method behind the function is inspired by Crameri, Fabio. (2020). deLight (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.3634423

    :param img: Image represneted as rgb(a) array
    :type img: ndarray
    :param preserve_colors: Preserve foreground colors, defaults to False.
    :type preserve_colors: bool(, optional)
    :param reduce_darkness: Increase the overall figure lightness (0 - 1), defaults to 0.1
    :type reduce_darkness: float(, optional)
    :param darkness_threshold: If reduce_darkness is non-zero, this parameter in the range (0 - 100) governs which pixels are presumed dark, defaults to 6.
    :type darkness_threshold: float
    :return im_modified: Modified image as rgb(a) array.
    :rtype ndarray:
    """
    if preserve_colors:
        temp = np.abs(img[:, :, 1] - img[:, :, 0]) + np.abs(img[:, :, 0] - img[:, :, 2]) + np.abs(img[:, :, 2] - img[:, :, 1])
        background = temp < 0.3
        im_orig = img.copy()

    has_alpha = img.shape[-1] > 3
    if has_alpha:
        lab = colorcheck.cmap._rgb2lab_converter(img[:, :, 0:3])
        alpha = img[:, :, -1]
    else:
        lab = colorcheck.cmap._rgb2lab_converter(img)

    lab[:, :, 0] = 100 - lab[:, :, 0]
    if reduce_darkness > 0:
        L = lab[:, :, 0] > darkness_threshold
        lab[L, 0] += (100 - lab[L, 0]) * reduce_darkness

    rgb = np.clip(colorcheck.cmap._lab2rgb_converter(lab), 0, 1)

    if has_alpha:
        rgb = np.concatenate((rgb, alpha[:, :, np.newaxis]), axis=-1)

    if preserve_colors:
        rgb[~background] = im_orig[~background]

    return rgb


def hex2rgb(hex_color):
    """Convert hex color(s) to RGB in the range 0 - 1.

    :param hex_color: hex color string or list of hex color strings
    :type img: str or list
    :return rgb_color: RGB color tuple in the range 0 - 1 or list of RGB color tuples
    :rtype tuple or list of tuples:
    """
    if isinstance(hex_color, str):
        return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    else:
        return [hex2rgb(hk) for hk in hex_color]
