# This file is part of colorcheck
# Copyright (C) 2022 Andreas Kvas
# See LICENSE for detailed licensing information.

import numpy as np
import colorspacious
import matplotlib
import scipy.interpolate
import scipy.ndimage


_rgb2lab_converter = colorspacious.cspace_converter('sRGB1', 'CAM02-UCS')
_lab2rgb_converter = colorspacious.cspace_converter('CAM02-UCS', 'sRGB1')
_rgb2cielab_converter = colorspacious.cspace_converter('sRGB1', 'CIELab')


def _rgb2lab(rgb):
    return _rgb2lab_converter(rgb[np.newaxis, :, :3])[0, :, :]


def _lab2rgb(lab):
    return _lab2rgb_converter(lab[np.newaxis, :, :3])[0, :, :]


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
    lab = _rgb2lab(colors[:, 0:3])

    return lab[:, 0], colors


def perceptual_gradient(cmap, samples=256, uniform_space=colorspacious.CAM02SCD):
    """
    Compute the perceptual gradient :math:`\Delta E` between consecutive colors of the colormap.

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

    dE = colorspacious.deltaE(colors[0:-1, 0:3], colors[1:, 0:3], input_space='sRGB1', uniform_space=uniform_space)

    return dE, np.cumsum(dE)


def ciede2000(cmap, samples=256, kl=1, kc=1, kh=1):
    """
    Compute the perceptual gradient :math:`\Delta E_\text{CIEDE2000}` between consecutive colors of the colormap.

    Equations from http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE2000.html.

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
    lab = _rgb2cielab_converter(colors[:, 0:3])

    L_prime = lab[0:-1, 0] * 0.5 + lab[1:, 0] * 0.5
    C = np.sqrt(lab[:, 1]**2 + lab[:, 2]**2)
    C_bar = C[0:-1] * 0.5 + C[1:] * 0.5
    G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))

    a1_prime = lab[0:-1, 1] * (1 + G)
    a2_prime = lab[1:, 1] * (1 + G)

    C1_prime = np.sqrt(a1_prime**2 + lab[0:-1, 2]**2)
    C2_prime = np.sqrt(a2_prime**2 + lab[1:, 2]**2)
    C_prime_bar = C1_prime * 0.5 + C2_prime * 0.5

    h1 = np.arctan2(lab[0:-1, 2], a1_prime)
    h2 = np.arctan2(lab[1:, 2], a2_prime)
    h1[h1 < 0] += 2 * np.pi
    h2[h2 < 0] += 2 * np.pi

    H_prime_bar = h1 * 0.5 + h2 * 0.5
    H_prime_bar[np.abs(h1 - h2) > np.pi] += np.pi

    T = 1 - 0.17 * np.cos(H_prime_bar - np.deg2rad(30)) + 0.24 * np.cos(2 * H_prime_bar) + 0.32 * np.cos(3 * H_prime_bar + np.deg2rad(6)) - 0.20 * np.cos(4 * H_prime_bar - np.deg2rad(63))

    delta_h_prime = h2 - h1 - 2 * np.pi
    delta_h_prime[np.abs(h1 - h2) <= np.pi] += 2 * np.pi
    delta_h_prime[np.logical_and(np.abs(h1 - h2) > np.pi, h2 <= h1)] += 4 * np.pi

    delta_L_prime = lab[1:, 0] - lab[0:-1, 0]
    delta_C_prime = C2_prime - C1_prime
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(delta_h_prime * 0.5)

    S_L = 1 + (0.015 * (L_prime - 50)**2) / np.sqrt(20 + (L_prime - 50)**2)
    S_C = 1 + 0.045 * C_prime_bar
    S_H = 1 + 0.015 * C_prime_bar * T

    delta_phi = np.deg2rad(30 * np.exp(-(((np.rad2deg(H_prime_bar) - 275) / 25)**2)))

    R_C = 2 * np.sqrt(C_prime_bar**7 / (C_prime_bar**7 + 25**7))
    R_T = -R_C * np.sin(2 * delta_phi)

    dE = np.sqrt((delta_L_prime / (kl * S_L))**2 +
                 + (delta_C_prime / (kc * S_C))**2 +
                 + (delta_H_prime / (kh * S_H))**2 +
                 + R_T * (delta_C_prime / (kc * S_C)) * (delta_H_prime / (kh * S_H)))

    return dE, np.cumsum(dE)


def replace_lightness(cmap, lightness_curve):

    x = np.linspace(0, 1, cmap.N)
    rgba = cmap(x)
    lab = _rgb2lab(rgba[:, 0:3])
    lab[:, 0] = lightness_curve(x)
    rgb_new = rgba.copy()
    rgb_new[:, 0:3] = _lab2rgb(lab)

    valid_colors = np.all(np.logical_and(rgb_new >= 0, rgb_new <= 1), axis=1)

    hue = np.arctan2(lab[:, 2], lab[:, 1])
    chroma = np.sqrt(lab[:, 2]**2 + lab[:, 1]**2)
    for k in range(x.size):
        if valid_colors[k]:
            continue

        c = np.linspace(0, chroma[k])
        candidates = np.empty((c.size, 3))
        candidates[:, 0] = lab[k, 0]
        candidates[:, 1] = c * np.cos(hue[k])
        candidates[:, 2] = c * np.sin(hue[k])

        rgb_tmp = _lab2rgb(candidates)
        valid_candidates = np.all(np.logical_and(rgb_tmp >= 0, rgb_tmp <= 1), axis=1)
        diff = candidates - lab[k:k + 1, :]
        d = np.sqrt(np.sum(diff**2, axis=1))

        lab[k, :] = candidates[valid_candidates, :][np.argmin(d[valid_candidates])]

    rgb_new = rgba.copy()
    rgb_new[:, 0:3] = _lab2rgb(lab)

    if isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        return matplotlib.colors.LinearSegmentedColormap.from_list(cmap.name + '_lc', rgb_new)
    elif isinstance(cmap, matplotlib.colors.ListedColormap):
        return matplotlib.colors.ListedColormap(rgb_new, cmap.name + '_lc')


def equalize_cmap(cmap, metric='lightness'):
    """
    Equalize the percetual gradient of a colormap.

    Parameters
    ----------
    cmap : colormap-like
        Colormap-like object.
    metric : str
        The metric used to linearize the perception. One of ('lightness', 'CIE76').

    Returns
    -------
    cmap_equalized : colormap-like
        The equalized colormap.
    """
    x = np.linspace(0, 1, cmap.N)
    colors = cmap(x)
    rgb = colors[np.newaxis, :, :3]
    lab = colorspacious.cspace_converter("sRGB1", "CAM02-UCS")(rgb)[0, :, :]

    if metric == 'lightness':
        dE = np.cumsum(np.abs(np.gradient(lab[:, 0])))
    elif metric == 'CIE76':
        dE = np.cumsum(np.sqrt(np.sum(np.gradient(lab, axis=0)**2, axis=1)))
    else:
        raise ValueError('Metric must be one of "lightness" or "CIE76"')

    x_interp = np.interp(dE, np.linspace(dE[0], dE[-1], cmap.N), x)

    if cmap.name.endswith('_r'):
        output_name = cmap.name[0:-2] + '_equalized_r'
    else:
        output_name = cmap.name + '_equalized'

    rgb_equalized = [(xk, ck) for xk, ck in zip(x_interp, colors)]

    return matplotlib.colors.LinearSegmentedColormap.from_list(output_name, rgb_equalized)


def cmapshow(cmap, ax=None, width=640, height=80, orientation='horizontal'):
    """
    Convenience function for displaying colormaps in an existing axes instance.

    Parameters
    ----------
    cmap : colormap-like
        Colormap-like object.
    ax : Axes object or None
        Axes into which the colormap is drawn. If None, the currently active axes are used.
    width : int
        Width of the image array in pixels.
    height : int
        Height of the image array in pixels.
    orientation : str
        Orientiation of the colormap gradient. One of 'horizonal' (default) or 'vertical'.

    Returns
    -------
    im : mappable
        The mappable returned by imshow.
    """
    if ax is None:
        ax = matplotlib.pyplot.gca()

    img = cmap_rgba(cmap, width, height, orientation)
    im = ax.imshow(img)

    return im


class ColormapLightness:
    """
    Class representation for pre-defined lightess curves in colormaps.
    """
    def __call__(self, x):
        """
        Return the lightness value for a given color index.

        Parameters
        ----------
        x : float
            Color index in the range [0, 1].
        """
        return self._lightness_map(x)


class SequentialLightness(ColormapLightness):
    """Linear lightness curve for sequential colormaps."""
    def __init__(self, start_lightness=10, end_lightness=90):
        self._lightness_map = scipy.interpolate.interp1d((0, 1), (start_lightness, end_lightness), bounds_error=True)


class MultiSequentialLightness(ColormapLightness):
    """Piecewise linear lightness curve for multi-sequential colormaps."""
    def __init__(self, segments, start_lightness=10, end_lightness=90):
        self._backend = SequentialLightness(start_lightness, end_lightness)
        self._segments = segments

    def __call__(self, x):
        arg = np.mod(x * self._segments, 1)
        last_segment = x >= (self._segments - 1) / self._segments
        arg[last_segment] = (x[last_segment] - (self._segments - 1) / self._segments) * self._segments
        return self._backend(arg)


class DivergingLightness(ColormapLightness):
    """Piecewise linear (optionally with a smoothed center) lightness curve for diverging colormaps."""
    def __init__(self, edge_lightness=10, center_lightness=90, smooth_center=False):

        if smooth_center:
            s1 = np.linspace(edge_lightness, center_lightness, 501)
            s2 = np.linspace(s1[-2], edge_lightness, 500)
            s_concat = np.concatenate((s1, s2))
            s = scipy.ndimage.gaussian_filter1d(s_concat, s_concat.size / 20)
            s_concat[int(s_concat.size / 4):int(3 / 4 * s_concat.size)] = s[int(s_concat.size / 4):int(3 / 4 * s_concat.size)]
            self._lightness_map = scipy.interpolate.interp1d(np.linspace(0, 1, s_concat.size), s_concat, bounds_error=True)
        else:
            self._lightness_map = scipy.interpolate.interp1d((0, 0.5, 1), (edge_lightness, center_lightness, edge_lightness), bounds_error=True)
