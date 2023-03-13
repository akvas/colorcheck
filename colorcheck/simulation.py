# This file is part of colorcheck
# Copyright (C) 2022 - 2023 Andreas Kvas
# See LICENSE for detailed licensing information.

import daltonlens
import numpy as np
import matplotlib.colors
import colorspacious


def simulate_cvd(img, cvd_type, severity):
    """Simulate various types of color vision deficiency (CVD) for a given image array.

    The simulation approach is chosen the same way as in the `daltonlens` package:

    - Tritanopia/Tritanomaly: Brettel 1997
    - Deuteranopia/Protanopia: Viennot 1999
    - Deuteranomaly/Protanomaly: Machado 2009

    :param img: Input image represented as rgb(a) array
    :type img: ndarray
    :param cvd_type: One of 'deutan', 'protan', 'tritan', 'mono'
    :rype cvd_type: str
    :param severity: CVD severity in percent (0 - 100)
    :type severity: float
    :return: Image  with simulated CVD as rgb(a) array
    :rtype: ndarray

    :raises ValueError: If an unkown `cvd_type` is passed.
    """
    if img.ndim == 2:
        img_cvd = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    else:
        img_cvd = img.copy()

    if cvd_type == 'tritan':
        simulator = daltonlens.simulate.Simulator_Brettel1997(daltonlens.convert.LMSModel_sRGB_SmithPokorny75())
        deficiency = daltonlens.simulate.Deficiency.TRITAN
    elif cvd_type in ('deutan', 'protan'):
        if severity < 100:
            simulator = daltonlens.simulate.Simulator_Machado2009()
        else:
            simulator = daltonlens.simulate.Simulator_Vienot1999(daltonlens.convert.LMSModel_sRGB_SmithPokorny75())
        deficiency = daltonlens.simulate.Deficiency.DEUTAN if cvd_type == 'deutan' else daltonlens.simulate.Deficiency.PROTAN
    elif cvd_type == 'mono':
        img_cvd[:, :, 0:3] = colorspacious.cspace_convert(img_cvd[:, :, 0:3], "sRGB1", "JCh")
        img_cvd[:, :, 1] = 0
        img_cvd[:, :, 0:3] = colorspacious.cspace_convert(img_cvd[:, :, 0:3], "JCh", "sRGB1")
        return np.clip(img_cvd, 0, 1, out=img_cvd)
    else:
        raise ValueError('cvd_type must be one of ("deutan", "protan", "tritan", "mono")')

    img_cvd[:, :, 0:3] = simulator._simulate_cvd_linear_rgb(img_cvd[:, :, 0:3], deficiency, severity / 100)

    return np.clip(img_cvd, 0, 1, out=img_cvd)


def simulate_cmap(cmap, cvd_type, severity):
    """Generate a simulated CVD colormap from an existing one.

    :param cmap: Colormap-like object
    :type cmap: colormap-like
    :param cvd_type: One of 'deutan', 'protan', 'tritan', 'mono'
    :rype cvd_type: str
    :param severity: CVD severity in percent (0 - 100)
    :type severity: float
    :return: Colormap as perceived with CVD
    :rtype: colormap-like

    :raises ValueError: If an unkown `cvd_type` is passed.
    """
    x = np.linspace(0, 1, cmap.N)
    img = cmap(x)[np.newaxis, :, :]
    cvd = simulate_cvd(img, cvd_type, severity).squeeze()

    if isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        return matplotlib.colors.LinearSegmentedColormap.from_list(cmap.name + '_' + cvd_type, cvd)
    elif isinstance(cmap, matplotlib.colors.ListedColormap):
        return matplotlib.colors.ListedColormap(cvd, cmap.name + '_' + cvd_type)
