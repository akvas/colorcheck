# This file is part of colorcheck
# Copyright (C) 2022 - 2024 Andreas Kvas
# See LICENSE for detailed licensing information.

import argparse
import os
import matplotlib.pyplot as plt
from .. import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='color (lightness) flipper inspired by Crameri, Fabio. (2020). deLight (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.3634423')
    parser.add_argument('image', metavar='IMG', help='File name for the image to be flipped')
    parser.add_argument('-s', '--suffix', type=str, default='flipped', help='Suffix to be appended to the image base name')
    parser.add_argument('-p', '--preserve-color', action='store_true', help='Preserve foreground colors, defaults to False.')
    parser.add_argument('-r', '--reduce-darkness', type=float, default=0.1, help='Increase the overall figure lightness (0 - 1), defaults to 0.1')
    parser.add_argument('-t', '--darkness-threshold', type=float, default=6.0, help='If reduce_darkness is non-zero, this parameter in the range (0 - 100) governs which pixels are presumed dark, defaults to 6.')

    args = parser.parse_args()

    base_name, ext = os.path.splitext(args.image)
    try:
        img = plt.imread(args.image, format='png')
    except SyntaxError:
        img = plt.imread(args.image) / 255

    img_out = utils.flip_lightness(img, args.preserve_color, args.reduce_darkness, args.darkness_threshold)
    output_name = base_name + '.' + args.suffix + ext
    plt.imsave(output_name, img_out)
