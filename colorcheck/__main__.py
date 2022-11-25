# This file is part of colorcheck
# Copyright (C) 2022 Andreas Kvas
# See LICENSE for detailed licensing information.


import argparse
import os 
import matplotlib.pyplot
from . import simulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CVD simulator')
    parser.add_argument('image', metavar='IMG', help='image file name for CVD simulations')
    parser.add_argument('-s', '--severity', type=int, default=100)

    args = parser.parse_args()

    img = matplotlib.pyplot.imread(args.image)

    for cvd_type in ('deutan', 'protan', 'tritan', 'mono'):

        file_path, ext = os.path.splitext(args.image)
        output_filename = file_path + '.' + cvd_type + ext 
        
        img_cvd = simulation.simulate_cvd(img, cvd_type, args.severity)
        matplotlib.pyplot.imsave(output_filename, img_cvd)
