#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import craftutils.utils as u
from craftutils.retrieve import load_catalogue

import lib

description = """
Finds the nuisance stars in Gaia and makes some check plots.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb210912 = lib.fld

    lib.load_images()
    R_img = lib.R_img
    g_img = lib.g_img

    plot_path = os.path.join(lib.output_path, "Gaia")
    u.mkdir_check(plot_path)

    gaia = load_catalogue(
        cat_name="gaia",
        cat=os.path.join(lib.repo_data, "gaia_FRB20210912A.csv"),
    )
    # Get pixel coordinates for the Gaia stars in each image
    gaia_coord = SkyCoord(gaia["ra"], gaia["dec"])
    gaia["skycoord"] = gaia_coord
    gaia["x_R"], gaia["y_R"] = R_img.world_to_pixel(gaia_coord)
    gaia["x_g"], gaia["y_g"] = g_img.world_to_pixel(gaia_coord)
    gaia["in_R"] = R_img.wcs[0].footprint_contains(gaia_coord)
    gaia["in_g"] = g_img.wcs[0].footprint_contains(gaia_coord)

    # Plot Gaia positions on image
    R_gaia = gaia[gaia["in_R"]]
    g_gaia = gaia[gaia["in_g"]]
    ax, fig = g_img.plot()
    ax.scatter(g_gaia["x_g"], g_gaia["y_g"], marker="x", c='red')
    fig.savefig(os.path.join(plot_path, "gaia_g.pdf"))
    ax, fig = R_img.plot()
    ax.scatter(R_gaia["x_R"], R_gaia["y_R"], marker="x", c='red')
    fig.savefig(os.path.join(plot_path, "gaia_R.pdf"))

    # Get brightest stars and plot their IDs on the g image.
    brightest = g_gaia[:10]
    brightest["distance"] = frb210912.frb.position.separation(brightest["skycoord"]).to("arcsec")
    brightest.sort("distance")
    frb_x_g, frb_y_g = g_img.world_to_pixel(frb210912.frb.position)
    ax, fig = g_img.plot()
    ax.scatter(frb_x_g, frb_y_g, c="violet")
    ax.scatter(brightest["x_g"], brightest["y_g"], marker="x", c='red')

    for i, row in enumerate(brightest):
        plt.text(row["x_g"], row["y_g"], row["SOURCE_ID"], c="red")

    fig.savefig(os.path.join(plot_path, "bright_stars_g.pdf"))
    brightest.write(os.path.join(plot_path, "bright_stars.ecsv"), overwrite=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=description
    )

    parser.add_argument(
        "-o",
        help="Path to output directory.",
        type=str,
        default=lib.default_output_path
    )
    parser.add_argument(
        "-i",
        help="Path to directory containing input files.",
        type=str,
        default=lib.default_input_path
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path
    )
