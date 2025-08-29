#!/usr/bin/env python
# Code by Lachlan Marnoch, 2023-2024

import os
import copy

import numpy as np
import matplotlib.pyplot as plt

from astropy import units, table
from astropy.coordinates import SkyCoord, angles
from astropy.io import fits

import craftutils.utils as u
import craftutils.params as p
from craftutils.observation import objects, field, instrument, image, sed
import craftutils.astrometry as astm
import craftutils.plotting as pl

import lib

description = """
Generates imaging figures.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    output_dict = {}

    fld = lib.fld

    pl.latex_setup()

    decam = lib.decam
    r_decam = lib.r_decam

    g_img = image.SurveyCutout(lib.cutout_path("g"))
    g_img.filter = lib.g_decam
    r_img = image.SurveyCutout(lib.cutout_path("r"))
    r_img.filter = lib.r_decam
    z_img = image.SurveyCutout(lib.cutout_path("z"))
    z_img.filter = lib.z_decam

    print("Generating imaging plot...")

    figsize = pl.textwidths["mqthesis"]
    fig = plt.figure(figsize=(figsize, figsize / 2))

    for i, img in enumerate((g_img, r_img, z_img)):
        img.instrument = lib.decam
        img.load_data()
        fig, ax, _ = fld.plot_host(
            img=img,
            fig=fig,
            n=i + 1,
            n_x=3,
            n_y=1,
            frame=9 * units.arcsec,
            frb_kwargs={"edgecolor": "black", "lw": 1.5},
            draw_scale_bar=True,
            scale_bar_kwargs={
                "spread_factor": 0.8,
                "size": 3 * units.arcsec,
                "extra_height_top_factor": 1
            },
            # imshow_kwargs=dict(cmap="plasma"),
            normalize_kwargs=dict(vmax=0.002 * np.max(img.data[0]).value),
            # output_path=os.path.join(lib.output_path, "FRB20230718A_decam_r.pdf")
        )
        ra, dec = ax.coords
        dec.set_axislabel(" ")
        ra.set_axislabel(" ")

        ax.set_title(f"{img.instrument.formatted_name}, ${img.filter.name}$")

        if i > 0:
            dec.set_ticklabel_visible(False)
        else:
            dec.set_ticklabel(fontsize=pl.tick_fontsize)

        ra.set_ticklabel(
            # rotation=30,
            fontsize=pl.tick_fontsize,
            # pad=25,
            horizontalalignment="right",
            exclude_overlapping=True
        )
        # ra.set_ticks(spacing=10 * units.arcsec)
        # ra.set_ticklabel_position("bottom")

    ax = fig.add_subplot(1, 1, 1)
    ax.set_frame_on(False)
    ax.tick_params(left=False, right=False, top=False, bottom=False)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    # ax.set_ticks([])
    # ax.set_aspect("equal")
    ax.set_xlabel(
        "Right Ascension (J2000)",
        labelpad=-10,
        fontsize=pl.axis_fontsize
    )
    ax.set_ylabel(
        "Declination (J2000)",
        rotation=-90,
        labelpad=20,
        fontsize=pl.axis_fontsize
    )
    ax.yaxis.set_label_position("right")

    # plt.tight_layout()

    plt.savefig(
        os.path.join(lib.output_path, "FRB20230718A_decam.pdf"),
        bbox_inches="tight",
        # pad_inches=-0.05,
    )


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
    parser.add_argument(
        "--skip_path",
        help="Skips PATH runs; in other words, only does the imaging plot.",
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
