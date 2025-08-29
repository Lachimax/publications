#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX
import matplotlib.pyplot as plt
import os

import numpy as np

from astropy import units
from astropy.coordinates import SkyCoord

import craftutils.utils as u
import craftutils.observation.image as image
import craftutils.observation.epoch as epoch
import craftutils.observation.field as field
import craftutils.plotting as pl

import lib

description = """
Generates figures illustrating pipeline stages.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    db_path = os.path.join(lib.dropbox_path, "figures")

    pl.latex_setup()

    fld = field.Field.from_params("FRB20220105A")
    epc = epoch.ImagingEpoch.from_params(
        "FRB20220105A_FORS2_2",
        instrument="vlt-fors2",
        field=fld
    )
    frb = fld.frb

    input_dir = os.path.join(input_dir, 'imaging', 'stages')
    output_dir = os.path.join(output_dir, 'stages')
    os.makedirs(output_dir, exist_ok=True)

    # fld = field.Field.from_params("FRB20220105A")

    imgs = {
        "00-downloaded": "00-FORS2.2022-03-01T06_33_21.629.fits",
        "01-reduced": "01-FRB20220105A_FORS2_2_FORS2.2022-03-01T06_33_21.629_SCIENCE_REDUCED_IMG.fits",
        "02-trimmed": "02-FRB20220105A_FORS2_2_FORS2.2022-03-01T06_33_21.629_SCIENCE_REDUCED_IMG_trim.fits",
        "06-astrometry": "06-FRB20220105A_FORS2_2_FORS2.2022-03-01T06_33_21.629_SCIENCE_REDUCED_IMG_norm_astrometry.fits",
        "09-coadded": "09-FRB20220105A_FORS2_2_2022-03-01_R_SPECIAL_coadded_mean-sigmaclip.fits",
        "11-coadded_trimmed": "11-FRB20220105A_FORS2_2_2022-03-01_R_SPECIAL_coadded_mean-sigmaclip_trimmed.fits"
    }
    img_objs = {}
    img_paths = {}
    # Inset zoom?
    # Montage - clipped, mean, median insets
    # Astrometry frame - Gaia overlay? Before/after
    for name, img_path in imgs.items():
        img_path = os.path.join(input_dir, img_path)
        img_paths[name] = img_path
        img = image.FORS2Image(
            img_path
        )
        img_objs[name] = img
        img.load_data()
        img.load_wcs()
        vmin = np.nanmedian(img.data[0]).value
        print(name, img.name)
        fig = plt.figure(
            figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"]),
        )
        ax = fig.add_subplot(projection=img.wcs[0])
        fig, ax, _ = img.plot_subimage(
            centre=frb.position,
            imshow_kwargs={"cmap": "cmr.bubblegum"},
            normalize_kwargs={"interval": "zscale", "stretch": "linear", "vmin": vmin},
            fig=fig,
            ax=ax
        )
        fig.savefig(
            os.path.join(output_dir, name + ".pdf"),
            bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(db_path, name + ".pdf"),
            bbox_inches="tight"
        )

    fld = field.Field.from_params("FRB20210807D")
    epc = epoch.ImagingEpoch.from_params(
        "FRB20210807_FORS2_2",
        instrument="vlt-fors2",
        field=fld
    )
    fig = plt.figure(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] * 2 / 3))
    fld.load_catalogue("gaia", data_release=3)
    gaia = epc.epoch_gaia_catalogue()
    gaia["coord"] = SkyCoord(gaia["ra"], gaia["dec"])

    path_before = "04-FRB20210807_FORS2_2_FORS2.2022-04-08T09_15_16.836_SCIENCE_REDUCED_IMG_norm.fits"
    path_after = "06-FRB20210807_FORS2_2_FORS2.2022-04-08T09_15_16.836_SCIENCE_REDUCED_IMG_norm_astrometry.fits"

    n_width = 2
    n_height = 1

    for i, name in enumerate((path_before, path_after)):
        path = os.path.join(input_dir, "astrometry", name)
        img = image.FORS2Image(path)
        img.load_data()
        img.load_wcs()
        n = i + 1
        ax = fig.add_subplot(1, 2, n, projection=img.wcs[0])
        img.plot_subimage(
            centre=SkyCoord("19h56m52.9931 -0d45m10s"),
            img=img,
            ax=ax,
            frame=0.8 * units.arcmin,
            imshow_kwargs={"cmap": "cmr.bubblegum"},
            normalize_kwargs={
                "interval": "zscale", 
                "stretch": "linear",
                "vmin": np.nanmedian(img.data[0]).value
            },
            show_frb=False
        )
        gaia["x"], gaia["y"] = img.world_to_pixel(gaia["coord"])
        ax.scatter(
            gaia["x"], gaia["y"],
            marker="x",
            s=10,
            color="red"
        )
        ra, dec = ax.coords
        dec.set_axislabel(" ")
        ra.set_axislabel(" ")
        n_x = n % n_width
        is_left = n_x == 1
        is_bottom = np.ceil(n / n_width) == n_height
        if not is_bottom:
            ra.set_ticklabel_visible(False)
        if not is_left:
            dec.set_ticklabel_visible(False)

    ax_big = fig.add_subplot(1, 1, 1)
    ax_big.set_frame_on(False)
    ax_big.tick_params(left=False, right=False, top=False, bottom=False)
    ax_big.yaxis.set_ticks([])
    ax_big.xaxis.set_ticks([])
    ax_big.set_xlabel(
        "Right Ascension (J2000)",
        # labelpad=30,
        fontsize=14
    )
    ax_big.set_ylabel(
        "Declination (J2000)",
        labelpad=30.,
        fontsize=14,
        rotation=-90
    )
    ax_big.yaxis.set_label_position("right")

    fig.savefig(os.path.join(output_dir, "06-astrometry_delta.pdf"))
    fig.savefig(os.path.join(db_path, "06-astrometry_delta.pdf"))



    fig = plt.figure(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"]))
    fld = field.Field.from_params("FRB20191228A")

    img_names = [
        "FRB20191228A_VLT-FORS2_g-HIGH_2020-09-21.fits",
        "FRB20191228A_VLT-FORS2_g-HIGH_back-subtracted-patch.fits",
        "FRB20191228A_VLT-FORS2_I-BESS_2020-09-21.fits",
        "FRB20191228A_VLT-FORS2_I-BESS_back-subtracted-patch.fits",
    ]
    n_height = n_width = 2
    for i, img_name in enumerate(img_names):
        n = i + 1
        img_path = os.path.join(input_dir, "backsub", img_name)
        img = image.FORS2CoaddedImage(img_path)
        img.load_data()
        img.load_wcs()
        ax = fig.add_subplot(n_height, n_width, n, projection=img.wcs[0])
        ra, dec = ax.coords
        fig, ax, props = fld.plot_host(
            img=img,
            n=i + 1,
            n_x=n_width,
            n_y=n_height,
            frame=12 * units.arcsec,
            fig=fig,
            ax=ax,
            show_frb=False,
            normalize_kwargs={"stretch": "sqrt", "interval": "minmax"},
        )
        dec.set_axislabel(" ")
        ra.set_axislabel(" ")
        n_x = n % n_width
        is_left = n_x == 1
        is_bottom = np.ceil(n / n_width) == n_height
        if not is_bottom:
            ra.set_ticklabel_visible(False)
        if not is_left:
            dec.set_ticklabel_visible(False)

    ax_big = fig.add_subplot(1, 1, 1)
    ax_big.set_frame_on(False)
    ax_big.tick_params(left=False, right=False, top=False, bottom=False)
    ax_big.yaxis.set_ticks([])
    ax_big.xaxis.set_ticks([])
    ax_big.set_xlabel(
        "Right Ascension (J2000)",
        labelpad=30,
        fontsize=14
    )
    ax_big.set_ylabel(
        "Declination (J2000)",
        labelpad=30.,
        fontsize=14,
        rotation=-90
    )
    ax_big.yaxis.set_label_position("right")
    fig.savefig(os.path.join(output_dir, "08-backsub.pdf"))
    fig.savefig(os.path.join(db_path, "08-backsub.pdf"))


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
