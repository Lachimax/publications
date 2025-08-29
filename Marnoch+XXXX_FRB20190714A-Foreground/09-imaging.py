#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from matplotlib.patches import Ellipse

from astropy import units
from astropy.table import QTable

import craftutils.utils as u
import craftutils.plotting as pl

import lib

description = """
Generates imaging figures of foreground galaxies.
"""


def plot_all_bands(
        row,
        scale: float = None,
        c_text: str = "black"
):
    print("=" * 100)
    print(row["id"])
    z = row["z"]

    bands = {}
    for name, img in lib.img_dict.items():
        if row["in_" + name]:
            bands[name] = img

    n_total = len(bands)
    if n_total == 0:
        return None

    n_width = 2
    if n_total < n_width:
        n_width = n_total
    n_height = int(np.ceil(n_total / n_width))
    n_squares = (n_width * n_height)
    print(f"\t{n_total=}")
    print(f"\t{n_width=}")
    print(f"\t{n_height=}")
    fig: plt.Figure = plt.figure(figsize=(
        (pl.textwidths["mqthesis"] * n_width / 2),
        (pl.textwidths["mqthesis"] * n_height / 2)
    ))

    img = bands[list(bands.keys())[0]]
    # Pick a nice frame for this galaxy, or otherwise the scale as set by common_scale
    if scale is not None and z != 999.:
        print("CASE 1")
        frame = scale * units.kpc
    # elif fld in frames:
    #     print("CASE 2")
    #     frame = frames[fld]
    else:
        print("CASE 3")
        img.extract_pixel_scale()
        print(row["a"], row["b"], row["kron_radius"])
        frame = img.nice_frame(
            row={
                "A_WORLD": row["a"],
                "B_WORLD": row["b"],
                "KRON_RADIUS": row["kron_radius"]
            },
            frame=5 * units.arcsec
        ).to(units.arcsec, img.pixel_scale_y)
    print("\tframe:", frame)
    frame_eff = frame.to(units.pix, img.pixel_scale_y)

    for i, (name, img) in enumerate(bands.items()):
        print("\t", name)
        n = i + 1
        is_bottom = np.ceil(n / n_width) == n_height
        n_x = n % n_width
        print(f"\t\t{n_x=}")
        print(f"\t\t{n=}, {n_width=}, {n_height=}, {n_total=}")
        print(f"\t\t{n_total=}, {n_squares=}")
        if is_bottom and n_total < n_squares:
            n_width_this = n_width - (n_squares - n_total)
            n_squares_this = n_height * n_width_this
            n_this = n_squares_this - (n_total - n)
        else:
            n_width_this = n_width
            n_this = n
        print(f"\t\t{n_width_this=}, {n_this=}")
        is_left = (n_x == 1 or n_width_this == 1)
        print(f"\t\t{is_bottom=}")
        fil_name = img.filter.name

        # normalize_kwargs = {}
        # if fld in stretch:
        #     normalize_kwargs["stretch"] = stretch[fld]
        # if fld in interval:
        # normalize_kwargs["interval"] = "zscale"  # interval[fld]
        # if "flux_max" in phot_dict:
        #     peak = phot_dict["flux_max"] + phot_dict["background_se"]
        #     normalize_kwargs["vmax"] = peak.value
        # print(f"\t\t{normalize_kwargs}")

        ax = fig.add_subplot(n_height, n_width_this, n_this, projection=img.wcs[0])
        ra, dec = ax.coords
        ra.set_ticklabel(
            fontsize=pl.tick_fontsize,
            exclude_overlapping=True
            # rotation=45,
            # pad=50
        )
        dec.set_ticklabel(fontsize=pl.tick_fontsize)
        ax.set_title(f"{img.instrument.formatted_name}, {img.filter.formatted_name}", size=14)

        centre = row["coord"]

        x, y = img.world_to_pixel(centre, origin=0)

        # psf = img.extract_psf_fwhm()

        fig, ax, vals = img.plot_subimage(
            fig=fig,
            ax=ax,
            img=img,
            frame=frame,
            scale_bar_kwargs=dict(
                text_kwargs=dict(color=c_text),
                size=1 * units.arcsec,
                bold=True,
                line_kwargs=dict(
                    color=c_text
                )
            ),
            # astm_crosshairs=True,
            # astm_kwargs={
            #     "color": c_text
            # },
            # show_frb=True,
            # frb_kwargs={
            #     "edgecolor": c_frb,
            #     "lw": 2,
            #     "include_img_err": True
            # },
            # latex_kwargs=
            centre=centre,
            # normalize_kwargs=normalize_kwargs,
        )
        dec.set_axislabel(" ")
        ra.set_axislabel(" ")

        # if not is_bottom:
        #     ra.set_ticklabel_visible(False)
        if not is_left:
            dec.set_ticklabel_visible(False)

        # if is_bottom

        # if z is not None:
        #     ax.text(
        #         s=f"\\textbf{{{fld.replace('FRB', '')}}}" + f"\n$\mathbf{{z = {z}}}$",
        #         x=0.05,
        #         y=0.95,
        #         c=c_text,
        #         ha='left',
        #         va="top",
        #         fontsize=10,
        #         transform=ax.transAxes,
        #     )
        # else:
        #     ax.text(
        #         s=f"\\textbf{{{fld.replace('FRB', '')}}}",
        #         x=0.05,
        #         y=0.95,
        #         c=c_text,
        #         ha='left',
        #         va="top",
        #         fontsize=10,
        #         transform=ax.transAxes,
        #     )

    ax_big = fig.add_subplot(1, 1, 1)
    ax_big.set_frame_on(False)
    ax_big.tick_params(left=False, right=False, top=False, bottom=False)
    ax_big.yaxis.set_ticks([])
    ax_big.xaxis.set_ticks([])
    # ax.set_ticks([])
    # ax.set_aspect("equal")
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
    # fig.suptitle(
    #     fld_.name.replace("FRB", r"HG\,"),
    #     # y=1
    # )

    fig.subplots_adjust(hspace=0.3)

    lib.savefig(
        fig=fig,
        subdir="grids",
        filename=f"{row['id']}_grid",
        # tight=True
    )

    fig.clear()
    plt.close(fig)
    del fig


def main(
        output_dir: str,
        input_dir: str,
        test: bool
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    pl.latex_setup()

    frb190714 = lib.fld
    frb190714.load_all_objects()
    frb = frb190714.frb.position

    lib.load_imaging()

    main_props = lib.read_master_properties()
    halos = lib.read_master_table()

    img_dict = lib.img_dict

    # plt.style.use('default')


    for name, img in img_dict.items():
        frb_x, frb_y = img.world_to_pixel(frb)
        main_props["frb_x" + name] = frb_x
        main_props["frb_y" + name] = frb_y

        halos["x_" + name], halos["y_" + name] = img.world_to_pixel(halos["coord"])

        halos["in_" + name] = img.wcs[0].footprint_contains(halos["coord"])

    if test:
        img_dict = {"vlt-fors2_I": img_dict["vlt-fors2_I"]}

    #######################################################################
    # All galaxies

    for name, img in img_dict.items():
        frb_x, frb_y = img.world_to_pixel(frb)
        print("Processing", name)
        _, fig, ax = lib.label_objects(
            tbl=halos,
            img=img,
            output=f"all_labelled_{name}",
            short_labels=True,
            frb_kwargs={"colour": "limegreen"},
            show_frb=False,
            do_cut=True,
            ellipse_kwargs={"linewidth": 1},
            figsize=(lib.figwidth, lib.figwidth * 2 / 3),
            # figsize = (0.75 * lib.figwidth_sideways, lib.figwidth_sideways * 2 / 3)
        )
        # muse_size = img.pixel(1 * units.arcmin).value
        # muse_rad = muse_size / 2
        ra = frb.ra + 5 * units.arcsec
        dec = frb.dec - 5 * units.arcsec
        delta_dec = 0.5 * units.arcmin
        delta_ra = delta_dec / np.cos(dec)

        muse_coord = SkyCoord(
            [
                (ra - delta_ra, dec + delta_dec),
                (ra + delta_ra, dec + delta_dec),
                (ra + delta_ra, dec - delta_dec),
                (ra - delta_ra, dec - delta_dec),
                (ra - delta_ra, dec + delta_dec)
             ],
            unit=(units.deg, units.deg)
        )
        muse_x, muse_y = img.world_to_pixel(muse_coord)

        # muse_x = (frb_x - muse_rad, frb_x + muse_rad, frb_x + muse_rad, frb_x - muse_rad, frb_x - muse_rad)
        # muse_y = (frb_y + muse_rad, frb_y + muse_rad, frb_y - muse_rad, frb_y - muse_rad, frb_y + muse_rad)

        ax.plot(muse_x, muse_y, c="violet", lw=1)
        # if name == "vlt-fors2_g":
        #     ax.set_xlabel(" ")

        lib.savefig(fig, f"all_labelled_{name}_fov", subdir="imaging", tight=True)

        _, fig, ax = lib.label_objects(
            tbl=halos,
            img=img,
            output=f"all_labelled_no_ellipse_{name}",
            short_labels=True,
            show_frb=False,
            frb_kwargs={"colour": "limegreen"},
            do_cut=True,
            do_ellipses=False,
            figsize=(0.75 * lib.figwidth_sideways, lib.figwidth_sideways * 2 / 3)
        )
        # R_200
        for row in halos:
            r200_pix = img.pixel(row["r_200_angle_K18"])
            e = Ellipse(
                xy=(row[f"x_{name}"], row[f"y_{name}"]),
                width=2 * r200_pix,
                height=2 * r200_pix,
                angle=0.0
            )
            e.set_edgecolor("white")
            e.set_facecolor("none")
            ax.add_artist(e)
        lib.savefig(fig, f"all_r200_{name}", subdir="imaging")

        lib.label_objects(
            tbl=halos,
            img=img,
            output=f"all_z_{name}",
            ellipse_colour="z",
            short_labels=True,
            text_colour="black",
            do_cut=True,
            show_frb=False,
            imshow_kwargs={"cmap": "binary"},
            figsize=(0.75 * lib.figwidth_sideways, lib.figwidth_sideways * 2 / 3)
        )

    #######################################################################
    # MUSE only

    for name, img in img_dict.items():

        if "hst" in name:
            c = "white"
        else:
            c = "black"

        # if name.startswith("vlt"):
        print("Processing", name)
        lib.label_objects(
            tbl=halos[halos["sample"] == "MUSE"],
            img=img,
            output=f"muse_labelled_{name}",
            short_labels=True,
            frb_kwargs={"colour": c, "linewidth": 2},
        )

        lib.label_objects(
            tbl=halos[halos["sample"] == "MUSE"],
            img=img,
            output=f"muse_z_{name}",
            ellipse_colour="z",
            short_labels=True,
            text_colour="black",
            imshow_kwargs={"cmap": "binary"},
        )

    if test:
        halos = halos[[0, -1]]

    ########################################################################################
    # Individual portraits

    if not test:
        for row in halos:

            plot_all_bands(row=row)

    if not test:
        lib.write_master_table(halos)


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
        "--test",
        help="Test mode. Run only a limited selection of plots.",
        action="store_true",
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        test=args.test
    )
