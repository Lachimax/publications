#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np

import astropy.table as table
import astropy.units as units
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import craftutils.utils as u
import craftutils.observation.field as field
import craftutils.observation.image as image
import craftutils.plotting as pl

import lib

description = """
Does something cool.
"""


def ba_from_inclination(i: units.Quantity[units.degree], q_0: float = 0.2):
    return np.sqrt((np.cos(i) ** 2) * (1 - q_0 ** 2) + q_0 ** 2)


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    output_path_fig = os.path.join(lib.output_path, "imaging")
    os.makedirs(output_path_fig, exist_ok=True)

    q_0 = 0.2

    galfit_tbl = table.QTable.read(os.path.join(input_dir, "galfit_table.ecsv"))

    c_frb = "black"
    c_text = "white"
    c_ellipse = "black"

    frbs = {
        "FRB20210807D": {
            "file_vlt": "FRB20210807D_VLT-FORS2_g-HIGH_combined.fits",
            "file_b24": "FRB20210807D_PS1.fits",
            "stretch": "log",
            "vmax_vlt": 31.5,
            "vmax_b24": 3023,
            "i_b24": 41 * units.degree
            # "frame":
        },
        "FRB20191001A": {
            "file_vlt": "FRB20191001A_VLT-FORS2_g-HIGH_2020-09-19.fits",
            "file_b24": "FRB20191001A_LS.fits",
            # "stretch": "log",
            "vmax_vlt": 29,
            "vmax_b24": 0.35,
            "invert": True,
            "i_b24": 53 * units.degree
            # "frame":
        },
        "FRB20190714A": {
            "file_vlt": "FRB20190714A_VLT-FORS2_g-HIGH_combined.fits",
            "file_b24": "FRB20190714A_panstarrs1_r_None.fits",
            # "stretch": "log",
            "vmax_vlt": 31,
            # "vmax_b24": 0.35,
            # "invert": True
            # "frame":
            "i_b24": 51 * units.degree
        },
        "FRB20200906A": {
            "file_vlt": "FRB20200906A_VLT-FORS2_g-HIGH_combined.fits",
            "file_b24": "FRB20200906A_LS.fits",
            # "stretch": "log",
            "vmax_vlt": 14,
            "vmax_b24": 0.12,
            "invert": True,
            # "frame":
            "i_b24": 63 * units.degree
        },
        "FRB20210320C": {
            "file_vlt": "FRB20210320C_VLT-FORS2_g-HIGH_2021-04-15.fits",
            "file_b24": "FRB20210320C_panstarrs1_r.fits",
            "stretch": "log",
            "vmax_vlt": 24,
            # "vmin_vlt": 0
            # "vmax_b24": 1000,
            # "vmin_b24": 0
            # "invert": True
            # "frame":,
            "i_b24": 28 * units.degree
        },
    }

    n_height = 2
    n_width = 1
    scale = None

    for frb, params in frbs.items():
        print(frb)
        fld_ = field.Field.from_params(frb)
        path_vlt = os.path.join(lib.input_path, "imaging", frb, params["file_vlt"])
        path_b24 = os.path.join(lib.input_path, "imaging", frb, params["file_b24"])

        fig: plt.Figure = plt.figure(
            figsize=(
                (1.2 * pl.textwidths["PASA"] / 2,
                 1.2 * pl.textwidths["PASA"])
            )
        )

        index = list(galfit_tbl["field_name"]).index(frb)
        row = galfit_tbl[index]
        coord = SkyCoord(row["galfit_ra"], row["galfit_dec"])

        img_vlt = image.Image.from_fits(path_vlt)
        img_b24 = image.Image.from_fits(path_b24)
        img_b24.load_wcs()

        if "PSF_FWHM" not in img_b24.headers[0]:

            if "CELL.GAIN" in img_b24.headers[0]:
                img_b24.headers[0]["GAIN"] = img_b24.headers[0]["CELL.GAIN"]
            elif "GAIN" not in img_b24.headers[0]:
                img_b24.headers[0]["GAIN"] = 1.

            output_dir_img = os.path.join(output_path_fig, frb)
            os.makedirs(output_dir_img, exist_ok=True)
            img_b24.source_extraction_psf(
                output_dir=output_dir_img,
            )
            img_b24.psf_diagnostics(
                output_path=output_dir_img,
                star_class_tol = 0.95
            )

        psf_b24 = img_b24.extract_psf_fwhm()
        psf_vlt = img_vlt.extract_psf_fwhm()

        print(img_b24, type(img_b24))

        host = fld_.frb.host_galaxy
        z = host.z

        deep = host.select_deepest_sep()
        if deep is None:
            print(f"No photometry found for {frb}. You might have to run the objects pipeline.")
            return None
        elif deep["a"] < 0:
            deep = host.select_deepest()

        if scale is not None and z != 999.:
            print("CASE 1")
            frame = scale * units.kpc
        elif "frame" in params:
            print("CASE 2")
            frame = params["frame"]
        else:
            print("CASE 3: PANIC")
            # raise ValueError
            img_vlt.extract_pixel_scale()
            print(deep["a"], deep["b"], deep["kron_radius"])
            frame = img_vlt.nice_frame(
                row={
                    "A_WORLD": deep["a"],
                    "B_WORLD": deep["b"],
                    "KRON_RADIUS": deep["kron_radius"]
                },
                frame=5 * units.arcsec
            ).to(units.arcsec, img_vlt.pixel_scale_y)

        ax_vlt = fig.add_subplot(n_height, n_width, 1, projection=img_vlt.wcs[0])

        print("\tframe:", frame)
        print("\tpix scale", img_vlt.pixel_scale_y)

        normalize_kwargs = {}
        if "stretch" in params:
            normalize_kwargs["stretch"] = params["stretch"]
        # if fld in interval:
        normalize_kwargs["interval"] = "zscale"  # interval[fld]
        if "vmax_vlt" in params:
            normalize_kwargs["vmax"] = params["vmax_vlt"]
        if "vmin_vlt" in params:
            normalize_kwargs["vmin"] = params["vmin_vlt"]
        print(f"\t\t{normalize_kwargs}")

        ba_vlt = row["galfit_axis_ratio"]
        r_eff = row["galfit_r_eff"]
        theta = row["galfit_theta"].value

        a_vlt = img_vlt.pixel(r_eff * 2)
        b_vlt = a_vlt * ba_vlt

        fig, ax_vlt, vals = fld_.plot_host(
            fig=fig,
            ax=ax_vlt,
            img=img_vlt,
            frame=frame,
            draw_scale_bar=True,
            scale_bar_kwargs=dict(
                text_kwargs=dict(color=c_text),
                size=psf_vlt,
                bold=True,
                line_kwargs=dict(
                    color=c_text
                )
            ),
            # astm_crosshairs=True,
            astm_kwargs={
                "color": c_text
            },
            show_frb=True,
            frb_kwargs={
                "edgecolor": c_frb,
                "lw": 2,
                "include_img_err": True
            },
            # latex_kwargs=
            # centre=centre,
            normalize_kwargs=normalize_kwargs,
        )

        x, y = img_vlt.world_to_pixel(coord, 0)
        e = Ellipse(
            xy=(x, y),
            width=a_vlt,
            height=b_vlt,
            angle=theta + 90,
            edgecolor=c_ellipse,
            ls=":",
            facecolor="none",
            lw=2
        )
        ax_vlt.add_artist(e)

        ax_vlt.text(
            0.1, 0.85, fr"$b/a = {ba_vlt.round(2)}$; $i = {int(row['galfit_inclination'].value)}^\circ$",
            transform = ax_vlt.transAxes,
            color="white",
        )


        ba_b24 = ba_from_inclination(i=params["i_b24"], q_0=q_0)
        print("B/A for B+24 is:", ba_b24)
        print("B/A for VLT is:", ba_vlt)
        print("\t- Recalculated:", ba_from_inclination(i=row['galfit_inclination'], q_0=q_0))

        a_b24 = img_b24.pixel(r_eff * 2)
        b_b24 = a_b24 * ba_b24

        ra, dec = ax_vlt.coords

        dec.set_axislabel(" ")
        dec.set_ticklabel(
            fontsize=12,
            exclude_overlapping=True
        )
        ra.set_axislabel(" ")
        ra.set_ticklabel_visible(False)

        normalize_kwargs = {}
        if "stretch" in params:
            normalize_kwargs["stretch"] = params["stretch"]
        normalize_kwargs["interval"] = "zscale"  # interval[fld]
        if "vmax_b24" in params:
            normalize_kwargs["vmax"] = params["vmax_b24"]
        if "vmin_b24" in params:
            normalize_kwargs["vmax"] = params["vmin_b24"]
        print(f"\t\t{normalize_kwargs}")

        ax_b24 = fig.add_subplot(n_height, n_width, 2, projection=img_b24.wcs[0])
        fig, ax_b24, vals = fld_.plot_host(
            fig=fig,
            ax=ax_b24,
            img=img_b24,
            frame=frame,
            draw_scale_bar=True,
            scale_bar_kwargs=dict(
                text_kwargs=dict(color=c_text),
                size=psf_b24,
                bold=True,
                line_kwargs=dict(
                    color=c_text
                )
            ),
            # astm_crosshairs=True,
            astm_kwargs={
                "color": c_text
            },
            show_frb=True,
            frb_kwargs={
                "edgecolor": c_frb,
                "lw": 2,
                "include_img_err": True
            },
            # latex_kwargs=
            # centre=centre,
            normalize_kwargs=normalize_kwargs,
            imshow_kwargs={"cmap": "plasma"}
        )

        if "invert" in params and params["invert"]:
            ax_b24.invert_xaxis()
            theta = -theta

        x, y = img_b24.world_to_pixel(coord, 0)
        e = Ellipse(
            xy=(x, y),
            width=a_b24,
            height=b_b24,
            angle=theta + 90,
            edgecolor=c_ellipse,
            ls=':',
            facecolor="none",
            lw=2
        )
        ax_b24.add_artist(e)

        text = ax_b24.text(
            0.1, 0.85, fr"$b/a = {ba_b24.round(2)}$; $i = {int(params['i_b24'].value)}^\circ$",
            transform = ax_b24.transAxes,
            color="white",
        )


        ra, dec = ax_b24.coords
        dec.set_axislabel(" ")
        dec.tick_params(labelsize=12)
        ra.set_axislabel(" ")
        ra.set_ticklabel(
            fontsize=12,
            exclude_overlapping=True
            # rotation=45,
            # pad=50
        )

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
            # fontsize=14
        )
        ax_big.set_ylabel(
            "Declination (J2000)",
            labelpad=10.,
            # fontsize=14,
            rotation=-90
        )
        ax_big.yaxis.set_label_position("right")

        # fig.tight_layout(h_pad=-5.)

        for f in (".pdf", ".png"):
            fig.savefig(os.path.join(output_path_fig, frb + f), bbox_inches='tight')
        fig.clear()
        plt.close(fig)
        del fig


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
