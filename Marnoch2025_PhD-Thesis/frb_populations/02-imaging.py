#!/usr/bin/env python
# Code by Lachlan Marnoch, 2024 - 2025
import shutil

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List

import astropy.units as units
from astropy.coordinates import SkyCoord

import craftutils.utils as u
import craftutils.plotting as pl
from craftutils.params import save_params

import craftutils.observation.field as field
import craftutils.observation.filters as filters
import craftutils.observation.image as image

import lib

description = """
Generates imaging figures. This script is for internal use only, as it depends on the fully-processed imaging data being in the correct place; but it is presented here to show how the figures were generated.
"""

pl.latex_setup()

c_frb = "black"
c_text = "black"
scale = None
tick_fontsize = pl.tick_fontsize
draw_centre: bool = True
skip: bool = False
if lib.dropbox_figs is not None:
    db_path = os.path.join(lib.dropbox_figs, "imaging")
    os.makedirs(db_path, exist_ok=True)

frames = {
    "FRB20231027": 6 * units.arcsec
}
stretch = {
    # "FRB20180924B": "log",
    "FRB20171020A": "log",
    "FRB20190608B": "log",
    "FRB20191001C": "log",
    "FRB20211127I": "log",
    "FRB20211212A": "log",
    "FRB20210807D": "log",
    "FRB20220725A": "log",
    "FRB20221106A": "log",
    # "FRB20231226": "log",
    "FRB20231230": "log",
    "FRB20240201": "log",
    "FRB20240210A": "log"
}
interval = {
    "FRB20210807D": "zscale",
    "FRB20240210A": "zscale"
}

frames_eff = {}


def plot_all_bands(fld: str):
    print("=" * 100)
    print(fld)
    fld_ = field.Field.from_params(fld)
    frb_name = fld_.frb.tns_name
    if frb_name is None:
        frb_name = fld
    fld_.load_output_file()
    host = fld_.frb.host_galaxy
    if host is not None:
        host.load_output_file()
    else:
        print(f"{fld}: No host galaxy param file found.")
        return None
    z = host.z
    output_path_fig = os.path.join(lib.output_path, "imaging", fld)

    if z is not None and z > 0.:
        z_text = ", at redshift $\zhost=""" + str(z) + "$,"
    else:
        z_text = " (redshift unknown),"


    if not (skip and os.path.isfile(output_path_fig)):
        bands: List[filters.FORS2Filter] = fld_.get_filters()
        bands = list(filter(lambda fi: fi.instrument.name.startswith("vlt"), bands))
        print("\tfilters:", [b.name for b in bands])
        n_total = len(bands)
        if n_total == 0:
            return None

        bands.sort(key=lambda fi: fi.lambda_eff)

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

        deep = host.select_deepest_sep()
        if deep is None:
            print(f"No photometry found for {fld}. You might have to run the objects pipeline.")
            return None
            # fld_.proc_refine_photometry(None)
            # deep = host.select_deepest_sep()
            # if deep is None:
            #     return None
            # else:
            #     deep = host.select_deepest_sep()
        elif deep["a"] < 0:
            deep = host.select_deepest()

        deep_img = image.Image.from_fits(deep["good_image_path"])
        deep_img.extract_pixel_scale()
        global scale, c_frb, c_text
        # Pick a nice frame for this galaxy, or otherwise the scale as set by common_scale
        if scale is not None and z != 999.:
            print("CASE 1")
            frame = scale * units.kpc
        elif fld in frames:
            print("CASE 2")
            frame = frames[fld]
        else:
            print("CASE 3")
            deep_img.extract_pixel_scale()
            print(deep["a"], deep["b"], deep["kron_radius"])
            frame = deep_img.nice_frame(
                row={
                    "A_WORLD": deep["a"],
                    "B_WORLD": deep["b"],
                    "KRON_RADIUS": deep["kron_radius"]
                },
                frame=5 * units.arcsec
            ).to(units.arcsec, deep_img.pixel_scale_y)
        print("\tframe:", frame)
        print("\tpix scale", deep_img.pixel_scale_y)
        frames_eff[fld] = frame.to(units.pix, deep_img.pixel_scale_y)

        for i, band in enumerate(bands):
            print("\t", band, band.formatted_name)
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
            instrument_name = band.instrument.name
            fil_name = band.name
            img_dict = fld_.deepest_in_band(band)
            print("\t\t", img_dict)
            if instrument_name == "vlt-hawki":
                img = image.HAWKIImage(img_dict["path"])
            elif instrument_name == "vlt-fors2":
                img = image.FORS2CoaddedImage(img_dict["path"])
            else:
                raise ValueError("Not sure how we got here, but the filter name is ", band.instrument.name)
            img.load_data()
            img.load_wcs()
            if "epoch" in img_dict:
                epoch_name = img_dict["epoch"]
            else:
                print(f"{fld}: epoch not found in the field imaging dict; need to run objects pipeline.")
                return None
            phot_dict = host.photometry[instrument_name][fil_name][epoch_name]
            normalize_kwargs = {}
            if fld in stretch:
                normalize_kwargs["stretch"] = stretch[fld]
            # if fld in interval:
            normalize_kwargs["interval"] = "zscale"  # interval[fld]
            if "flux_max" in phot_dict:
                peak = phot_dict["flux_max"] + phot_dict["background_se"]
                normalize_kwargs["vmax"] = peak.value
            print(f"\t\t{normalize_kwargs}")

            ax = fig.add_subplot(n_height, n_width_this, n_this, projection=img.wcs[0])
            ra, dec = ax.coords
            ra.set_ticklabel(
                fontsize=tick_fontsize,
                exclude_overlapping=True
                # rotation=45,
                # pad=50
            )
            dec.set_ticklabel(fontsize=tick_fontsize)
            ax.set_title(f"{band.instrument.formatted_name}, {band.formatted_name}", size=14)

            galfit = None

            if "best" in host.galfit_models:
                galfit = host.galfit_models["best"]["COMP_2"]
                centre = SkyCoord(galfit["ra"], galfit["dec"])
            else:
                centre = host.position_photometry
            x, y = img.world_to_pixel(centre, origin=0)

            psf = img.extract_psf_fwhm()

            fig, ax, vals = fld_.plot_host(
                fig=fig,
                ax=ax,
                img=img,
                frame=frame,
                draw_scale_bar=True,
                scale_bar_kwargs=dict(
                    text_kwargs=dict(color=c_text),
                    size=psf,
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
                centre=centre,
                normalize_kwargs=normalize_kwargs,
            )
            dec.set_axislabel(" ")
            ra.set_axislabel(" ")

            if not is_bottom:
                ra.set_ticklabel_visible(False)
            if not is_left:
                dec.set_ticklabel_visible(False)
            print(f"\t{galfit is None =}")
            print(f"\t{draw_centre=}")
            if galfit is not None and draw_centre:
                ax.scatter(x, y, marker="x", c=c_frb)

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

        # fig.tight_layout()

        if db_path is not None:
            output_path_db = os.path.join(db_path, fld)
            fig.savefig(output_path_db + ".pdf", bbox_inches='tight')

        for f in (".pdf", ".png"):
            fig.savefig(output_path_fig + f, bbox_inches='tight')
        fig.clear()
        plt.close(fig)
        del fig

    dm_str, _, _ = u.uncertainty_string(
        value=fld_.frb.dm,
        uncertainty=fld_.frb.dm_err,
        brackets=False
    )
    dm_str = dm_str[1:-1]

    fld_nice = frb_name.replace("FRB", r"FRB\,")
    hg_nice = frb_name.replace("FRB", r"HG\,")

    intro_str = """VLT imaging of the host galaxy of """ + fld_nice + z_text + r"""
        with $\DMFRB=\DMSI{""" + dm_str + r"""}$. """

    first = fld == "FRB20180924B"

    """The black cross in the lower-right corner demonstrates the astrometric uncertainty (as defined in \autoref{pipeline:astrometry:comparison}) of the image in 
    Right Ascension and Declination; this may not be visible in all images, as the uncertainty is often small.
    """

    if first:
        caption = intro_str + r"""The black ellipse outlines the 
            1-$\sigma$ uncertainty region, as projected onto the axes of right ascension and declination, and added in 
            quadrature with the astrometric uncertainty for the particular image.
            The scale bar in each image uses the delivered PSF FWHM for the angular scale. 
            \figscript{""" + u.latex_sanitise(os.path.basename(__file__)) + r"""}"""
    else:
        caption = intro_str + (r" Markings are explained in the caption of \autoref{fig:imaging:FRB20180924B}. "
                               r"\figscript{") + u.latex_sanitise(os.path.basename(__file__)) + "}"

    if n_height > 2:
        float_spec = "p"
    else:
        float_spec = "t"

    fig_text = r"""
    \begin{figure}[""" + float_spec + r"""]
        \centering
        \includegraphics[width=\textwidth]{03_host_properties/figures/imaging/""" + fld + r""".pdf}
        \caption[VLT imaging of """ + hg_nice + r"""]{""" + caption + r"""}
        \label{fig:imaging:""" + frb_name + r"""}
    \end{figure}
    """
    return fig_text


def main(
        input_dir: str,
        output_dir: str,
        colour_frb: str,
        colour_text: str,
        common_scale: float,
        test_plot: bool,
        centre: bool,
        skip_successful: bool,
        single: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)
    output_dir_this = os.path.join(lib.output_path, "imaging")
    os.makedirs(output_dir_this, exist_ok=True)
    exclude = [
        "FRB20121102A",
        "FRB20171020A",
        "FRB20230718A",
        # "FRB20220610A",
        # "FRB20230731A"
    ]
    if single is not None:
        flds = [single]
        skip_tex = True
    elif not test_plot:
        flds = list(filter(lambda n: n.startswith("FRB") and n not in exclude, field.list_fields()))
        skip_tex = False
    else:
        flds = [
            # "FRB20240615",
            # "FRB20230731A"
            "FRB20190714A"
        ]
        skip_tex = False

    global c_frb, c_text, scale, draw_centre, skip
    c_frb = colour_frb
    c_text = colour_text
    scale = common_scale
    draw_centre = centre
    skip = skip_successful

    file_lines = []
    failed = []
    tex_path = os.path.join(lib.tex_path, "imaging_figures.tex")

    for fld in flds:
        text = plot_all_bands(fld)
        if text is not None and not test_plot:
            file_lines.append(text)
        else:
            failed.append(fld)

    if not skip_tex or test_plot:
        with open(tex_path, "w") as f:
            f.writelines(file_lines)
        shutil.copy(tex_path, os.path.join(lib.dropbox_path, "figures", "imaging"))

    print("Failed fields:")
    print(failed)

    save_params(os.path.join(lib.output_path, "frames.yaml"), frames_eff)


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
        "--frb_colour",
        help="Colour of FRB  localisation ellipse.",
        type=str,
        default="black"
    )
    parser.add_argument(
        "--text_colour",
        help="Colour of inset text.",
        type=str,
        default="white"
    )
    parser.add_argument(
        "--common_scale",
        help="Size of common scale, in kpc.",
        type=float,
        default=None
    )
    parser.add_argument(
        "--test_plot",
        help="Generates only a couple of plots, for testing purposes. Latex file generation will be skipped.",
        action="store_true"
    )
    parser.add_argument(
        "--single",
        help="Specify a single galaxy to be plotted. Latex file generation will be skipped.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--draw_centre",
        help="X marks the spot.",
        action="store_true"
    )
    parser.add_argument(
        "--skip_successful",
        help="Skip previously-successful figures?",
        action="store_true"
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        colour_frb=args.frb_colour,
        colour_text=args.text_colour,
        common_scale=args.common_scale,
        test_plot=args.test_plot,
        centre=args.draw_centre,
        skip_successful=args.skip_successful,
        single=args.single
    )
