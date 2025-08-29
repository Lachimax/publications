# Code by Lachlan Marnoch, 2023

import matplotlib.pyplot as plt
import astropy.units as units
from astropy.coordinates import SkyCoord
import craftutils.observation.field as field
import craftutils.observation.image as image
import craftutils.observation.filters as filters
import craftutils.params as p
import craftutils.utils as u
import craftutils.plotting as pl
import os
import numpy as np

import lib

description = "Generates the Imaging plots seen in Shannon+2024."


def main(
        output_dir: str,
        cmap: str,
        colour_frb: str,
        colour_text: str,
        common_scale: float,
        test_plot: bool
):
    exclude = [
        "FRB20121102A",
        "FRB20171020A",
        "FRB20210407A"
        "FRB20210912A",
        "FRB20230731A",
        "FRB20231027",
        # "FRB20231226",
        "FRB20231230",
        "FRB20240117",
        "FRB20240210A",
        # "FRB20240310"
    ]
    # flds = ["FRB20171020A"]
    if not test_plot:
        flds = list(filter(lambda n: n.startswith("FRB") and n not in exclude, field.list_fields()))
    else:
        flds = ["FRB20181112A", "FRB20220105A", "FRB20190711A"]

    fld_dict = {}
    img_dict = {}
    deep_dict = {}
    positions = {}
    z_dict = {}

    frames = {
        # "FRB20190711A": 12 * units.pix,
        # "FRB20181112A": 30 * units.pix,
        "FRB20191001C": 30 * units.pix,
        "FRB20191228A": 15 * units.pix,
        "FRB20210807D": 45 * units.pix,
        "FRB20220501C": 20 * units.pix,
        "FRB20190714A": 20 * units.pix,
        "FRB20210912A": 11 * units.arcsec,
        "FRB20221106A": 5 * units.arcsec,
        "FRB20230718A": 7.5 * units.arcsec,
        # "FRB20230526A": 5 * units.arcsec
        # "FRB20210320":
    }

    scale = {
        # "FRB20180924B": "log",
        "FRB20171020A": "log",
        "FRB20190608B": "log",
        "FRB20191001C": "log",
        "FRB20211127I": "log",
        "FRB20211212A": "log",
        "FRB20210807D": "log",
        "FRB20220725A": "log"
    }

    ics = []

    for i, fld_name in enumerate(flds):
        print(fld_name)
        fld = field.FRBField.from_params(fld_name)
        if not fld.survey or fld.survey.name != "CRAFT_ICS" or fld_name > "FRB20240318A":
            print("\tSkipping (not CRAFT)")
            continue
        if isinstance(fld.frb.tns_name, str):
            ics.append(fld.frb.tns_name)
        else:
            ics.append(fld_name)
        fld.frb.get_host()
        z = fld.frb.host_galaxy.z
        if z == 0. or z is None:
            z = 999.
        z_dict[fld_name] = z
        host = fld.frb.host_galaxy
        host.load_output_file()
        # if fld_name == "FRB20190102C":
        #     deep = {
        #         "band": "g_HIGH",
        #         "image_path": "/home/lachlan/Data/FRB20190102/imaging/vlt-fors2/2019-01-12-FRB20190102_1/FRB20190102_VLT-FORS2_g-HIGH_2019-01-12.fits",
        #         "a": 0 * units.arcsec,
        #         "b": 0 * units.arcsec,
        #         "kron_radius": 1,
        #         "epoch_name": "FRB20190102_FORS2_1"
        #     }
        # elif fld_name == "FRB20191228A":
        #     deep = {
        #         "band": "I_BESS",
        #         "image_path": "/home/lachlan/Data/FRB20191228/imaging/vlt-fors2/2020-09-21-FRB20191228_1/FRB20191228A_VLT-FORS2_I-BESS_2020-09-21.fits",
        #         "a": 0 * units.arcsec,
        #         "b": 0 * units.arcsec,
        #         "kron_radius": 1,
        #         "epoch_name": "FRB20191228_FORS2_1"
        #     }
        # elif fld_name == "FRB20210117A":
        #     deep = {
        #         "band": "I_BESS",
        #         "image_path": "/home/lachlan/Data/FRB20210117/imaging/vlt-fors2/2021-06-12-FRB20210117_1/FRB20210117A_VLT-FORS2_I-BESS_2021-06-12.fits",
        #         "a": 0 * units.arcsec,
        #         "b": 0 * units.arcsec,
        #         "kron_radius": 1,
        #         "epoch_name": "FRB20210117_FORS2_1"
        #     }
        # elif fld_name == "FRB20220105A":
        #     deep = {
        #         "band": "R_SPECIAL",
        #         "image_path": "/home/lachlan/Data/FRB20220105A/imaging/vlt-fors2/FRB20220105A_FORS2_2/FRB20220105A_VLT-FORS2_R-SPECIAL_2022-03-01.fits",
        #         "a": 0 * units.arcsec,
        #         "b": 0 * units.arcsec,
        #         "kron_radius": 1,
        #         "epoch_name": "FRB20220105A_FORS2_2"
        #     }
        # elif fld_name == "FRB20210807D":
        #     deep = {
        #         "band": "I_BESS",
        #         "image_path": "/home/lachlan/Data/FRB20210807/imaging/vlt-fors2/2021-09-06-FRB20210807_1/FRB20210807D_VLT-FORS2_I-BESS_2021-09-06.fits",
        #         "a": 0 * units.arcsec,
        #         "b": 0 * units.arcsec,
        #         "kron_radius": 1,
        #         "epoch_name": "FRB20210807_FORS2_1"
        #     }
        # elif fld_name == "FRB20220501C":
        #     deep = {
        #         "band": "I_BESS",
        #         "image_path": "/home/lachlan/Data/FRB20220501C/imaging/vlt-fors2/FRB20220501C_FORS2_1/FRB20220501C_VLT-FORS2_I-BESS_2022-05-29.fits",
        #         "a": 0 * units.arcsec,
        #         "b": 0 * units.arcsec,
        #         "kron_radius": 1,
        #         "epoch_name": "FRB2020501C_FORS2_1"
        #     }
        # elif fld_name == "FRB20221106A":
        #     deep = {
        #         "band": "K",
        #         "image_path": "/home/lachlan/Data/FRB20221106A/imaging/vlt-hawki/FRB20221106A_HAWKI_1/FRB20221106A_VLT-HAWK-I_Ks_2022-12-18.fits",
        #         "a": 0 * units.arcsec,
        #         "b": 0 * units.arcsec,
        #         "kron_radius": 1,
        #         "epoch_name": "FRB20221106A_FORS2_1"
        #     }
        # # elif fld_name == "FRB20230526A":
        # #     deep = {
        # #         "band": "K",
        # #         "image_path": "/home/lachlan/Data/FRB20230526A/imaging/vlt-hawki/FRB20230526A_HAWKI_2/FRB20230526A_VLT-HAWK-I_Ks_2023-08-03.fits",
        # #         "a": 0 * units.arcsec,
        # #         "b": 0 * units.arcsec,
        # #         "kron_radius": 1,
        # #         "epoch_name": "FRB20230526A_HAWKI_2",
        # #     }
        # elif fld_name == "FRB20210912A":
        #     deep = {
        #         "band": "R",
        #         "image_path": "/home/lachlan/Data/publications/Marnoch+2023_FRB20210912A-Missing-Host/input/imaging/FRB20210912A_VLT-FORS2_R-SPECIAL_2021-10-09_subbed_trimmed.fits",
        #         "a": 0 * units.arcsec,
        #         "b": 0 * units.arcsec,
        #         "kron_radius": 1,
        #         "epoch_name": "FRB20210912A_FORS2_2",
        #         "position": fld.frb.position
        #     }
        if fld_name == "FRB20230718A":
            deep = {
                "band": "z",
                "image_path": "/home/lachlan/Data/FRB20230718A/imaging/DECaps/cutout-z.fits",
                "a": 0 * units.arcsec,
                "b": 0 * units.arcsec,
                "kron_radius": 1,
                "epoch_name": "FRB20230718A_DECAPS_1",
                "position": SkyCoord("08h32m38.758 -40d27m04.882")
            }
        # elif fld_name == "FRB20180924B":
        #     position = SkyCoord(
        #         fld.frb.host_galaxy.position.ra,
        #         fld.frb.host_galaxy.position.dec + 0.5 * units.arcsec
        #     )
        #     deep = {
        #         "band": "g",
        #         "image_path": "/home/lachlan/Data/FRB20180924/imaging/vlt-fors2/FRB20180924B_FORS2_combined/FRB20180924B_VLT-FORS2_g-HIGH_combined.fits",
        #         "a": 0 * units.arcsec,
        #         "b": 0 * units.arcsec,
        #         "kron_radius": 1,
        #         "epoch_name": "FRB20180924B_FORS2_combined",
        #         "position": position
        #     }
        else:
            deep = host.select_deepest_sep()
        if deep is None:
            fld.proc_refine_photometry(None)
            deep = host.select_deepest_sep()
            if deep is None:
                continue
            else:
                deep = host.select_deepest_sep()
        best_path = deep["image_path"]
        if "ra" in deep and "dec" in deep:
            position = SkyCoord(deep["ra"], deep["dec"])
        elif "position" in deep:
            position = deep["position"]
        else:
            position = host.position
        positions[fld_name] = position
        deep_dict[fld_name] = deep
        img = image.Image.from_fits(best_path)
        if fld_name == "FRB20230718A":
            img.instrument_name = "decam"
            img.filter = filters.Filter.from_params("z", "decam")
        img_dict[fld_name] = img
        fld_dict[fld_name] = fld

    flds_z = list(fld_dict.keys())
    flds_z.sort(key=lambda f: z_dict[f])
    # z_dict["FRB20220610A"] = 999.

    print()
    print("=" * 15)
    print("Number of fields to plot:", len(flds_z))
    print("=" * 15)
    print()

    disp = {
        "panstarrs1": "PS1",
        "vlt-fors2": "VLT",
        "vlt-hawki": "VLT",
        "decam": "DECam"
    }

    output_this = os.path.join(output_dir, "host-stamps")
    u.mkdir_check(output_this)

    # latex_kwargs = {
    #     "font_family": "Helvetica",
    #     # "font.sans-serif": "Helvetica",
    #     # "math_fontset": "dejavusans",
    #     # "packages": ["helvet"],
    #     "use_tex": True,
    # }
    # pl.latex_setup(**latex_kwargs)

    for i, fld_name in enumerate(flds_z):
        fld = fld_dict[fld_name]
        img = img_dict[fld_name]
        img.extract_filter()

        deep = deep_dict[fld_name]
        z = z_dict[fld_name]

        best_epoch_name = deep["epoch_name"]

        if common_scale is not None and z != 999.:
            frame = common_scale * units.kpc
            print(frame)
        elif fld_name in frames:
            frame = frames[fld_name]
        else:
            frame = img.nice_frame(
                row={
                    "A_WORLD": deep["a"],
                    "B_WORLD": deep["b"],
                    "KRON_RADIUS": deep["kron_radius"]
                },
                frame=20 * units.pix
            )

        print(z)

        position = positions[fld_name]
        print(img.extract_pixel_scale())
        frame_pix = img.pixel(frame, z=z)

        frame_arc = frame_pix.to(units.arcsec, img.pixel_scale_y)
        print(frame, img.pixel(frame, z=z), frame_arc)
        print(i, fld.name, best_epoch_name, img.filter_name)
        print(img.path)
        print(img.instrument_name, img.filter_name)
        print(img.extract_astrometry_err())
        print(type(img))

        f = img.extract_filter()
        normalize_kwargs = {"interval": "zscale"}
        if f in fld.frb.host_galaxy.plotting_params:
            if "normalize" in fld.frb.host_galaxy.plotting_params[f]:
                normalize_kwargs = fld.frb.host_galaxy.plotting_params[f]["normalize"]

        if fld_name in scale:
            normalize_kwargs["stretch"] = scale[fld_name]
        # if "vmin" in normalize_kwargs:
        #     normalize_kwargs.pop("vmin")
        # normalize_kwargs["vmin"] = "median"

        # Individual imaging plot
        box = 2.25  # * pl.textwidths["PASA"] / 3
        fig_2 = plt.figure(figsize=(
            box, box
        ))

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
        })

        if "flux_max" in deep:
            peak = deep["flux_max"] + deep["background_se"]
            normalize_kwargs["vmax"] = peak.value

        fig_2, ax_2, other = fld.plot_host(
            fig=fig_2,
            img=img,
            frame=frame,
            centre=position,
            normalize_kwargs=normalize_kwargs,
            imshow_kwargs={
                "cmap": cmap
            },
            show_frb=False,
            draw_scale_bar=True,
            scale_bar_kwargs={
                "x_ax": 0.05, "y_ax": 0.05,
                "text_kwargs": {"fontsize": 10},
                "bold": True
            },
            do_latex_setup=False,
            # latex_kwargs=latex_kwargs
        )

        fld.frb_ellipse_to_plot(
            ax=ax_2,
            img=img,
            colour=colour_frb,
            include_img_err=False,
            frb_kwargs={"lw": 2}
        )
        if fld.frb.tns_name is None:
            frb_name = fld.frb.name
        else:
            frb_name = fld.frb.tns_name
        if z < 999:
            ax_2.text(
                s=f"\\textbf{{{frb_name.replace('FRB', '')}}}" + f"\n$\mathbf{{z = {z}}}$",
                x=0.05,
                y=0.95,
                c=colour_text,
                ha='left',
                va="top",
                fontsize=10,
                transform=ax_2.transAxes,
            )
        else:
            ax_2.text(
                s=f"\\textbf{{{frb_name.replace('FRB', '')}}}",
                x=0.05,
                y=0.95,
                c=colour_text,
                ha='left',
                va="top",
                fontsize=10,
                transform=ax_2.transAxes,
            )
        # ax_2.text(
        #     0.05,
        #     0.95,
        #     f"$z$ = {z}",
        #     transform=ax_2.transAxes, c=colour_text,
        #     fontsize=10,
        #     verticalalignment="top"
        # )
        ax_2.text(
            0.95,
            0.95,
            f"\\textbf{{{disp[img.instrument_name]}}}\n\\textbf{{{img.filter.band_name}}}",
            transform=ax_2.transAxes,
            c=colour_text,
            fontsize=10,
            horizontalalignment="right", verticalalignment="top"
        )

        # ax_2.set_ylabel("Declination (J2000)", fontsize=10, position="right")

        ra, dec = ax_2.coords
        # print(dir(dec))
        ra.tick_params(direction="in")
        # ra.set_axislabel_position("top")
        ra.set_axislabel(r"\textbf{Right Ascension (J2000)}", fontsize=10)
        ra.set_ticklabel_position("top")
        ra.set_ticklabel(
            exclude_overlapping=True,
        )

        dec.set_ticklabel_position("right")
        dec.tick_params(direction="in")
        dec.set_ticklabel(
            rotation="vertical",  # -90,
            # rotation_mode="anchor",
            # pad=-0.1,
            exclude_overlapping=True,
            # verticalalignment="bottom",
            # ha="right"
        )

        # dec.set_axislabel_position("right")
        dec.set_axislabel(
            r"\textbf{Declination (J2000)}",
            # labelpad=ylabelpad,
            fontsize=10,
            # rotation=90,
        )

        fig_2.savefig(
            os.path.join(
                output_this,
                f"host_{fld.name}_{img.filter.band_name}_{cmap}_{common_scale}.png"
            ),
            # bbox_inches="tight",
            dpi=200
        )
        fig_2.savefig(
            os.path.join(
                output_this,
                f"host_{fld.name}_{img.filter.band_name}_{cmap}_{common_scale}.pdf"
            ),
            # bbox_inches="tight"
        )

    n = 1
    print("\n", "=" * 15, "\n")
    print("FRB hosts in order of redshift:\n")
    for fld in flds_z:
        print(fld)
        if n % 3 == 0:
            print("\n")
        n += 1

    if not test_plot:
        p.save_params(os.path.join(output_dir, "fields.yaml"), {"ics": ics})


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="reproduces..."
                    "(arXiv: TBC)"
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
        "--cmap",
        help="Colour map.",
        type=str,
        default="cmr.bubblegum"
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
        help="Generates only one postage stamp, for testing purposes.",
        action="store_true"
    )
    args = parser.parse_args()

    main(
        output_dir=args.o,
        cmap=args.cmap,
        colour_frb=args.frb_colour,
        colour_text=args.text_colour,
        common_scale=args.common_scale,
        test_plot=args.test_plot
    )
