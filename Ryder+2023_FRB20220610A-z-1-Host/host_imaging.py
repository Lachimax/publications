#!/usr/bin/env python
# Code by Lachlan Marnoch, 2022

import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

import astropy.units as units
from astropy.coordinates import SkyCoord, Angle
import astropy.table as table

from craftutils.observation import field, image
from craftutils import plotting as pl

import lib

description = """
Produces panels A, B & C of **Figure 2** and performs photometry on the imaging as specified in **S1.6**.
"""


def main(
        output_dir: str,
        input_dir: str,
        skip_photometry: bool,
):
    tick_fontsize = 13
    script_dir = os.path.dirname(__file__)
    field_name = "FRB20220610A"
    frb220610_field = field.FRBField.from_file(os.path.join(script_dir, "param", field_name, field_name))
    # Load the image files.
    g_img = image.FORS2CoaddedImage(
        os.path.join(
            input_dir,
            "FRB20220610A_VLT-FORS2_g-HIGH_combined.fits"
        )
    )
    R_img = image.FORS2CoaddedImage(
        os.path.join(
            input_dir,
            "FRB20220610A_VLT-FORS2_R-SPECIAL_2022-07-01.fits"
        )
    )
    J_img = image.HAWKICoaddedImage(
        os.path.join(
            input_dir,
            "FRB20220610A_VLT-HAWK-I_J_2022-09-29.fits"
        )
    )
    K_img = image.HAWKICoaddedImage(
        os.path.join(
            input_dir,
            "FRB20220610A_VLT-HAWK-I_Ks_2022-07-24.fits"
        )
    )
    # Set the centre of the primary aperture.
    centre = SkyCoord("23h24m17.65s -33d30m49")
    # Set up apertures for the purposes of photometry of the individual clumps.
    # Thetas are from due west = 0
    apertures = {
        "all": {
            "centre": centre,
            "a": 3 * units.arcsec,
            "b": 3 * units.arcsec,
            "theta": 45 * units.deg,
            "text_offset_factor": 0
        },
        "A": {
            "centre": SkyCoord("23h24m17.55s -33d30m49.9s"),
            "a": 1 * units.arcsec,
            "b": 1 * units.arcsec,
            "theta": -45 * units.deg,
            "text_offset_factor": 1.8
        },
        "C": {
            "centre": SkyCoord("23h24m17.75s -33d30m49.9s"),
            "a": 0.8 * units.arcsec,
            "b": 0.7 * units.arcsec,
            "theta": 80 * units.deg,
            "text_offset_factor": 2.
        },
        "B": {
            "centre": SkyCoord("23h24m17.64 -33d30m48.2"),
            "a": 1.6 * units.arcsec,
            "b": 1.1 * units.arcsec,
            "theta": 35 * units.deg,
            "text_offset_factor": 1.5
        },

    }
    # For Science, we want sans-serif plots.
    latex_kwargs = {
        "font_family": "sans-serif",
        "packages": [],
        "use_tex": False,
    }
    pl.latex_setup(**latex_kwargs)
    imgs = [g_img, R_img, K_img]
    # Set up panel titles.
    header_strings = [
        r"$g$-band",
        r"$R$-band",
        r"$K_\mathrm{s}$-band"
    ]

    def plot_science(
            vertical: bool,
            output_path: str,
            all_loc: bool = False
    ):

        if vertical:
            fig = plt.figure(figsize=(6, 12))
        else:
            fig = plt.figure(figsize=(12, 6))

        for i, img in enumerate(imgs):

            panel_id = chr(ord("A") + i)

            img.load_wcs()
            if vertical:
                ax_1 = fig.add_subplot(len(imgs), 1, i + 1, projection=img.wcs[0])
                extra_height_top_factor = 1.5
                spread_factor = 0.5
            else:
                ax_1 = fig.add_subplot(1, len(imgs), i + 1, projection=img.wcs[0])
                extra_height_top_factor = 0.8
                spread_factor = 1.5
            ra, dec = ax_1.coords

            ra.set_ticks(
                values=units.Quantity([
                    Angle("23h24m17.92s"),
                    Angle("23h24m17.67s"),
                    Angle("23h24m17.42s"),
                    # Angle("23h24m17.45s")
                ]).to("deg")
                # spacing=0.2 * units.hourangle / 3600
            )
            ra.set_ticklabel(
                fontsize=tick_fontsize,
                # rotation=45,
                # pad=50
            )
            dec.set_ticklabel(fontsize=tick_fontsize)

            ax_1.set_title(header_strings[i], size=16)

            # Draw the scale bar only on the final image.
            if i == 2:
                scale_bar_obj = frb220610_field.frb.host_galaxy
            else:
                scale_bar_obj = None

            show_frb = False
            if all_loc or i == 1:
                show_frb = True

            ax_1, fig, other_args = frb220610_field.plot_host(
                fig=fig,
                ax=ax_1,
                img=img,
                frame=4 * units.arcsec,
                imshow_kwargs={"cmap": "plasma"},
                show_frb=show_frb,
                frb_kwargs={
                    "edgecolor": "black",
                    "lw": 2,
                    "include_img_err": False
                },
                latex_kwargs=latex_kwargs,
                centre=centre,
                scale_bar_object=scale_bar_obj,
                scale_bar_kwargs={
                    "spread_factor": spread_factor,
                    "text_kwargs": {"fontsize": 14},
                    "extra_height_top_factor": extra_height_top_factor
                }
            )
            # Draw panel label
            ax_1.text(
                0.05, 0.95,
                f"{panel_id}",
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax_1.transAxes,
                c="white",
                fontsize=15
            )
            cbar = plt.colorbar(
                other_args["mapping"],
                ax=ax_1,
                location="bottom"
            )
            cbar.set_label(
                label="counts s$^{-1}$",
                size=14
            )
            cbar.ax.tick_params(
                labelsize=12,
                labelrotation=45,
            )

            # Turn off horizontal axis tick labels for vertical format.
            if vertical:
                if i < 2:
                    ra.set_ticklabel_visible(False)
            else:
                if i > 0:
                    dec.set_ticklabel_visible(False)
            # Draw apertures only on first panel.
            if i == 0:
                for clump_name in apertures:
                    aperture = apertures[clump_name]
                    aperture["obj_name"] = clump_name
                    a = aperture["a"]
                    b = aperture["b"]
                    theta = aperture["theta"]
                    text_coord = SkyCoord(
                        aperture["centre"].ra + aperture["text_offset_factor"] * 1.1 * a,
                        aperture["centre"].dec - b / 2
                    )
                    x, y = img.world_to_pixel(aperture["centre"])
                    x_text, y_text = img.world_to_pixel(text_coord)

                    if clump_name != "all":
                        ax_1.text(x_text, y_text, f"{clump_name.lower()}", fontsize=22, color="white")
                    # Draw ellipse
                    e = Ellipse(
                        xy=(x, y),
                        width=2 * a.to(units.pix, img.pixel_scale_y).value,
                        height=2 * b.to(units.pix, img.pixel_scale_y).value,
                        angle=theta.value,
                        edgecolor="white",
                        facecolor="none",
                        lw=2
                    )
                    ax_1.add_artist(e)

            # Draw X-Shooter slits
            if i == 1:
                acq_star = SkyCoord("23h24m17.084s -33d30m23.540s")
                # target = SkyCoord(acq_star.ra + 6.2 * units.arcsec, acq_star.dec - 26.3 * units.arcsec)
                target = SkyCoord("23h24m17.580s -33d30m49.840s")
                print(acq_star.to_string(style="hmsdms"))
                print(target.to_string(style="hmsdms"))
                print(frb220610_field.frb.position.to_string(style="hmsdms"))
                target_x, target_y = img.world_to_pixel(target)
                slit_width = img.pixel(1 * units.arcsec).value
                slit_length = img.pixel(11 * units.arcsec).value

                theta = 45
                rec_x = target_x + np.sin(np.deg2rad(theta)) * slit_length / 2 + np.cos(
                    np.deg2rad(theta)) * slit_width / 2
                rec_y = target_y - np.cos(np.deg2rad(theta)) * slit_length / 2 - np.sin(
                    np.deg2rad(theta)) * slit_width / 2

                print("\nRECTANGLE")
                print(rec_x, rec_y, slit_width, slit_length)
                print("\n")
                # ax_1.scatter(target_x, target_y, marker="x", c="white")
                # ax_1.scatter(rec_x, rec_y, marker="x", c="black")
                rect_1 = Rectangle(
                    (rec_x, rec_y),
                    slit_width,
                    slit_length,
                    # rotation_point="center",
                    angle=theta,
                    linewidth=2,
                    edgecolor='white',
                    facecolor='none'
                )
                ax_1.add_artist(rect_1)

                theta = 90
                rec_x = target_x + np.sin(np.deg2rad(theta)) * slit_length / 2 + np.cos(
                    np.deg2rad(theta)) * slit_width / 2
                rec_y = target_y - np.cos(np.deg2rad(theta)) * slit_length / 2 - np.sin(
                    np.deg2rad(theta)) * slit_width / 2

                print("\nRECTANGLE")
                print(rec_x, rec_y, slit_width, slit_length)
                print("\n")
                # ax_1.scatter(target_x, target_y, marker="x", c="white")
                # ax_1.scatter(rec_x, rec_y, marker="x", c="black")

                rect_2 = Rectangle(
                    (rec_x, rec_y),
                    slit_width,
                    slit_length,
                    # rotation_point="center",
                    angle=90,
                    linewidth=2,
                    edgecolor='white',
                    facecolor='none'
                )
                ax_1.add_artist(rect_2)

        fig.savefig(output_path + ".pdf", bbox_inches='tight')
        fig.savefig(output_path + ".png", bbox_inches='tight')

    plot_science(vertical=True, output_path=os.path.join(output_dir, "FRB20220610A_gRK_vertical"))
    plot_science(vertical=False, output_path=os.path.join(output_dir, "FRB20220610A_gRK_horizontal"))
    plot_science(
        vertical=False,
        all_loc=True,
        output_path=os.path.join(output_dir, "FRB20220610A_gRK_horizontal_all_loc")
    )

    if not skip_photometry:

        imgs = [
            g_img,
            R_img,
            J_img,
            K_img
        ]

        clumps = []

        for img in imgs:
            print(img.filter_name)
            band_name = img.filter.band_name
            for clump_name in apertures:
                aperture = apertures[clump_name]
                # Extract photometry using SEP
                phot = img.sep_elliptical_magnitude(
                    centre=aperture["centre"],
                    a_world=aperture["a"],
                    b_world=aperture["b"],
                    theta_world=-aperture["theta"],  # Reverse theta for SEP convention
                    mask_nearby=False,
                    output=os.path.join(
                        output_dir,
                        f"frb20220610A_{clump_name}_{img.filter_name}"
                    )
                )

                phot_better = {
                    f"mag_{band_name}": phot["mag"][0],
                    f"mag_err_{band_name}": phot["mag_err"][0],
                    f"snr_{band_name}": phot["snr"][0]
                }
                aperture.update(phot_better)

                plt.title(f"{img.instrument.nice_name()}, {img.filter.nice_name()}")
                print(clump_name, ":", phot["mag"][0], "+/-", phot["mag_err"][0], "; SNR ==", phot["snr"][0])
                clumps.append(apertures[clump_name])

        # Write table of photometry to disk.
        clump_tbl = table.QTable(clumps)
        clump_tbl.write(os.path.join(output_dir, "photometry_table.csv"), overwrite=True)
        clump_tbl.write(os.path.join(output_dir, "photometry_table.ecsv"), overwrite=True)


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
        "--skip_photometry",
        help="Skip photometric measurements (they take a while) and just do plotting.",
        action="store_true"
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    skip_photometry = args.skip_photometry
    main(
        output_dir=output_path,
        input_dir=input_path,
        skip_photometry=skip_photometry
    )
