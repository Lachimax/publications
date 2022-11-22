"""
Code by Lachlan Marnoch, 2022

Produces the figures seen in **Figure 2a**, and performs photometry on the imaging as specified in **S1.6**,
of Ryder et al 2022 (arXiv: https://arxiv.org/abs/2210.04680)

Prerequisites:
- `craft-optical-followup`; tested on `ryder+2022` branch (latest commit):
    https://github.com/Lachimax/craft-optical-followup/tree/ryder+2022
- `astropy`; tested with version `5.0.4`
- `matplotlib`; tested with version `3.5.2`
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import astropy.units as units
from astropy.coordinates import SkyCoord
import astropy.table as table

from craftutils.observation import field, image
from craftutils import plotting as pl


def main(
        output_dir: str,
        input_dir: str,
):
    frb220610_field = field.FRBField.from_params("FRB20220610A")
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
        r"VLT/FORS2, $g$",
        r"VLT/FORS2, $R$",
        r"VLT/HAWK-I, $K_\mathregular{short}$"
    ]

    def plot_science(
            vertical: bool,
            output_path: str
    ):

        if vertical:
            fig = plt.figure(figsize=(6, 12))
        else:
            fig = plt.figure(figsize=(12, 6))

        for i, img in enumerate(imgs):
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

            # Draw the scale bar only on the final image.
            if i == 2:
                scale_bar_obj = frb220610_field.frb.host_galaxy
            else:
                scale_bar_obj = None

            frb220610_field.plot_host(
                fig=fig,
                ax=ax_1,
                img=img,
                frame=4 * units.arcsec,
                imshow_kwargs={"cmap": "plasma"},
                show_frb=(i == 1),
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
                        ax_1.text(x_text, y_text, clump_name, fontsize=22, color="white")
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
            ax_1.set_title(header_strings[i])

        fig.savefig(output_path + ".pdf", bbox_inches='tight')
        fig.savefig(output_path + ".png", bbox_inches='tight')

    plot_science(vertical=True, output_path=os.path.join(output_dir, "FRB20220610A_gRK_vertical"))
    plot_science(vertical=False, output_path=os.path.join(output_dir, "FRB20220610A_gRK_horizontal"))

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

    paper_name = "Ryder+2022_FRB20220610A"
    default_data_dir = os.path.join(
        os.path.expanduser("~"),
        "Data",
        "publications",
        paper_name
    )
    default_output_path = os.path.join(
        default_data_dir, "output"
    )
    default_input_path = os.path.join(
        default_data_dir, "input"
    )

    parser = argparse.ArgumentParser(
        description="Produces the figures seen in **Figure 2a**, and performs photometry on the imaging as specified in"
                    " **S1.6** of Ryder et al 2022 (arXiv: https://arxiv.org/abs/2210.04680)"
    )
    parser.add_argument(
        "-o",
        help="Path to output directory.",
        type=str,
        default=default_output_path
    )
    parser.add_argument(
        "-i",
        help="Path to directory containing input files.",
        type=str,
        default=default_input_path
    )

    args = parser.parse_args()

    main(
        input_dir=args.i,
        output_dir=args.o
    )
