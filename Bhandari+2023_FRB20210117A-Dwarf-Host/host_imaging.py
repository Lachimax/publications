#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import matplotlib.pyplot as plt

from astropy import units

from craftutils.observation import field, image
from craftutils.utils import mkdir_check_nested

import lib

description = "Produces the left-hand panel of **Figure 1**."


def main(
        output_dir: str,
        input_dir: str,
):
    # Establish field object
    script_dir = os.path.dirname(__file__)
    field_name = "FRB20210117"
    field_path = os.path.join(script_dir, "param", field_name, field_name)
    field_frb20210117 = field.Field.from_file(
        field_path
    )

    # Establish image object
    img_fors2_I = image.FORS2CoaddedImage(os.path.join(input_dir, "FRB20210117_VLT-FORS2_I-BESS_2021-06-12.fits"))

    # Use image object to plot FRB host, including error ellipse
    for img in [
        img_fors2_I,
    ]:
        ax, fig, other = field_frb20210117.plot_host(
            # fig=fig,
            img=img,
            draw_scale_bar=True,
            frame=3.7 * units.arcsec,
            normalize_kwargs={
                "stretch": "sqrt",
                "interval": "minmax"
            },
            frb_kwargs={
                "lw": 2
            },
            scale_bar_kwargs={
                "text_kwargs": {"fontsize": 16}
            }
        )
        ra, dec = ax.coords
        ra.set_axislabel("Right Ascension (J2000)", fontsize=15)
        dec.set_axislabel("Declination (J2000)", fontsize=15, minpad=-1)
        ax.tick_params(labelsize=12)
        plt.text(0.05, 0.88, f"z = {field_frb20210117.frb.host_galaxy.z}", transform=ax.transAxes, c="white",
                 fontsize=15)
        plt.tight_layout()
        mkdir_check_nested(output_dir, remove_last=False)
        fig.savefig(fname=os.path.join(output_dir, f"HG210117{img.filter.band_name.lower()}.pdf"), bbox_inches="tight")


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
