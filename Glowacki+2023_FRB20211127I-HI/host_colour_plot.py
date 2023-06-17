#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np
import matplotlib.pyplot as plt

from craftutils.observation import field, image

import lib

description = "Reproduces the colour optical image used in the background of **Figure X** of Glowacki et al 2022."


def main(
        input_dir: str,
        output_dir: str,
):
    field_name = "FRB20211127"
    epoch_name = "FRB20211127_FORS2_1"

    script_dir = os.path.dirname(__file__)
    field_frb20220610 = field.Field.from_file(os.path.join(script_dir, "param", field_name, field_name))

    fig = plt.figure(figsize=(6, 6))
    epoch = field_frb20220610.epoch_from_params(
        epoch_name=epoch_name,
        instrument="vlt-fors2"
    )
    epoch.load_output_file()
    print("Plotting...")

    vmaxes = {}
    for f in epoch.coadded_unprojected:
        if f in field_frb20220610.frb.host_galaxy.plotting_params:
            if "normalize" in field_frb20220610.frb.host_galaxy.plotting_params[f]:
                normalize_kwargs = field_frb20220610.frb.host_galaxy.plotting_params[f]["normalize"]
                if "vmax" in normalize_kwargs:
                    vmaxes[f] = normalize_kwargs["vmax"]

    vmaxes = (
        vmaxes["I_BESS"],
        np.mean([vmaxes["I_BESS"], vmaxes["g_HIGH"]]),
        vmaxes["g_HIGH"]
    )

    g_img = image.FORS2CoaddedImage(
        os.path.join(
            input_dir,
            "FRB20211127_VLT-FORS2_g-HIGH_2022-01-29.fits"
        )
    )

    I_img = image.FORS2CoaddedImage(
        os.path.join(
            input_dir,
            "FRB20211127_VLT-FORS2_I-BESS_2022-01-29.fits"
        )
    )

    ax, fig, colour = field_frb20220610.plot_host_colour(
        fig=fig,
        red=I_img,
        blue=g_img,
        output_path=os.path.join(output_dir, f"{field_name}_colour.png"),
        frame=300,
        vmaxes=vmaxes,
        show_coords=True,
        scale_to_jansky=True,
        scale_to_rgb=True,
        show_frb=False
    )
    fig.savefig(os.path.join(output_dir, f"{field_name}_colour_tight.png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(output_dir, f"{field_name}_colour_tight.pdf"), bbox_inches="tight", dpi=200)
    print("Done.")


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
