#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np

from astropy import constants

import craftutils.utils as u

import lib

description = """
Analysis of whether FGe is actually int he foreground.
"""

def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    halo_tbl = lib.read_master_table()
    HG_row = halo_tbl[0]
    FGe_row = halo_tbl[5]
    print(FGe_row["id"], FGe_row["z"])

    z_hg = HG_row["z"]
    z_fge = FGe_row["z"]
    z_avg = np.mean([z_hg, z_fge])
    delta_z = z_hg - z_fge
    print(f"{delta_z=}")

    delta_v = lib.peculiar_velocity(z_hg, z_avg).to("km/s") - peculiar_velocity(z_fge, z_avg).to("km/s")
    print(f"{delta_v=}")

    z_And = -0.001004
    v_And = constants.c * z_And
    print("Consider Andromeda:", v_And.to("km/s"))

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
