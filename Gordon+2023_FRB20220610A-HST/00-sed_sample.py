#!/usr/bin/env python
# Code by Lachlan Marnoch, 2023

import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as units
import astropy.table as table
import astropy.coordinates as coordinates

import craftutils.utils as u
import craftutils.params as p
import craftutils.plotting as pl
import craftutils.observation.image as image
import craftutils.observation.instrument as instrument
import craftutils.observation.sed as sed

import lib

description = """
Runs PATH (Probabilistic Association of Transients; Aggarwal et al 2021) in various configurations on the HST imaging 
data covering the field of FRB 20220610A, and generates some figures.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    # Set up instruments and bands
    hst_ir = instrument.Instrument.from_params("hst-wfc3_ir")
    f160w = hst_ir.filters["F160W"]
    hst_uv = instrument.Instrument.from_params("hst-wfc3_uvis2")
    f606w = hst_uv.filters["F606W"]

    # fors2 = instrument.Instrument.from_params("vlt-fors2")
    # g_fors2 = fors2.filters["g_HIGH"]
    # I_fors2 = fors2.filters["I_BESS"]

    bands = [f160w, f606w]

    # Calculate redshifted SED model sample
    sample = lib.sample
    sample.z_displace_sample(
        bands=bands,
        z_max=5.,
        n_z=500,
        save_memory=True
    )
    sample.write_z_mag_tbls()
    sample.update_output_file()

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
