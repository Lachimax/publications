#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import matplotlib.pyplot as plt

# import astropy.
import astropy.units as units
import astropy.table as table

import craftutils.utils as u
import craftutils.observation.field as field
import craftutils.params as p
import numpy as np

import lib

description = """
Does Monte Carlo modelling of foreground halos.
"""


def main(
        output_dir: str,
        input_dir: str,
        rel: str,
        n_real: float
):

    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb190714 = lib.fld

    step_size_halo = 1 * units.kpc
    rmax = 1

    for n in range(n_real):
        np.random.seed(1994 + n)
        print("=" * 50)
        print("Iteration", n + 1, "of", n_real)
        properties, halo_tbl = frb190714.frb.foreground_halos(
            rmax=rmax,
            step_size_halo=step_size_halo,
            cat_search="panstarrs1",
            skip_other_models=True,
            load_objects=False,
            do_mc=True,
            smhm_relationship=rel,
            do_profiles=False
        )
        # halo_tbl = properties.pop("halo_table")
        # dm_tbl = properties.pop("dm_cum_table")
        halo_tbl.sort("offset_angle")
        properties.pop("halo_dm_cum")
        properties.pop("halo_models")

        # p.save_params(os.path.join(output_path, ""), properties)
        # lib.write_dm_table_mc(tbl=dm_tbl, n=n)
        lib.write_halo_table_mc(tbl=halo_tbl, n=n, relationship=rel)
        lib.write_properties_mc(properties, n=n, relationship=rel)


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
        "-r",
        help="SMHM relationship.",
        type=str,
        default="K18"
    )
    parser.add_argument(
        "-n",
        help="Number of instantiations.",
        type=int,
        default=10000
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        rel=args.r,
        n_real=args.n
    )
