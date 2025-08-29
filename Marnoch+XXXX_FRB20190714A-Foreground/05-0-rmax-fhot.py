#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import matplotlib.pyplot as plt

# import astropy.
import astropy.units as units
import astropy.table as table
from astropy.visualization import ImageNormalize, LogStretch

import craftutils.utils as u
import craftutils.observation.field as field
import craftutils.params as p
import numpy as np

import lib

description = """
Does the grid modelling to constrain Rmax and fhot.
"""


def main(
        output_dir: str,
        input_dir: str,
        rel: str,
        # n_real: float
):

    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb190714 = lib.fld

    # rel = "K18"
    step_size_halo = 1 * units.kpc
    rmaxes = lib.rmaxes
    fs_hot = lib.fs_hot

    halo_tbls_all = []

    dms_halos = np.zeros(shape=(len(rmaxes), len(fs_hot))) * lib.dm_units

    for i, rmax in enumerate(rmaxes):
        # np.random.seed(1994 + n)
        print()
        print("=" * 50)
        print("=" * 50)
        print(f"R_max = {rmax} (iteration {i + 1} of {len(rmaxes)})")
        print("=" * 50)
        print("=" * 50)

        for j, f_hot in enumerate(fs_hot):
            print()
            print("=" * 50)
            print(f"R_max = {rmax} (iteration {i + 1} of {len(rmaxes)})")
            print(f"f_hot = {f_hot} (iteration {j + 1} of {len(rmaxes)})")
            print("=" * 50)

            properties, halo_tbl = frb190714.frb.foreground_halos(
                rmax=rmax,
                step_size_halo=step_size_halo,
                skip_other_models=True,
                load_objects=False,
                # do_mc=True,
                smhm_relationship=rel,
                fhot=f_hot,
                cosmic_tbl=None,
                do_incidence=False,
                do_profiles=False
            )
            # halo_tbl = properties.pop("halo_table")
            # dm_tbl = properties.pop("dm_cum_table")
            halo_tbl.sort("offset_angle")
            properties.pop("halo_dm_cum")
            properties.pop("halo_models")
            properties["dm_mw_halo"] = frb190714.frb.dm_mw_halo(model="pz19", distance=rmax, model_kwargs=dict(fhot=f_hot))
            print("DM_MW_Halo ==", properties["dm_mw_halo"])

            # p.save_params(os.path.join(output_path, ""), properties)
            # lib.write_dm_table_mc(tbl=dm_tbl, n=n)
            lib.write_halo_table(tbl=halo_tbl, fhot=f_hot, rmax=rmax, relationship=rel)
            lib.write_properties(properties, fhot=f_hot, rmax=rmax, relationship=rel, simple=True)

            halo_tbls_all.append(halo_tbl)

            dms_halos[j, i] = properties["dm_halos_inclusive"] + properties["dm_mw_halo"]

            print("DM_halos_all ==", properties["dm_halos_inclusive"])

    np.save(lib.constraints_npy_path(relationship=rel), dms_halos.value)

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

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        rel=args.r,
    )
