#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os
import requests

import numpy as np

from astropy.time import Time
from astropy import table, units
from astropy.coordinates import SkyCoord

import frb.dm.igm as igm

import craftutils.utils as u
import craftutils.param as p
from craftutils.observation import objects

import lib

description = """
Uses the FRBSTATS catalogue of FRBs (https://www.herta-experiment.org/frbstats/) to enumerate those that have DM 
consistent with z>1.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    output_this = os.path.join(lib.output_path, "FRBStats")
    u.mkdir_check(output_this)

    frbstats_response = requests.get("https://www.herta-experiment.org/frbstats/catalogue.csv").content
    frbstats_path = os.path.join(output_this, f"FRBSTATS-{Time.now()}.csv".replace(" ", "T"))
    with open(frbstats_path, "wb") as file:
        file.write(frbstats_response)

    frbstats = table.QTable.read(frbstats_path)
    frbstats["dm"] *= objects.dm_units

    frbstats_1000 = frbstats[frbstats["dm"] > 1000 * objects.dm_units]
    frbstats_1000_write = frbstats_1000.copy()
    frbstats_1000_write.write(os.path.join(output_this, "FRBSTATS_DM1000.ecsv"), overwrite=True)

    dm_host = 50 * objects.dm_units
    dm_mw_halo = 40 * objects.dm_units
    # frbs = []
    dm_ism = []
    dm_igm = []
    z_from_dm = []
    z_from_dm_nohost = []
    # z_factor = 1 / (923 * objects.dm_units)

    frbstats.sort("frb")
    n = len(frbstats)
    for i, row in enumerate(frbstats):
        frb = objects.FRB(
            dm=row["dm"],
            position=SkyCoord(row["ra"], row["dec"], unit=(units.hourangle, units.deg))
        )

        dm_ism_this = frb.dm_mw_ism()
        dm_ism.append(dm_ism_this)

        try:
            z = float(row["redshift"])

            dm_nohost_this = row["dm"] - dm_mw_halo - dm_ism_this
            dm_igm_this = dm_nohost_this - (dm_host / (1 + z))

        except ValueError:
            dm_nohost_this = 0 * objects.dm_units
            dm_igm_this = 0 * objects.dm_units

        dm_igm.append(dm_igm_this)
        # Ignore nearby FRBs
        if dm_igm_this >= 101 * objects.dm_units:
            z_from_dm_this = igm.z_from_DM(dm_igm_this)
        else:
            z_from_dm_this = 0.

        z_from_dm.append(z_from_dm_this)

        if dm_nohost_this >= 101 * objects.dm_units:
            z_from_dm_nohost_this = igm.z_from_DM(dm_nohost_this)
            z_from_dm_nohost.append(z_from_dm_nohost_this)
        else:
            z_from_dm_nohost.append(0.)

        del frb
        # frbs.append(frb)
        print(f"Processed {i} / {n}: {row['frb']}:", z_from_dm_this, dm_igm_this)
    # print(len(frbs))
    # frbstats["frbobject"] = frbs
    frbstats["dm_ism"] = dm_ism
    frbstats["dm_igm"] = dm_igm
    frbstats["z_from_dm"] = z_from_dm
    frbstats["z_from_dm_nohost"] = z_from_dm_nohost

    frbstats_z1 = frbstats[frbstats["z_from_dm"] > 1.]

    values = {
        "n_z_1": len(frbstats_z1)
    }
    print("Number of FRBs with DM consistent with z > 1:")
    print(values["n_z_1"])

    frbstats_write = frbstats.copy()
    frbstats_write.write(os.path.join(output_this, "FRBSTATS_mod.ecsv"), overwrite=True)
    frbstats_z1.write(os.path.join(output_this, "FRBSTATS_z1.ecsv"), overwrite=True)


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
