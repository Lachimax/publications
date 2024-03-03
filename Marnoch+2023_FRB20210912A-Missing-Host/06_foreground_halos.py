#!/usr/bin/env python
# Code by Lachlan Marnoch, 2023

import os

import numpy as np
import yaml

from astropy import table, units, coordinates, cosmology

import craftutils.utils as u
import craftutils.params as p
from craftutils.observation import field, objects

import lib

description = """
Uses GAMA DR4 (Driver et al 2022) and mNFW profiles (Prochaska & Zheng 2019) to estimate a partial foreground halo 
contribution to the DM of FRB 20210912A.
"""


def main(
        output_dir: str,
        input_dir: str,
        max_r_perp: units.Quantity
):
    if max_r_perp is not None:
        max_r_perp *= units.kpc
    else:
        max_r_perp = np.inf * units.kpc

    cosmo = cosmology.WMAP9

    output_dir = os.path.join(output_dir, "foreground")
    u.mkdir_check(output_dir)

    frb210912 = lib.fld
    gama = table.QTable.read(os.path.join(lib.repo_data, "GAMA_KN2Fm5.csv"))
    gama = gama[gama["NQ"] > 2]
    # gama = gama[gama["PROB"] > 0.]
    gama["RAcen"] *= units.deg
    gama["Deccen"] *= units.deg
    gama["COORD"] = coordinates.SkyCoord(gama["RAcen"], gama["Deccen"])
    # gama["in_R"] = R_img.wcs[0].footprint_contains(gama["COORD"])
    # gama["x_R"], gama["y_R"] = R_img.world_to_pixel(gama["COORD"])
    gama["SEPARATION"] = frb210912.frb.position.separation(gama["COORD"]).to(units.arcsec)
    gama["D_A"] = cosmo.angular_diameter_distance(gama["Z"])
    gama["R_PERP"] = (gama["SEPARATION"].to(units.rad).value * gama["D_A"]).to(units.kpc)

    # prospect_gama = table.QTable.read((os.path.join(input_dir, "gkvProSpectv02.csv")))
    # prospect_gama["StellarMass_bestfit"] *= units.solMass
    # prospect_gama["StellarMass_50"] *= units.solMass
    # prospect_gama["StellarMass_16"] *= units.solMass
    # prospect_gama["StellarMass_84"] *= units.solMass

    # gama_joined = table.join(gama, prospect_gama, keys="uberID", table_names=["GAMA", "PROSPECT"])

    subset = gama[gama["R_PERP"] < max_r_perp]

    galaxies = []
    for row in subset:
        galaxy = objects.Galaxy(
            name=str(row["uberID"]),
            field=frb210912,
            position=row["COORD"],
            z=row["Z"],
            mass_stellar=row["mstar"],
            mass_stellar_err=row["delmstar"]
            # mass_stellar_err_minus=row["StellarMass_50"] - row["StellarMass_16"],
            # mass_stellar_err_plus=row["StellarMass_84"] - row["StellarMass_50"]
        )
        galaxies.append(galaxy)
    subset["GalaxyObject"] = galaxies

    results_dict = frb210912.frb.foreground_accounting(
        foreground_objects=list(subset["GalaxyObject"])
    )
    halo_tbl = results_dict.pop("halo_table")
    halo_tbl.write(os.path.join(output_dir, f"GAMA_halo_table_{np.round(max_r_perp.value)}.ecsv"), overwrite=True)

    halo_dm_cum = results_dict.pop("halo_dm_cum")
    dm_cum_path = os.path.join(output_dir, "GAMA_halo_dm_cumulative_tables")
    u.mkdir_check(dm_cum_path)
    for obj_name in halo_dm_cum:
        halo_dm_cum[obj_name].write(os.path.join(dm_cum_path, f"{obj_name}_DM_cumulative.ecsv"), overwrite=True)

    dm_cum_all = results_dict.pop("dm_cum_table")
    dm_cum_all.write(os.path.join(output_dir, f"GAMA_DM_cumulative_all_{np.round(max_r_perp.value)}.ecsv"),
                     overwrite=True)

    results_dict.pop("halo_models")
    results_dict.pop("halo_dm_profiles")

    object_dir = os.path.join(output_dir, "objects")
    u.mkdir_check(object_dir)
    for galaxy in galaxies:
        galaxy.output_file = os.path.join(object_dir, f"{galaxy.name}_results.yaml")
        try:
            galaxy.update_output_file()
        except yaml.representer.RepresenterError:
            print(f"Output file for galaxy {galaxy.name} could not be written.")

    p.save_params(
        os.path.join(output_dir, "GAMA_results.yaml"),
        results_dict
    )


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
        "-d",
        help="Max R_perp to consider, in kpc.",
        type=float,
        default=None
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        max_r_perp=args.d
    )
