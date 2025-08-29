#!/usr/bin/env python
# Code by Lachlan Marnoch, 2025

import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.table as table
import astropy.units as units

import craftutils.utils as u

import lib

description = """
Collates the results of MC modelling.
"""


def main(
        output_dir: str,
        input_dir: str,
        n_real: float
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb190714 = lib.fld

    bin_colour = "violet"

    comm_dict = {}
    tbl_dict = {}

    props_main = lib.read_master_properties()

    rmax=1.

    for rel in "M13", "K18":

        print()
        print("=" * 50)
        print(rel)

        halo_tbls_all = []
        properties_all = []
        halo_props_individual = {}

        keys_get = [
            "dm_halo", "log_mass_halo", "log_mass_stellar",
            "offset_angle", "r_perp", "r_200", "r_200_angle",
            "path_length",
        ]

        for n in range(n_real):
            # np.random.seed(1994 + n)
            print("=" * 50)
            print("Iteration", n + 1, "of", n_real)

            halo_tbl = lib.read_halo_table_mc(relationship=rel, n=n)
            properties = lib.read_properties_mc(relationship=rel, n=n)

            halo_tbls_all.append(halo_tbl)

            for row in halo_tbl:
                gal_id = row["id"]
                if gal_id not in halo_props_individual:
                    halo_props_individual[gal_id] = {}
                    halo_props_individual[gal_id]["id"] = gal_id
                    for key in keys_get:
                        halo_props_individual[gal_id][key] = []
                for key in keys_get:
                    if key not in row and key == "path_length":
                        val = 2 * np.sqrt(rmax * row["r_200"]**2 - row["r_perp"]**2)
                        if np.isnan(val):
                            val = 0 * units.kpc
                        if gal_id == "HG20190714A":
                            val /= 2
                    else:
                        val = row[key]
                    halo_props_individual[gal_id][key].append(val)

            properties["run"] = n
            properties_all.append(properties)

        prop_tbl = table.QTable(properties_all)
        prop_tbl.remove_column("halo_dm_profiles")

        print("=" * 50)

        prop_list = []
        for obj_name, prop_dict in halo_props_individual.items():
            print(obj_name, ":")
            new_dict = {"name": obj_name}
            prop_dict.pop("id")
            prop_tbl_ind = table.QTable(prop_dict)
            lib.write_halo_individual_table_collated_mc(
                tbl=prop_tbl_ind,
                obj_id=obj_name,
                relationship=rel,
            )
            for key_2, value in prop_dict.items():
                if isinstance(value[0], str):
                    continue
                q = units.Quantity(prop_dict[key_2])
                mu = np.nanmean(q)
                med = np.nanmedian(q)
                sigma = np.nanstd(q)
                d = 68.27 / 2
                perc_68_up = np.percentile(q, q=50+d)
                perc_68_down = np.percentile(q, q=50-d)
                if perc_68_down == 0 and sigma < mu:
                    perc_68_down = mu - sigma
                if perc_68_up == 0:
                    perc_68_up = mu + sigma
                new_dict[key_2] = mu
                new_dict[key_2 + "_med"] = med
                new_dict[key_2 + "_err"] = sigma
                new_dict[key_2 + "_upper"] = perc_68_up
                new_dict[key_2 + "_lower"] = perc_68_down
                new_dict[key_2 + "_med_err_minus"] = med - perc_68_down
                new_dict[key_2 + "_med_err_plus"] = perc_68_up - med
                new_dict[key_2 + "_mean_err_minus"] = mu - perc_68_down
                new_dict[key_2 + "_mean_err_plus"] = perc_68_up - mu
                print(
                    "\t", key_2, ":", mu, "+/-", sigma,
                    "(+", new_dict[key_2 + "_med_err_plus"], ", -", new_dict[key_2 + "_med_err_minus"],
                    ") (+", new_dict[key_2 + "_mean_err_plus"], "-", new_dict[key_2 + "_mean_err_minus"]
                )

            prop_list.append(new_dict)

        halo_tbl_mc = table.QTable(prop_list)
        halo_tbl_mc.sort("offset_angle")
        lib.write_halo_table_mc_collated(halo_tbl_mc, relationship=rel)
        lib.write_properties_table_mc(tbl=prop_tbl, relationship=rel)


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
        "-n",
        help="Number of instantiations to read.",
        type=int,
        default=10000
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        n_real=args.n
    )
