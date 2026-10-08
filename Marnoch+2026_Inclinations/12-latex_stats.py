#!/usr/bin/env python
# Code by Lachlan Marnoch, 2023

import os
import numbers
import shutil

import numpy as np

import astropy.table as table
import astropy.units as units
from astropy.coordinates import SkyCoord

import craftutils.utils as u
from craftutils.observation import objects

import lib

description = """
Uses the derived FRB host table to generate some latex tables.
"""

this_script = u.latex_sanitise(os.path.basename(__file__))


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    if isinstance(lib.dropbox_path, str):
        db_path = os.path.join(lib.dropbox_path, "tables")
    else:
        db_path = None

    all_frb_table = lib.load_frb_table().copy()
    all_frb_table.remove_column("frb_object")
    all_frb_table["ra"] = all_frb_table["coord"].ra
    all_frb_table["dec"] = all_frb_table["coord"].dec

    craft_table = all_frb_table.copy()[all_frb_table["team"] == "CRAFT"]

    # make_tex(all_frb_table, "frb_hosts_general")

    col_replace = {
        "name": "FRB",
        "survey": "CRAFT",
        "telescope": "Detection",
        "telescope_localisation": "Localisation",
        "team": "Survey /",
        "vlt_imaged": "VLT",
        "host_identified": "Host",
        "ra": r"$\alpha_\mathrm{FRB}$",
        "dec": r"$\delta_\mathrm{FRB}$",
        "a": "$\sigma_a$",  # "$a_\mathrm{FRB}$",
        "b": "$\sigma_b$",  # "$b_\mathrm{FRB}$",
        "theta": r"$\theta_\mathrm{FRB}$",
        "ref_position": r"Ref.",
        "dm": r"$\mathrm{DM_{FRB}}$",
        "ref_dm": r"Ref.\phantom{d}",
        "z": "$\zhost$",
        "ref_z": r"Ref.\phantom{z}"
    }
    subcols = {
        "survey": "Survey",
        "team": "Team",
        "telescope": "Telescope",
        "vlt_imaged": "imaging?",
        "ra": "(J2000)",
        "dec": "(J2000)",
        "telescope_localisation": "Telescope",
        "dm": r"($\dmunits$)",
        "host_identified": "identified?",
        "a": r"($\arcsec$)",
        "b": r"($\arcsec$)",
        "theta": "($^\circ$ E of N)"
    }

    craft_table.write(os.path.join(lib.table_path, "frbs_craft.csv"), overwrite=True)
    craft_table.write(os.path.join(lib.table_path, "frbs_craft.ecsv"), overwrite=True)

    craft_table_tex = craft_table.copy()[
        "name",
        "ra", "ra_err",
        "dec", "dec_err",
        "a", "b", "theta",
        "ref_position",
        "dm", "dm_err", "ref_dm",
        "vlt_imaged",
        "host_identified",
        "z", "z_err", "ref_z",
        "survey"
    ]

    a = []
    b = []
    theta = []
    for row in craft_table_tex:
        if row["a"] > 0 and row["b"] > 0:
            a_ = row["a"]
            b_ = row["b"]
            theta_ = row["theta"]
        else:
            ra_err = row["ra_err"]
            dec_err = row["dec_err"]
            if ra_err > dec_err:
                a_ = ra_err
                b_ = dec_err
                theta_ = 90. * units.deg
            else:
                a_ = dec_err
                b_ = ra_err
                theta_ = 0. * units.deg

        a.append(str(a_.value.round(3)))
        b.append(str(b_.value.round(3)))
        theta.append(str(theta_.value.round(1)))

    craft_table_tex["a"] = a
    craft_table_tex["b"] = b
    craft_table_tex["theta"] = theta

    for row in craft_table_tex:
        row["survey"] = str(row["survey"]).replace("CRAFT/", "")

    stats_lines = []

    ics_table = craft_table[craft_table["survey"] == "CRAFT/ICS"]
    craco_table = craft_table[craft_table["survey"] == "CRAFT/CRACO"]
    dsa_table = all_frb_table[all_frb_table["team"] == "DSA"]
    chime_table = all_frb_table[all_frb_table["team"] == "CHIME/FRB"]
    meertrap_table = all_frb_table[all_frb_table["team"] == "MeerTRAP"]

    print()
    print("=" * 100)

    print("Some statistics:")
    print()

    n_craft_loc = len(craft_table)
    print("CRAFT localisations:\t", n_craft_loc)
    stats_lines.append(r"\newcommand{\nCRAFTloc}{" + str(n_craft_loc) + "}\n")
    n_ics_loc = len(ics_table)
    print("ICS localisations:\t", n_ics_loc)
    stats_lines.append(r"\newcommand{\nICSloc}{" + str(n_ics_loc) + "}\n")
    n_craco_loc = len(craco_table)
    print("CRACO localisations:\t", n_craco_loc)
    stats_lines.append(r"\newcommand{\nCRACOloc}{" + str(n_craco_loc) + "}\n")
    craft_table_z = craft_table[craft_table["z"] > -990]
    n_craft_z = len(craft_table_z)
    print("CRAFT redshifts:\t", n_craft_z)
    stats_lines.append(r"\newcommand{\nCRAFTz}{" + str(n_craft_z) + "}\n")
    print()

    surveys = {
        "CRAFT": craft_table,
        "ICS": ics_table,
        "CRACO": craco_table,
        "DSA": dsa_table,
        "CHIME": chime_table,
        "MeerTRAP": meertrap_table,
        "All": all_frb_table
    }

    for survey, tbl in surveys.items():
        n_hosts = np.sum(tbl["host_identified"])
        print(f"{survey} hosts:\t\t", n_hosts)
        stats_lines.append(r"\newcommand{\n" + survey + "hosts}{" + str(n_hosts) + "}\n")

        tbl_z = tbl[tbl["z"] > -990]

        median_z = np.median(tbl_z["z"])
        print(f"{survey} median z:\t\t", median_z)
        stats_lines.append(r"\newcommand{\z" + survey + "median}{" + str(np.round(median_z, 6)) + "}\n")

        mean_z = tbl_z["z"].mean().round(5)
        print(f"{survey} mean z:\t\t", mean_z)
        stats_lines.append(r"\newcommand{\z" + survey + "mean}{" + str(mean_z) + "}\n")

        min_z = tbl_z["z"].min().round(5)
        print(f"{survey} min z:\t\t", min_z)
        stats_lines.append(r"\newcommand{\z" + survey + "min}{" + str(min_z) + "}\n")

        max_z = tbl_z["z"].max().round(5)
        print(f"{survey} max z:\t\t", max_z)
        stats_lines.append(r"\newcommand{\z" + survey + "max}{" + str(max_z) + "}\n")

        hosts = tbl[tbl["host_identified"]]

        median_dm = np.median(hosts["dm"]).round(2)
        print(f"{survey} host median DM:\t", median_dm)
        stats_lines.append(r"\newcommand{\DM" + survey + "median}{" + str(median_dm.value) + "}\n")

        mean_dm = np.mean(hosts["dm"]).round(2)
        print(f"{survey} host mean DM:\t", mean_dm)
        stats_lines.append(r"\newcommand{\DM" + survey + "mean}{" + str(mean_dm.value) + "}\n")

        min_dm = np.min(hosts["dm"]).round(2)
        print(f"{survey} host min DM:\t", min_dm)
        stats_lines.append(r"\newcommand{\DM" + survey + "min}{" + str(min_dm.value) + "}\n")

        max_dm = np.max(hosts["dm"]).round(2)
        print(f"{survey} host max DM:\t", max_dm)
        stats_lines.append(r"\newcommand{\DM" + survey + "max}{" + str(max_dm.value) + "}\n")

        stats_lines.append("\n")
        print()

    print()

    vlt_phot = lib.load_photometry_table()
    vlt_phot = vlt_phot[[n.startswith("HG") for n in vlt_phot["object_name"]]]
    n_vlt = len(vlt_phot)

    stats_lines.append("\n")
    print("Number of CRAFT hosts with VLT imaging:\t", n_vlt)
    stats_lines.append(r"\newcommand{\nVLTphot}{" + str(n_vlt) + "}\n")


    n_hosts_published = np.sum(all_frb_table["host_published"])
    n_craft_published = np.sum(craft_table["host_published"])
    n_craft_unpublished = np.sum(craft_table["host_identified"]) - np.sum(craft_table["host_published"])
    print("Number of total published FRB hosts:\t", n_hosts_published)
    stats_lines.append(u.latex_command("nHostsPublished", n_hosts_published))
    print("Number of published CRAFT FRB hosts:\t", n_craft_published)
    stats_lines.append(u.latex_command("nCRAFTHostsPublished", n_craft_published))
    print("Number of unpublished CRAFT FRB hosts:\t", n_craft_unpublished)
    stats_lines.append(u.latex_command("nCRAFTHostsUnpublished", n_craft_unpublished))


    for mag_col in list(filter(lambda c: c.startswith("mag_best_vlt-") and not c.endswith("err"), vlt_phot.colnames)):
        stats_lines.append("\n")
        fil = mag_col[9:]
        phot_fil = vlt_phot[vlt_phot[mag_col].value > -990]
        n_fil = len(phot_fil)
        print("\n")
        print(f"Number of CRAFT hosts with {fil} photometry:", n_fil)
        fil_com = fil.replace("_", "").replace("-", "").replace("2", "")
        stats_lines.append(
            r"\newcommand{\n" + fil_com + r"}{" + str(n_fil) + "}\n")

        median_fil = np.median(phot_fil[mag_col]).round(1)
        print(f"Median magnitude in {fil}:", median_fil)
        stats_lines.append(
            r"\newcommand{\median" + fil_com + r"}{" + str(median_fil.value) + "}\n")

        min_fil = np.min(phot_fil[mag_col]).round(1)
        print(f"Min magnitude in {fil}:", min_fil)
        stats_lines.append(
            r"\newcommand{\min" + fil_com + r"}{" + str(min_fil.value) + "}\n")

        max_fil = np.max(phot_fil[mag_col]).round(1)
        print(f"Max magnitude in {fil}:", max_fil)
        stats_lines.append(
            r"\newcommand{\max" + fil_com + r"}{" + str(max_fil.value) + "}\n")

    commands_file = os.path.join(lib.tex_path, "commands_frbpop_generated.tex")
    with open(commands_file, "w") as f:
        f.writelines(stats_lines)
    if isinstance(lib.dropbox_path, str):
        db_path = os.path.join(lib.dropbox_path, "commands")
        if os.path.isdir(lib.dropbox_path):
            shutil.copy(commands_file, db_path)


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
