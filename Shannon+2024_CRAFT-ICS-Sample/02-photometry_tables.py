#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX
import shutil

import os

import astropy.table as table

import craftutils.utils as u
import craftutils.params as p

import lib

description = "Generates the table of photometry seen in Shannon et al 2024, **Table X**."


def main(
        output_dir: str,
):
    # Load the object properties table.

    ics = p.load_params(os.path.join(os.path.join(output_dir, "fields.yaml")))["ics"]
    print(ics)
    script_dir = os.path.dirname(__file__)
    host_data_path = os.path.join(script_dir, "table_optical.ecsv")

    if isinstance(p.config["table_dir"], str) and os.path.isfile(os.path.join(p.config["table_dir"], "master_table_optical.ecsv")):
        host_data = table.QTable.read(os.path.join(p.config["table_dir"], "master_table_optical.ecsv"))
        remove = []
        for i, row in enumerate(host_data):
            if row["field_name"] not in ics:
                remove.append(i)
        host_data.remove_rows(remove)
        host_data.write(host_data_path, overwrite=True)
        host_data.write(host_data_path.replace(".ecsv", ".csv"), overwrite=True)
    else:
        host_data = table.QTable.read(host_data_path)
    # These are the table columns we care about for the purposes of this table.
    hosts_ics_all = host_data[
        "object_name",
        "field_name",
        'mag_best_vlt-fors2_u-HIGH', 'mag_best_vlt-fors2_u-HIGH_err',
        "mag_best_vlt-fors2_g-HIGH", 'mag_best_vlt-fors2_g-HIGH_err',
        'mag_best_vlt-fors2_R-SPECIAL', 'mag_best_vlt-fors2_R-SPECIAL_err',
        'mag_best_vlt-fors2_I-BESS', 'mag_best_vlt-fors2_I-BESS_err',
        'mag_best_vlt-fors2_z-GUNN', 'mag_best_vlt-fors2_z-GUNN_err',
        'mag_best_vlt-hawki_J', 'mag_best_vlt-hawki_J_err',
        'mag_best_vlt-hawki_H', 'mag_best_vlt-hawki_H_err',
        'mag_best_vlt-hawki_Ks', 'mag_best_vlt-hawki_Ks_err'
    ]
    # Limit the table to FRB host galaxies, which in this table all start with 'HG'.
    hosts = list(map(lambda r: r["object_name"].startswith("HG"), hosts_ics_all))
    hosts_ics = hosts_ics_all[hosts]
    # Make the host names a bit Latex-fancy (insert '\,' between HG and the number)
    hosts_ics["field_name"] = list(map(lambda f: f"{f[:3]}\,{f[3:]}", hosts_ics["field_name"]))
    hosts_ics.remove_column("object_name")

    # Specify the number of acceptable significant figures in the uncertainties.
    n_digits_err = 1
    # Cycle through each filter name to collect magnitudes
    for band in ("u-HIGH", "g-HIGH", "R-SPECIAL", "I-BESS", "z-GUNN", "J", "H", "Ks"):
        if band in ("J", "H", "Ks"):
            instrument = "vlt-hawki"
        else:
            instrument = "vlt-fors2"

        mag_str = []
        for row in hosts_ics:
            mag = row[f"mag_best_{instrument}_{band}"]
            mag_err = row[f"mag_best_{instrument}_{band}_err"]
            # Put the value and uncertainty in a nice format, as a string.
            this_str, value, uncertainty = u.uncertainty_string(
                value=mag,
                uncertainty=mag_err,
                n_digits_err=n_digits_err,
                n_digits_lim=3,
                limit_type="lower"
            )
            mag_str.append(this_str)
        # Generate  the column and remove the old ones.
        hosts_ics[f"mag_{band}"] = mag_str
        hosts_ics.remove_column(f"mag_best_{instrument}_{band}")
        hosts_ics.remove_column(f"mag_best_{instrument}_{band}_err")

    u.mkdir_check_nested(output_dir, remove_last=False)
    tbl_path = os.path.join(output_dir, "tbl.tex")
    # We spit this out as a latex table, ready to go.
    print("Writing table...")
    hosts_ics.write(tbl_path, format="ascii.latex", overwrite=True)



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
    )
