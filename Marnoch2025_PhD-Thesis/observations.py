#!/usr/bin/env python
# Code by Lachlan Marnoch, 2025

import os

from astropy import table, units

import craftutils.utils as u

import lib

description = """
Generates a Latex table of CRAFT VLT observations.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    tbl = table.QTable.read(os.path.join(input_dir, "master_imaging_table.ecsv"))

    tbl = tbl["field_name", "epoch_name", "filter_name", "date_utc", "program_id", "depth", "psf_fwhm"]

    round_cols = ["depth", "psf_fwhm"]

    bands = {
        "u_HIGH": r"\uhigh",
        "g_HIGH": r"\ghigh",
        "I_BESS": r"\Ibess",
        "R_SPECIAL": r"\Rspec",
        "z_GUNN": r"\zgunn",
        "Ks": r"\Ks",
        "H": "$H$",
        "J": "$J$"
    }

    for key in ("FORS2", "HAWKI"):
        tbl_n = tbl[[key in n for n in tbl["epoch_name"]]]


        tbl_n.write(os.path.join(output_dir, f"{key}.csv"), overwrite=True)

        keep = ["combined" not in n for n in tbl_n["epoch_name"]]
        tbl_n = tbl_n[keep]


        tbl_n["Filter"] = [bands[b] for b in tbl_n["filter_name"]]

        tbl_n = tbl_n["field_name", "program_id",  "Filter",  "date_utc", "psf_fwhm", "depth"]

        u.latexise_table(
            tbl_n,
            output_path=os.path.join(output_dir, f"{key}.tex"),
            round_digits=1,
            round_cols=round_cols
        )

    # fors2_tbl.write(os.path.join(output_dir, "fors2.tex"), format="ascii.latex", overwrite=True)
    # hawki_tbl.write(os.path.join(output_dir, "hawki.tex"), format="ascii.latex", overwrite=True)




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
