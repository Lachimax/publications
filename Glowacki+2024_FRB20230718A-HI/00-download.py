#!/usr/bin/env python
# Code by Lachlan Marnoch, 2023-2024

import os
import requests

import pyvo as vo

import craftutils.utils as u
from craftutils.retrieve import download_file

import lib

description = """
Retrieves imaging data.
"""


def main(
        output_dir: str,
        input_dir: str,
        overwrite: bool
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb = lib.fld.frb
    ra = frb.position.ra.value
    dec = frb.position.dec.value

    # Retrieve DECaPS catalogue for this field, in a 2-arcmin square.
    cat_path = lib.cat_path()
    if overwrite or not os.path.exists(cat_path):
        print("Retrieving DECaPS catalogue.")
        # Set up a query.
        service = vo.dal.TAPService("https://datalab.noirlab.edu/tap")

        query = f"""
        SELECT *
        FROM decaps_dr2.object
        WHERE ra < {ra + 1 / 60}
        AND ra > {ra - 1 / 60}
        AND dec < {dec + 1 / 60}
        AND dec > {dec - 1 / 60}
        """

        result = service.search(query)
        decaps_cat = result.table
        decaps_cat.write(
            cat_path,
            overwrite=True
        )

    # Retrieve cutout fits from service.

    for band in "grz":
        if overwrite or not os.path.exists(lib.cutout_path(band)):
            print(f"Retrieving {band}-band DECaPS cutout.")
            url_cutout = f"http://legacysurvey.org/viewer/fits-cutout/?layer=decaps2&ra={ra}&dec={dec}&pixscale=0.262&bands={band}"
            print(url_cutout)
            response = download_file(
                file_url=url_cutout,
                output_dir=lib.input_path,
                filename=lib.cutout_name.format(band)
            )
            print(response)


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
        "--overwrite",
        help="Overwrite existing files?",
        action="store_true"
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        overwrite=args.overwrite
    )
