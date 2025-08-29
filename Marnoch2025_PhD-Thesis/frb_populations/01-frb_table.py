#!/usr/bin/env python
# Code by Lachlan Marnoch, 2023

import os

import numpy as np

import astropy.table as table
import astropy.units as units
from astropy.coordinates import SkyCoord

import craftutils.utils as u
from craftutils.observation import objects

import lib

description = """
Downloads a CSV version of a Google Sheet I've been maintaining, which tabulates all of the FRB hosts in the literature 
that I'm aware of (as of September 2024), as well as unpublished CRAFT hosts. It then adds some derived values and re-saves it in a more 
machine-readable format. This takes quite a while; it will also not work .
"""


def main(
        output_dir: str,
        input_dir: str,
        field: str = None,
        quick: bool = False,
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)
    if field is None:
        flds = None
    else:
        flds = [field]

    lib.generate_table(flds=flds, quick=quick)


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
        "-f",
        help="Specific field (skips the rest). Otherwise updates all.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--quick",
        help="Skips the more computationally-expensive calculations and does not write the table to disk. Use for just "
             "updating the param yamls.",
        action="store_true"
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        field=args.f,
        quick=args.quick
    )
