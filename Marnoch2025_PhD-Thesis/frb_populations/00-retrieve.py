#!/usr/bin/env python
# Code by Lachlan Marnoch, 2024-2025

import os
import zipfile

import craftutils.utils as u
import craftutils.retrieve as r

import lib

description = """
Currently does nothing; will retrieve data for the following scripts.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    # Download main data zip

    dl_path = os.path.dirname(input_dir)
    fname = "input.zip"
    file_path = os.path.join(dl_path, fname)
    u.rm_check(file_path)
    print("Downloading input data.")
    r.download_file(
        file_url="https://drive.usercontent.google.com/download?id=1CH9szgykuW_1V0GuBrIe4Exsi1jlck3W&export=download&authuser=0&confirm=t&uuid=1d15b77a-11e4-462e-8ea8-51c3f3b4d7c4&at=AN8xHorlhsMgO5hNsAV38xGZqFPD:1758250784068",
        output_dir=dl_path,
        overwrite=True,
        filename=fname
    )

    with zipfile.ZipFile(
            file_path,
            'r'
    ) as zip_ref:
        zip_ref.extractall(dl_path)

    os.remove(file_path)

    # Download FRBHostData copy.

    fname = "FRBHostData.zip"
    dl_path = os.path.dirname(os.path.join(input_dir, "frb_populations"))
    file_path = os.path.join(dl_path, fname)
    u.rm_check(file_path)
    print("Downloading FRBHostData.")
    r.download_file(
        file_url="https://drive.usercontent.google.com/download?id=1k13wgOA-qM_PN6FAPkdZf7C-rQ_SwI88&export=download&authuser=0&confirm=t&uuid=e387ed21-45b0-4692-ade3-9edd090db00b&at=AN8xHorEPmXwGmSpWOLBzFUsVaUg:1757041448079",
        output_dir=dl_path,
        overwrite=True,
        filename=fname
    )

    with zipfile.ZipFile(
            file_path,
            'r'
    ) as zip_ref:
        zip_ref.extractall(dl_path)

    os.remove(file_path)
        

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
