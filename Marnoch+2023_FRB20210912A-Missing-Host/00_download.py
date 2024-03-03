#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os
import zipfile

import craftutils.utils as u
import craftutils.retrieve as r

import lib

description = """
Downloads and extracts the necessary input data.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)
    dl_path = os.path.expanduser(os.path.join(input_dir, ".."))
    file_path = os.path.join(dl_path, "input.zip")
    u.rm_check(file_path)
    print("Downloading input data.")
    r.download_file(
        file_url="https://drive.usercontent.google.com/download?id=1RIkATMiKgyT1U7OR6DOQ5JLffwrJXCcs&export=download&authuser=0&confirm=t&uuid=35bb6cb8-37ec-4038-93be-d8eb43ae2d63&at=APZUnTXhybFPE9i6UwQW95VzSFHj:1709205577049",
        output_dir=dl_path,
        overwrite=True,
        filename="input.zip"
    )

    with zipfile.ZipFile(
            file_path,
            'r'
    ) as zip_ref:
        zip_ref.extractall(dl_path)


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
