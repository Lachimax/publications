#!/usr/bin/env python
# Code by Lachlan Marnoch, 2024-2025

import os
import zipfile

import craftutils.utils as u
import craftutils.retrieve as r

import lib

description = """
Retrieve data for the following scripts.
"""


def main(
        output_dir: str,
        input_dir: str
):
    print(input_dir)
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    # Download main data zip
    print(input_dir)
    os.makedirs(input_dir, exist_ok=True)

    dl_path = input_dir
    print(dl_path)
    fname = "input.zip"
    file_path = os.path.join(dl_path, fname)
    print(file_path)
    u.rm_check(file_path)
    print("Downloading input data.")
    r.download_file(
        file_url="https://drive.usercontent.google.com/download?id=14tvBFN9-GKke0i2kO1VCNrnH8KMS6LQ_&export=download&authuser=0&confirm=t&uuid=df6d226b-c859-4d84-be84-9a2717c8e0c1&at=AMrWOn2oAI12Fwg-ZIix-IgXToqD%3A1791449722411",
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
