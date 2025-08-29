#!/usr/bin/env python
# Code by Lachlan Marnoch, 2024-2025

import os

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

    # Download ACS-GC fits files
    
    directory = {
        # "ACS-GC": {
        #     "url": "https://content.cld.iop.org/journals/0067-0049/200/1/9/revision1/apjs426137_ACS-GC_published_catalogs.tar.gz",
        #     "files": [
        #         "cosmos_i_public_catalog_V2.0.fits.gz",
        #         "egs_v_i_public_catalog_V2.0.fits.gz",
        #         "gems_v_z_public_catalog_V2.0.fits.gz",
        #         "goods_v_i_public_catalog_V2.0.fits.gz"
        #     ]
        # },
        "chrimes+2021": {
            "url": "https://raw.githubusercontent.com/achrimes2/MW-NS-Flight/refs/heads/master/Data/",
            "files": [
                "data_hmxb.txt",
                "data_lmxb.txt",
                "data_magnetars.txt",
                "data_pulsars.txt"
            ]
        },
        
    }

    for key, props in directory.items():
        download_dir = os.path.join(lib.output_path, key)
        os.makedirs(download_dir)
        for file in props["files"]:
            r.download_file(
            file_url=props["url"] + file,
            output_dir=download_dir,
            overwrite=True,
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

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path
    )
