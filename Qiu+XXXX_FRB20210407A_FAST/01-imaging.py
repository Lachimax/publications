#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import craftutils.utils as u

import lib

description = """
Does some processing of the DEIMOS and MMT imaging.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    fld = lib.fld
    frb = lib.frb210407

    img_z, img_i = lib.load_images()

    epoch_z = fld.epoch_from_params("FRB20210407A_DEIMOS_1", "keck-deimos")
    epoch_z.load_output_file()
    epoch_z.add_coadded_image(img_z, "z")

    epoch_z.proc_correct_astrometry_coadded(
        output_dir=os.path.join(lib.output_path, "astrometry"),
        image_type="coadded"
    )

    epoch_z.proc_source_extraction(
        output_dir=os.path.join(lib.output_path, "source_extraction"),
        image_type="astrometry"
    )

    img_z_astm = epoch_z.coadded_astrometry["z"]

    img_z_astm.load_output_file()
    img_z_astm.instrument = lib.deimos
    img_z_astm.filter = lib.deimos.filters["z"]
    # Zeropoints from A. Gordon via Slack
    img_z_astm.add_zeropoint(
        catalogue="panstarrs1",
        zeropoint=27.8954,
        zeropoint_err=0.0057,
        extinction=0.,
        extinction_err=0,
        airmass=0.,
        airmass_err=0.
    )
    img_z_astm.select_zeropoint(True)
    img_z_astm.update_output_file()

    epoch_i = fld.epoch_from_params("FRB20210407A_MMT_1", "sdss")
    epoch_i.load_output_file()
    epoch_i.add_coadded_image(img_i, "i")

    epoch_i.proc_correct_astrometry_coadded(
        output_dir=os.path.join(lib.output_path, "astrometry"),
        image_type="coadded"
    )

    epoch_i.proc_source_extraction(
        output_dir=os.path.join(lib.output_path, "source_extraction"),
        image_type="astrometry"
    )

    img_i_astm = epoch_i.coadded_astrometry["i"]

    img_i_astm.load_output_file()
    img_i_astm.instrument = lib.deimos
    img_i_astm.filter = lib.deimos.filters["i"]
    img_i_astm.add_zeropoint(
        catalogue="panstarrs1",
        zeropoint=26.679,
        zeropoint_err=0.0099,
        extinction=0.,
        extinction_err=0,
        airmass=0.,
        airmass_err=0.
    )
    img_i_astm.select_zeropoint(True)
    img_i_astm.update_output_file()
    
    print(img_z_astm.path)
    print(img_i_astm.path)

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
