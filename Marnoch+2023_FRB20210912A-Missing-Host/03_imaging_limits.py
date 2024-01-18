#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import astropy.units as units

import craftutils.utils as u

import lib

description = """
Derives the imaging limits.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    fld = lib.fld
    frb210912 = fld.frb

    lib.load_images()
    image_dict = {
        "unsubtracted": {
            "g_HIGH": lib.g_img,
            "R_SPECIAL": lib.R_img,
            "I_BESS": lib.I_img,
            "Ks": lib.K_img
        },
        "trimmed": {
            "g_HIGH": lib.g_img_trimmed,
            "R_SPECIAL": lib.R_img_trimmed,
            "I_BESS": lib.I_img_trimmed,
            "Ks": lib.K_img
        },
        "WISE": {
            "W1": lib.W1_img,
            "W2": lib.W2_img,
            "W3": lib.W3_img,
            "W4": lib.W4_img
        }
    }

    output_dir = os.path.join(output_dir, "limits")
    u.mkdir_check_nested(output_dir, False)

    for dict_name in image_dict:
        d = image_dict[dict_name]
        limits = {}
        output_dir_this = os.path.join(output_dir, dict_name)
        u.mkdir_check(output_dir_this)
        print("\n=====================================================")
        print("5-sigma imaging limits,", dict_name)
        for fil in d:
            img = d[fil]
            psf = img.extract_header_item("PSF_FWHM")
            if psf is None:
                if fil == "W4":
                    psf = 16.5
                else:
                    psf = 8.25

            ap_radius = 2 * psf * units.arcsec
            if img.zeropoint_best is None:
                img.zeropoint()
            results = img.test_limit_location(
                coord=frb210912.position,
                ap_radius=ap_radius
            )
            limits[fil] = results
            results.write(os.path.join(output_dir_this, f"limits_{dict_name}_{fil}.ecsv"), overwrite=True)
            print(fil, ":", results[4]["mag"], )  # img.get_zeropoint(cat_name="best", img_name="self"))


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
