#!/usr/bin/env python
# Code by Lachlan Marnoch, 2023

import lib

description = """
Runs PATH (Probabilistic Association of Transients; Aggarwal et al 2021) in various configurations on the imaging data 
covering FRB 20210912A.
"""


def main(
        output_dir: str,
        input_dir: str,
        skip_unsubtracted: bool,
        skip_exp: bool,
        skip_core: bool,
        skip_uniform: bool
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)
    # Set up field object
    fld = lib.fld
    frb210912 = fld.frb

    # Set up images
    lib.load_images()

    path_radius = 11

    # Run PATH on unsubtracted images.
    if not skip_unsubtracted:
        lib.multipath(
            imgs=[lib.R_img, lib.g_img, lib.I_img, lib.K_img],
            path_radius=path_radius,
            frb_object=frb210912,
            path_slug="unsubtracted",
        )

    frb210912.host_candidate_tables = {}

    # Run PATH on trimmed images.
    # With `exp` offset prior
    if not skip_exp:
        lib.multipath(
            imgs=[lib.R_img_trimmed, lib.g_img_trimmed, lib.I_img_trimmed, lib.K_img],
            path_radius=path_radius,
            frb_object=frb210912,
            path_slug="trimmed_exp",
            offset_prior="exp"
        )
    # With 'core' offset prior
    if not skip_core:
        lib.multipath(
            imgs=[lib.R_img_trimmed, lib.g_img_trimmed, lib.I_img_trimmed, lib.K_img],
            path_radius=path_radius,
            frb_object=frb210912,
            path_slug="trimmed_core",
            offset_prior="core"
        )
    # With `uniform` offset prior
    if not skip_uniform:
        lib.multipath(
            imgs=[lib.R_img_trimmed, lib.g_img_trimmed, lib.I_img_trimmed, lib.K_img],
            path_radius=path_radius,
            frb_object=frb210912,
            path_slug="trimmed_uniform",
            offset_prior="uniform"
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

    parser.add_argument(
        "--skip_unsubtracted",
        help="Skip PATH on unsubtracted images.",
        action="store_true"
    )

    parser.add_argument(
        "--skip_exp", "--skip_exponential",
        help="Skip PATH run with exponential host offset prior.",
        action="store_true"
    )

    parser.add_argument(
        "--skip_core",
        help="Skip PATH run with core host offset prior.",
        action="store_true"
    )

    parser.add_argument(
        "--skip_uniform",
        help="Skip PATH run with uniform host offset prior.",
        action="store_true"
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        skip_unsubtracted=args.skip_unsubtracted,
        skip_exp=args.skip_exp, #or args.skip_exponential,
        skip_core=args.skip_core,
        skip_uniform=args.skip_uniform
    )
