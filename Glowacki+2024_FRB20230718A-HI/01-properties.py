#!/usr/bin/env python
# Code by Lachlan Marnoch, 2023-2024

import os

from astropy import units
import numpy as np

import craftutils.utils as u
import craftutils.params as p
from craftutils.observation import objects, field, instrument, image
import craftutils.astrometry as astm

import lib

description = """
Crunches some properties of the FRB sightline.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    output_dict = {}

    fld = lib.fld
    frb = fld.frb

    dms = {}

    dms["cosmic"] = frb.dm_cosmic()
    print("<DM_cosmic>:\n\t", dms["cosmic"])
    dms["mw_ism_ymw16"] = frb.dm_mw_ism_ymw16()
    print("DM_MW,ISM,YMW16:\n\t", dms["mw_ism_ymw16"])
    dms["mw_ism_ne2001"] = frb.dm_mw_ism_ne2001()
    print("DM_MW,ISM,NE2001:\n\t", dms["mw_ism_ne2001"])
    dms["mw_ism_ne2001_baror"] = frb.dm_mw_ism_ne2001_baror()
    print("DM_MW,ISM,NE2001,Baror:\n\t", dms["mw_ism_ne2001_baror"])
    dms["mw_halo_pz19"] = frb.dm_mw_halo("pz19")
    print("DM_MW,halo,PZ19:\n\t", dms["mw_halo_pz19"])
    dms["mw_constraint"] = frb.dm - frb.dm_cosmic()
    print("DM_MW as constrained by <DM_cosmic>:\n\t", dms["mw_constraint"])
    dms["mw_ism_constraint"] = dms["mw_constraint"] - dms["mw_halo_pz19"]
    print("DM_MW,ISM as constrained by <DM_cosmic> and DM_MW,halo,PZ19:\n\t", dms["mw_ism_constraint"])
    output_dict["dm"] = dms
    print()

    output_dict["coord_galactic"] = frb.position.galactic.to_string("hmsdms")
    output_dict["coord_galactic_l"] = frb.position.galactic.l
    output_dict["coord_galactic_b"] = frb.position.galactic.b
    output_dict["extinction_fm07"] = {}
    print("FRB Galactic Coordinates:\n\t", output_dict["coord_galactic"])

    for band_name in "grz":
        band = lib.decam.filters[band_name]
        output_dict["extinction_fm07"][band_name] = frb.galactic_extinction(band.lambda_eff)[0]
        print(f"{band_name}-band Galactic extinction (FM07):\n\t", output_dict["extinction_fm07"][band_name])

    p.save_params(
        os.path.join(output_path, "frb_properties.yaml"),
        output_dict
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
