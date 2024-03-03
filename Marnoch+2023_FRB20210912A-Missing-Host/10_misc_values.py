#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import craftutils.utils as u
import craftutils.params as p

import lib

description = """
Derives some miscellaneous values quoted in the paper, including DM_MW,ISM from YMW16 (Yao et al 2016)
and NE2001 (Cordes & Lazio 2002).
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_output_path(output_dir)
    values = {}
    frb = lib.fld.frb
    frb.retrieve_extinction_table(True)
    print("Galactic E(B-V):", frb.ebv_sandf)
    values["e_b-v"] = frb.ebv_sandf
    print("DM_ISM,MW:")
    dm_ne2001 = frb.dm_mw_ism_ne2001()
    print("\tNE2001:", dm_ne2001)
    values["dm_ism_mw_ne2001"] = dm_ne2001
    dm_ymw = frb.dm_mw_ism_ymw16()
    print("\tYMW16:", dm_ymw)
    values["dm_ism_mw_ymw16"] = dm_ymw
    p.save_params(os.path.join(lib.output_path, "frb_properties.yaml"), values)
    # print(values)

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
