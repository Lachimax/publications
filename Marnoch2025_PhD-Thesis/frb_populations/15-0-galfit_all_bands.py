#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os
import shutil

from astropy import table

import craftutils.utils as u
import craftutils.observation.field as field

import lib

description = """
Collates all GALFIT results from all bandpasses.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    field_list = list(filter(lambda f: f.startswith("FRB"), field.list_fields()))

    output_path_galfit = os.path.join(output_dir, "galfit")
    shutil.rmtree(output_path_galfit)
    os.makedirs(output_path_galfit, exist_ok=True)

    fil_dicts = {}

    for field_name in field_list:
        print(field_name)
        fld = field.Field.from_params(field_name)
        frb = fld.frb
        host = frb.host_galaxy
        if host is not None:
            host.load_output_file()
            print("\t Checking for GALFIT model...")
            if host.galfit_models is not None:
                print("\t Found GALFIT models:")
                for key, model_dict in host.galfit_models.items():
                    print(f"\t\t{key}")
                    sersic = model_dict["COMP_2"].copy()
                    sersic["image"] = model_dict["image"]
                    sersic["field_name"] = field_name
                    sersic["object_name"] = host.name
                    if host.z is not None and host.z > -990.:
                        sersic["z"] = host.z
                    if key not in fil_dicts:
                        fil_dicts[key] = []
                    fil_dicts[key].append(sersic)
                    frame = sersic["frame"]
                    galfit_dir = os.path.join(
                        host.data_path, "GALFIT", os.path.basename(model_dict["image"]).replace(".fits", "")
                    )
                    fil_path = os.path.join(output_path_galfit, key)
                    os.makedirs(fil_path, exist_ok=True)
                    plot_path = os.path.join(galfit_dir, f"galfit_plot_frame{frame}.png")
                    plot_dest = os.path.join(fil_path, f"{field_name}_galfit_plot_frame{frame}.png")
                    if os.path.isfile(plot_path):
                        shutil.copyfile(plot_path, plot_dest)

    print()
    for key, ls in fil_dicts.items():
        fil_tbl = table.QTable(ls)
        print(f"For {key}: {len(fil_tbl)}")
        fil_tbl.write(os.path.join(output_path_galfit, f"galfit_table_{key}.csv"), overwrite=True)
        fil_tbl.write(os.path.join(output_path_galfit, f"galfit_table_{key}.ecsv"), overwrite=True)

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
