#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import craftutils.utils as u
import craftutils.params as p
from craftutils.observation import field

import lib

description = """
Uses scripts from James et al 2021 to generate p(z|DM) distributions for CRAFT FRBs.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    script_lines = [
        "#!/bin/bash\n",
        "\n"
    ]
    fields = field.list_fields()
    for field_name in fields:
        print(field_name)
        if field_name.startswith("FRB"):
            fld = field.Field.from_params(field_name)
            if fld.frb.survey == "CRAFT/ICS":
                command = lib.generate_zdm_command(fld.frb)
                print(command)
                if command:
                    script_lines.append(command)

    zdm_path = os.path.join(p.data_dir, "zdm")
    u.mkdir_check()
    script_path = os.path.join(zdm_path, "zdm_script.sh")
    with open(script_path, "w") as file:
        file.writelines(script_lines)


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
