#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np

from astropy import table

import craftutils.utils as u
import craftutils.observation.field as field

import lib

description = """
Generates the table of PATH results.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    fields = field.list_fields()
    frb_fields = list(filter(lambda n: n.startswith("FRB"), fields))

    db_path = os.path.join(lib.dropbox_path, "tables")
    os.makedirs(db_path, exist_ok=True)

    path_dicts = []



    for fld_name in frb_fields:
        fld = field.FRBField.from_params(fld_name)
        fld.load_output_file()
        frb = fld.frb
        host = frb.host_galaxy
        if host is None:
            continue
        good_path_img = host.probabilistic_association_img
        frb.load_output_file()
        path_row = {
            "frb": fld_name.replace("FRB", "")
        }
        print("\t", good_path_img)
        if good_path_img is None:
            continue
        print(f"{good_path_img in fld.path_runs}")
        if good_path_img not in fld.path_runs:
            continue
        run_dicts = fld.path_runs[good_path_img]
        path_row["band"] = fld.imaging[good_path_img]["filter"]

        for p_u, run_dict in run_dicts.items():
            path_row[f"pox_{p_u}"] = np.round(run_dict["max_P(O|x_i)"], 4)
            path_row[f"pux_{p_u}"] = np.round(run_dict["P(U|x)"], 4)
            if p_u == "calculated":
                path_row["pu_calculated"] = np.round(run_dict["priors"]["U"], 4)
        if f"pox_0.1" not in path_row:
            continue
        if "pu_calculated" not in path_row:
            path_row["pu_calculated"] = -999.
            path_row["pox_calculated"] = -999.
            path_row["pux_calculated"] = -999.
        path_row["dm"] = frb.dm
        path_row["dm_err"] = frb.dm_err
        path_dicts.append(path_row)
        print(path_row)

    path = table.QTable(path_dicts)

    col_dict = {
        "frb": "FRB",
        "dm": "DM",
        # "z": "$z$",
        "band_nice": "Band",
        "pox_0.0": r"\POx\phantom{.}",
        "pux_0.0": r"\PUx\phantom{.}",
        "pox_0.1": r"\POx\phantom{,}",
        "pux_0.1": r"\PUx\phantom{,}",
        "pox_0.2": r"\POx\phantom{'}",
        "pux_0.2": r"\PUx\phantom{'}",
        "pu_calculated": r"\PU",
        "pox_calculated": r"\POx\phantom{`}",
        "pux_calculated": r"\PUx\phantom{`}",
    }

    path.write(os.path.join(lib.table_path, "path_table.ecsv"), overwrite=True)
    path.write(os.path.join(lib.table_path, "path_table.csv"), overwrite=True)

    bands = {
        "g_HIGH": "$g$",
        "R_SPECIAL": "$R$",
        "I_BESS": "$I$",
        "Ks": "$K_s$"
    }

    path["band_nice"] = [bands[b] for b in path["band"]]

    u.latexise_table(
        tbl=u.add_stats(
            path[
                "frb",
                "dm", "dm_err",
                "band_nice",
                "pox_0.0", # "pux_0.0",
                "pox_0.1", "pux_0.1",
                "pox_0.2", "pux_0.2",
                "pu_calculated", "pox_calculated", "pux_calculated"
            ],
            name_col="frb",
            cols_exclude=["frb", "band_nice"],
            round_n=4
        ),
        sub_colnames={"DM": "\dmunits{}"},
        column_dict=col_dict,
        output_path=os.path.join(lib.tex_path, "craft_photometry.tex"),
        short_caption="PATH outputs.",
        caption=r"PATH outputs with varying $P(U)$. "
                r"\tabscript{" + u.latex_sanitise(os.path.basename(__file__)) + "}",
        label="tab:path",
        longtable=True,
        landscape=True,
        coltypes="ccc|c|cc|cc|ccc",
        multicolumn=[(3, "c|", ""), (1, "c|", "$P(U)=0$"), (2, "c|", "$P(U)=0.1$"), (2, "c|", "$P(U)=0.2$"),
                     (3, "c", "$P(U)=P(U|\DM{})$")],
        second_path=os.path.join(db_path, "path.tex"),
    )

    # # PATH TABLE
    # # ============================
    # all_objects = lib.load_photometry_table()
    # hosts = all_objects[[n.startswith("HG") for n in all_objects["object_name"]]]
    # path = hosts["field_name", "path_img", "path_pox", "path_pu", "path_pux"]
    # u.latexise_table(
    #     tbl=u.add_stats(
    #         path,
    #         name_col="field_name",
    #         cols_exclude=["field_name"]
    #     ),
    #     column_dict=col_dict,
    #     output_path=os.path.join(lib.tex_path, "craft_photometry.tex"),
    #     short_caption="CRAFT host VLT photometry",
    #     caption=r"PATH outputs."
    #             r"\tabscript{" + u.latex_sanitise(os.path.basename(__file__)) + "}",
    #     label="tab:craft-photometry",
    #     longtable=True,
    #     landscape=True,
    #     coltypes="cccc|ccccc|ccc",
    #     multicolumn=[(4, "c|", ""), (5, "c|", "FORS2"), (3, "c", "HAWK-I")],
    #     second_path=os.path.join(db_path, "craft_photometry.tex"),
    # )


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
