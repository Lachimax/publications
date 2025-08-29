#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os
import shutil

import astropy.units as units
import astropy.table as table
import astropy.io.fits as fits
from astropy.cosmology import Planck18

import numpy as np
import matplotlib.pyplot as plt

import craftutils.utils as u
import craftutils.plotting as pl

import lib

description = """
Does statistical tests of CRAFT GALFIT FRB host properties against other object populations.
"""

latex_commands = []


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    fontsize_legend = 12
    lw_main = 4
    figsize = (pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] / 3.)
    legend_x = 0
    legend_y = 1.05

    craft_galfit = table.QTable.read(lib.craft_galfit_path())
    craft_galfit_z = craft_galfit[craft_galfit["z"] > 0]

    lib.hist_prop(
        "z",
        craft_galfit_z
    )
    lib.hist_prop(
        "dm_excess_rest",
        craft_galfit_z
    )
    lib.hist_prop(
        "galfit_offset",
        craft_galfit
    )
    lib.hist_prop(
        "galfit_offset",
        craft_galfit[craft_galfit["galfit_offset"] < 8 * units.arcsec],
        filename="hist_galfit_offset_clipped"
    )
    lib.hist_prop(
        "galfit_offset_deproj",
        craft_galfit_z[craft_galfit_z["galfit_offset_deproj"] < 100 * units.arcsec]
    )
    lib.hist_prop(
        "galfit_offset_proj",
        craft_galfit_z
    )
    lib.hist_prop(
        "galfit_offset_disk",
        craft_galfit_z[craft_galfit_z["galfit_offset_deproj"] < 100 * units.arcsec]
    )
    lib.hist_prop(
        "galfit_offset_norm",
        craft_galfit
    )
    lib.hist_prop(
        "galfit_offset_norm_deproj",
        craft_galfit[craft_galfit["galfit_offset_deproj"] < 100 * units.arcsec]
    )

    # Load tables

    tables = []

    craft = {
        "table": craft_galfit,
        "c": "purple",
        "ls": "solid",
        "label": "CRAFT FRBs",
        "object_name": "object_name",
    }
    craft_z = craft.copy()
    craft_z["table"] = craft_galfit_z

    # Type Ia SNe: Uddin+2020

    iasne_color = "gold"
    iasne_a1 = table.QTable.read(os.path.join(input_dir, "uddin+2020", "table_A1.txt"), format="ascii")
    iasne_c1 = table.QTable.read(os.path.join(input_dir, "uddin+2020", "table_C1.txt"), format="ascii")
    iasne_tbl = table.join(iasne_a1, iasne_c1, keys="Name")
    iasne_tbl["distance_proj"] *= units.kpc

    iasne = {
        "table": iasne_tbl,
        "c": iasne_color,
        "ls": "dotted",
        "label": "SNe Ia", # (Uddin et al., 2020)",
        "object_name": "Name",
        "z": "z_CMB",
        "galfit_offset_proj": "distance_proj"
    }

    tables.append(iasne)

    ccsne_color = "red"  # d633ff"

    # CCSNe: PTF
    # For projected offsets
    ccsne_ptf_tbl = table.QTable(fits.open(os.path.join(input_dir, "PTF/catalog/PTF_CCSN.fits"))[1].data)
    ccsne_ptf_tbl = ccsne_ptf_tbl[ccsne_ptf_tbl["OFFSET"] > 0.]
    ccsne_ptf_tbl["OFFSET"] *= units.arcsec
    ccsne_ptf_tbl["DA_PLANCK18"] = Planck18.angular_diameter_distance(ccsne_ptf_tbl["REDSHIFT"])
    ccsne_ptf_tbl["OFFSET_KPC_PLANCK18"] = ccsne_ptf_tbl["DA_PLANCK18"].to("kpc") * ccsne_ptf_tbl["OFFSET"].to("rad").value

    ccsne_ptf = {
        "table": ccsne_ptf_tbl,
        "c": ccsne_color,
        "ls": "solid",
        "label": "CCSNe", # (PTF)",
        "object_name": "OBJECT",
        "z": "REDSHIFT",
        "galfit_offset": "OFFSET",
        "galfit_offset_proj": "OFFSET_KPC_PLANCK18"
    }

    tables.append(ccsne_ptf)

    slsne_ptf_tbl = ccsne_ptf_tbl[["SLSN" in n for n in ccsne_ptf_tbl["TYPE"]]]
    slsne_color = "cyan"
    slsne_ptf = ccsne_ptf.copy()
    slsne_ptf.update(
        {
            "table": slsne_ptf_tbl,
            "c": slsne_color,
            "ls": "dashed",
            "label": "SLSNe" # (PTF)",
        }
    )
    tables.append(slsne_ptf)

    # CCSNe: Kelly & Kirshner 2012
    # For normalised offsets
    # From https://content.cld.iop.org/journals/0004-637X/759/2/107/revision1/apj441065t8_mrt.txt
    ccsne_kk_tbl = table.QTable.read(
        os.path.join(input_dir, "kelly+2012/apj441065t8_mrt.txt"),
        format="ascii"
    )
    ccsne_kk_color = ccsne_color
    ccsne_kk = {
        "table": ccsne_kk_tbl,
        "c": ccsne_color,
        "ls": "solid",
        "label": "CCSNe", # (Kelly and Kirshner, 2012)",
        "object_name": "SN",
        "galfit_offset_norm": "Offset"
    }

    tables.append(ccsne_kk)

    # Magnetars and other NS populations: Chrimes 202

    chrime_cols = {
        "object_name": "index",
        "galfit_offset_norm": "offset_norm_B",
        "galfit_offset_proj": "offset",
        "galfit_offset_norm_deproj": "offset_norm_B",
        "galfit_offset_disk": "offset"
    }

    chrimes_cats = {
        "Magnetars": {
            "c": "violet", # "#ff335c",
            "ls": "dashed"
        },
        "Pulsars": {
            "c": "green", # "#ff33c2"
            "ls": "dotted"
        },
        "HMXB": {
            "c": "grey", #c2ff33"
            "ls": "dashdot"
        },
        "LMXB": {
            "c": "black", #ff7033",
            "ls": (5, (10,3))
        },
    }

    for cat_name, cat_dict in chrimes_cats.items():
        cat_dict["label"] = cat_name # + " (Chrimes et al., 2021)"
        if not cat_name.endswith("s"):
            cat_dict["label"] += "s"
        offset = []
        cat = table.QTable.read(
            os.path.join(input_dir, f"chrimes+2021/data_{cat_name.lower()}.txt"),
            delimiter="\t",
            format="ascii.csv", guess=False,
        )
        cat["index"] = range(1, len(cat) + 1)
        for row in cat:
            if lib.isreal(row["offset"]):
                offset.append(row["offset"] * units.kpc)
            else:
                offset.append(-999 * units.kpc)
        cat["offset"] = offset
        cat_dict["table"] = cat
        cat_dict.update(chrime_cols)
        tables.append(cat_dict)

    # d633ff
    # ff33c2
    # ff335c
    # ff7033
    # ffd633
    # c2ff33

    # DAF7A6
    # FFC300
    # FF5733
    # C70039
    # 900C3F
    # 581845

    def populations(
            main_dict,
            col,
            clean_min = None,
            suffix: str = None
    ):

        fig = plt.figure(figsize=figsize)
        ax_lin = fig.add_subplot(1, 1, 1)
        axes = (ax_lin,)
        # axes = (ax_lin, ax_log)
        # ax_lin = fig.add_subplot(1, 2, 1)
        # ax_log = fig.add_subplot(1, 2, 2)

        if col in main_dict:
            col_ = main_dict[col]
        else:
            col_ = col

        main_tbl = main_dict["table"]
        # latex_commands.append(
        #     u.latex_command(
        #         command=f"NMain{col}-{main_dict['label']}",
        #         value=len(main_tbl[col_][main_tbl[col_] >= clean_min])
        #     )
        # )
        # latex_commands.append(
        #     u.latex_command(
        #         command=f"MedianMain{col}-{main_dict['label']}",
        #         value=np.median(main_tbl[col_][main_tbl[col_]>=clean_min])
        #     )
        # )
        # latex_commands.append(
        #     u.latex_command(
        #         command=f"MeanMain{col}-{main_dict['label']}",
        #         value=np.mean(main_tbl[col_][main_tbl[col_]>=clean_min])
        #     )
        # )

        for i, ax in enumerate(axes):
            log = bool(i)
            fig, _, _, _ = lib.cdf_prop(
                col=col_,
                tbl=main_tbl,
                fig=fig,
                ax=ax,
                name_col=main_dict["object_name"],
                label=main_dict["label"],
                filename=False,
                color=main_dict["c"],
                lw=lw_main,
                ls=main_dict["ls"],
                zorder=-1,
                log=log,
                clean_min=clean_min
            )

        for tbl_dict in tables:
            if tbl_dict is not main_dict and col in tbl_dict:
                col_this = tbl_dict[col]
                tbl_this = tbl_dict["table"]
                stats, fig, _, ax_lin = lib.compare_cat(
                    col_1=col_,
                    col_2=col_this,
                    tbl_1=main_tbl,
                    tbl_2=tbl_this,
                    name_col_1=main_dict["object_name"],
                    name_col_2=tbl_dict["object_name"],
                    label_1=main_dict["label"],
                    label_2=tbl_dict["label"],
                    clean_min=clean_min,
                    filename=False,
                    fig=fig,
                    # ax_cdf=ax_log,
                    ax_cdf=ax_lin,
                    # ax_2=ax_lin,
                    c_2=tbl_dict["c"],
                    legend=False,
                    plot_1=False,
                    do_hist=False,
                    p_on_plot=False,
                    log_and_linear=False,
                    # ls_2=tbl_dict["ls"]
                )
                latex_commands.append(stats["K-S"]["command"])
                latex_commands.append(stats["A-D"]["command"])
                column = tbl_this[col_this][tbl_this[col_this] >= clean_min]
                # latex_commands.append(
                #     u.latex_command(
                #         command=f"N{col}-{tbl_dict['label']}",
                #         value=len(column)
                #     )
                # )
                # latex_commands.append(
                #     u.latex_command(
                #         command=f"Median{col}-{tbl_dict['label']}",
                #         value=np.round(np.median(column), 3)
                #     )
                # )
                # latex_commands.append(
                #     u.latex_command(
                #         command=f"Mean{col}-{tbl_dict['label']}",
                #         value=np.mean(column)
                #     )
                # )

        ax_lin.legend(loc=(legend_x, legend_y), fontsize=fontsize_legend)
        # fig.tight_layout()
        fig.subplots_adjust(wspace=0.)
        fname = f"compare_populations_{col}"
        if suffix is not None:
            fname += "_" + suffix
        lib.savefig(
            fig,
            fname,
            subdir="offset_comparisons",
            tight=True
        )

    # Redshift
    populations(
        main_dict=craft_z,
        col="z",
        clean_min=0.
    )

    # Angular offset
    populations(
        main_dict=craft,
        col="galfit_offset",
        clean_min=0.
    )

    # Host-normalised offset
    populations(
        main_dict=craft,
        col="galfit_offset_norm",
        clean_min=0.
    )

    # Projected Offset
    populations(
        main_dict=craft_z,
        col="galfit_offset_proj",
        clean_min=0.
    )

    craft_deproj = craft.copy()
    craft_deproj["table"] = craft_galfit[craft_galfit["galfit_offset_norm_deproj"] < 100]

    # Deprojected, normalised offset
    populations(
        main_dict=craft_deproj,
        col="galfit_offset_norm_deproj",
        clean_min=0.
    )

    craft_deproj_z = craft.copy()
    craft_deproj_z["table"] = craft_galfit_z[craft_galfit_z["galfit_offset_norm_deproj"] < 100]

    # Disk offset
    populations(
        main_dict=craft_deproj_z,
        col="galfit_offset_disk",
        clean_min=0.
    )

    # Projected Offset, CCSNe v NS
    populations(
        main_dict=ccsne_ptf,
        col="galfit_offset_proj",
        clean_min=0.,
        suffix="ccsne"
    )

    # Normalised Offset, CCSNe v Magnetars
    populations(
        main_dict=ccsne_kk,
        col="galfit_offset_norm",
        clean_min=0.,
        suffix="ccsne"
    )

    commands_file = os.path.join(lib.tex_path, "commands_offsets_generated.tex")
    with open(commands_file, "w") as f:
        f.writelines(latex_commands)
    db_path = os.path.join(lib.dropbox_path, "commands")
    if os.path.isdir(lib.dropbox_path):
        shutil.copy(commands_file, db_path)


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
