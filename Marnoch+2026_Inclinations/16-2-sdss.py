#!/usr/bin/env python
# Code by Lachlan Marnoch, 2024

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

import astropy.table as table
import astropy.units as units
from astropy.io import fits
from astropy import cosmology

import craftutils.utils as u
import craftutils.params as p
import craftutils.plotting as pl

from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid

import lib

description = """
Performs statistical tests of CRAFT host properties against SDSS DR16 (SDSS Collaboration 2020).
"""

latex_commands = []

wspace = 0.05
tight = True


def compare_galfit(
        tbl_craft,
        tbl_sdss,
        q_0: float,
        label_1: str = "CRAFT",
        label_2: str = "SDSS",
        skip_commands: bool = False
):
    stats = {}
    def add_stats():
        if not skip_commands:
            for c in (
                    stats["n_2_command"],
                    stats["K-S"]["command"],
                    stats["A-D"]["command"]
            ):
                if c not in latex_commands:
                    latex_commands.append(c)

    if q_0 == 0.2:
        suffix = ""
    else:
        suffix = f"_{q_0}"

    all_stats = {}
    fig = plt.figure(figsize=[pl.textwidths["mqthesis"], pl.textheights["mqthesis"]])
    n_rows = 3

    colname = "galfit_axis_ratio"

    stats, _, _, _ = lib.compare_cat(
        col_1=colname, col_2="expAB_r",
        tbl_1=tbl_craft, tbl_2=tbl_sdss,
        label_1=label_1, label_2=label_2,
        clean_min=0., clean_max=1.,
        fig=fig,
        filename=False,
        # n_cols=2,
        n_rows=n_rows,
        n=1,
        legend=False,
        name_col_1="object_name",
        name_col_2="objID",
        mc_bounds=True,
        do_hist=False
    )
    all_stats[colname] = stats
    add_stats()

    colname = "galfit_cos_inclination"

    stats, _, _, _ = lib.compare_cat(
        col_1=colname, col_2="cos_i",
        tbl_1=tbl_craft, tbl_2=tbl_sdss,
        label_1=label_1, label_2=label_2,
        clean_min=0., clean_max=1.,
        fig=fig,
        filename=False,
        # n_cols=2,
        n_rows=n_rows,
        n=2,
        legend=False,
        name_col_1="object_name",
        name_col_2="objID",
        mc_bounds=True,
        do_hist=False
    )
    all_stats[colname] = stats
    add_stats()

    fig.subplots_adjust(
        wspace=wspace,
        hspace=0.4
    )

    filename = f"compare_galfit_inclinations_{lib.sanitise_filename(filename=label_1)}-v-{lib.sanitise_filename(filename=label_2)}" + suffix
    p.save_params(
        os.path.join(lib.output_path, "sdss", filename + ".yaml"),
        all_stats
    )
    lib.savefig(
        filename=filename,
        fig=fig,
        subdir="sdss",
        tight=tight
    )


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    os.makedirs(os.path.join(lib.output_path, "sdss"), exist_ok=True)
    q_0 = 0.2

    craft_galfit = table.QTable.read(lib.craft_galfit_path(q_0))

    sdss = table.QTable.read(os.path.join(lib.input_path, "SDSS.csv"))

    sdss = u.inclination_table(
        sdss,
        axis_ratio_column="expAB_r",
        inclination_column="i",
        cos_column="cos_i",
        q_0=q_0,
    )

    skip_commands = False

    compare_galfit(
        craft_galfit,
        sdss,
        q_0=q_0,
        skip_commands=skip_commands
    )
    # compare_galfit(
    #     craft_galfit[craft_galfit["galfit_mag"] < 21. * units.mag],
    #     acsgc_v_mag[acsgc_v_mag["MAG_GALFIT"] < 21 * units.mag],
    #     label_1="CRAFT, mag < 21",
    #     label_2="ACS-GC, mag < 21",
    #     q_0=q_0,
    #     skip_commands=skip_commands
    # )

    bad_bulges = ["20190608B", "20220725A", "20231226A"]
    craft_wo_bulges = craft_galfit.copy()[[n not in bad_bulges for n in craft_galfit["name"]]]

    compare_galfit(
        craft_wo_bulges,
        sdss,
        label_1=f"CRAFT, poor fits removed",
        q_0=q_0,
        skip_commands=skip_commands
    )

    commands_file = os.path.join(lib.tex_path, "commands_sdss_generated.tex")
    with open(commands_file, "w") as f:
        f.writelines(latex_commands)
    if isinstance(lib.dropbox_path, str):
        db_path = os.path.join(lib.dropbox_path, "commands")
        if os.path.isdir(lib.dropbox_path):
            shutil.copy(commands_file, db_path)

    fig = plt.figure(figsize=(0.6 * pl.textwidths["mqthesis"], 0.6 * pl.textwidths["mqthesis"]))

    tbl_craft = craft_galfit
    tbl_sdss =  sdss

    name_col = "object_name"

    label_1 = "CRAFT"
    label_2 = "SDSS"

    colname = "galfit_cos_inclination"
    stats, _, _, _ = lib.compare_cat(
        col_1=colname, col_2="cos_i",
        tbl_1=tbl_craft, tbl_2=tbl_sdss,
        label_1=label_1, label_2=label_2,
        clean_min=0., clean_max=1.,
        fig=fig,
        filename=False,
        do_hist=False,
        n=3,
        legend=False,
        name_col_1=name_col,
        name_col_2="objID",
        n_rows=1,
        n_cols=1
    )

    lib.savefig(fig=fig, filename="CDF_cosi", subdir="sdss", tight=True)


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
