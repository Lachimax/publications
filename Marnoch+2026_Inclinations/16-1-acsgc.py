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
Performs statistical tests of CRAFT host properties against the ACS-GC (Griffith et al 2012).
"""

latex_commands = []

wspace = 0.05
tight = True


def compare_galfit(
        tbl_craft,
        tbl_acs,
        q_0: float,
        band_tbl=False,
        label_1: str = "CRAFT",
        label_2: str = "ACS-GC",
        both_acs: bool = False,
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

    if both_acs:
        colname = "BA_GALFIT"
        name_col = "OBJNO"
    elif band_tbl:
        colname = "axis_ratio"
        name_col = "object_name"
    else:
        colname = "galfit_axis_ratio"
        name_col = "object_name"
    stats, _, _, _ = lib.compare_cat(
        col_1=colname, col_2="BA_GALFIT",
        tbl_1=tbl_craft, tbl_2=tbl_acs,
        label_1=label_1, label_2=label_2,
        clean_min=0., clean_max=1.,
        fig=fig,
        filename=False,
        # n_cols=2,
        n_rows=n_rows,
        n=1,
        legend=False,
        name_col_1=name_col,
        mc_bounds=True,
        do_hist=False
    )
    all_stats[colname] = stats
    add_stats()

    if both_acs:
        colname = "COS_I_GALFIT"
    else:
        colname = "galfit_cos_inclination"
    stats, _, _, _ = lib.compare_cat(
        col_1=colname, col_2="COS_I_GALFIT",
        tbl_1=tbl_craft, tbl_2=tbl_acs,
        label_1=label_1, label_2=label_2,
        clean_min=0., clean_max=1.,
        fig=fig,
        filename=False,
        # n_cols=2,
        n_rows=n_rows,
        n=2,
        legend=False,
        name_col_1=name_col,
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
        os.path.join(lib.output_path, "acsgc", filename + ".yaml"),
        all_stats
    )
    lib.savefig(
        filename=filename,
        fig=fig,
        subdir="acsgc",
        tight=tight
    )


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    os.makedirs(os.path.join(lib.output_path, "acsgc"), exist_ok=True)
    input_dir = os.path.join(lib.input_path, "ACS-GC")

    q_0 = 0.2

    craft_galfit = table.QTable.read(lib.craft_galfit_path(q_0))

    n_cut = 2.

    craft_galfit_n = craft_galfit[craft_galfit["galfit_n"] < n_cut]
    craft_galfit_z = craft_galfit[craft_galfit["z"] > 0]

    z_max = np.max(craft_galfit_z["z"])
    mag_max = np.max(craft_galfit["galfit_mag"])

    # For converting HST pixels to sky angles
    pix_scale_acs = 0.03 * units.arcsec / units.pix

    goods = fits.open(os.path.join(input_dir, "goods_v_i_public_catalog_V2.0.fits"))
    goods_galfit = table.QTable(goods[1].data)
    goods_z = goods_galfit[goods_galfit["Z"] < z_max]
    goods_z = goods_z[goods_z["Z"] > 0.]

    cosmos = fits.open(os.path.join(input_dir, "cosmos_i_public_catalog_V2.0.fits"))
    cosmos_galfit = table.QTable(cosmos[1].data)
    cosmos_z = cosmos_galfit[cosmos_galfit["Z"] < z_max]
    cosmos_z = cosmos_z[cosmos_z["Z"] > 0.]

    egs = fits.open(os.path.join(input_dir, "egs_v_i_public_catalog_V2.0.fits"))
    egs_galfit = table.QTable(egs[1].data)
    egs_z = egs_galfit[egs_galfit["Z"] < z_max]
    egs_z = egs_z[egs_z["Z"] > 0.]

    gems = fits.open(os.path.join(input_dir, "gems_v_z_public_catalog_V2.0.fits"))
    gems_galfit = table.QTable(gems[1].data)
    gems_z = gems_galfit[gems_galfit["Z"] < z_max]
    gems_z = gems_z[gems_z["Z"] > 0.]


    acsgc_v = table.vstack([gems_galfit, egs_galfit, goods_galfit])
    for colname in acsgc_v.colnames:
        if colname.endswith("_LOW"):
            new_col = colname.replace("_LOW", "")
            acsgc_v[new_col] = acsgc_v[colname]
            acsgc_v.remove_column(colname)
        elif colname.endswith("_HI"):
            acsgc_v.remove_column(colname)
    acsgc_v = acsgc_v[acsgc_v["FLAG_GALFIT"] == 0]
    acsgc_v["PA_GALFIT"] *= units.deg
    acsgc_v["RE_GALFIT"] *= units.pix
    acsgc_v["RE_GALFIT_ANG"] = acsgc_v["RE_GALFIT"] * pix_scale_acs
    acsgc_v["MAG_GALFIT"] *= units.mag
    acsgc_v["MAGERR_GALFIT"] *= units.mag

    acsgc_v = u.inclination_table(
        acsgc_v,
        axis_ratio_column="BA_GALFIT",
        inclination_column="I_GALFIT",
        cos_column="COS_I_GALFIT",
        q_0=q_0,
        # q_0=acsgc_v["q_0"]
    )
    # acsgc_v["COS_I_GALFIT"] = np.cos(acsgc_v["I_GALFIT"])
    acsgc_v_z = acsgc_v[acsgc_v["Z"] < z_max]
    acsgc_v_z = acsgc_v_z[acsgc_v_z["Z"] > 0.]
    acsgc_v_z["DA"] = cosmology.Planck18.angular_diameter_distance(acsgc_v_z["Z"])
    acsgc_v_z["RE_GALFIT_PROJ"] = acsgc_v_z["RE_GALFIT_ANG"].to("rad").value * acsgc_v_z["DA"].to("kpc")
    acsgc_v_mag = acsgc_v[acsgc_v["MAG_GALFIT"] < mag_max]
    acsgc_v_n = acsgc_v_mag[acsgc_v_mag["N_GALFIT"] < n_cut]

    skip_commands = False

    compare_galfit(
        craft_galfit,
        acsgc_v_mag,
        q_0=q_0,
        skip_commands=skip_commands
    )
    compare_galfit(
        craft_galfit[craft_galfit["galfit_mag"] < 21. * units.mag],
        acsgc_v_mag[acsgc_v_mag["MAG_GALFIT"] < 21 * units.mag],
        label_1="CRAFT, mag < 21",
        label_2="ACS-GC, mag < 21",
        q_0=q_0,
        skip_commands=skip_commands
    )
    compare_galfit(
        craft_galfit_n,
        acsgc_v_n,
        label_1=f"CRAFT, $n < {int(n_cut)}$",
        label_2=f"ACS-GC, $n < {int(n_cut)}$",
        q_0=q_0,
        skip_commands=skip_commands
    )

    bad_bulges = ["20190608B", "20220725A", "20231226A"]
    craft_wo_bulges = craft_galfit.copy()[[n not in bad_bulges for n in craft_galfit_z["name"]]]

    compare_galfit(
        craft_wo_bulges,
        acsgc_v,
        label_1=f"CRAFT, poor fits removed",
        q_0=q_0,
        skip_commands=skip_commands
    )

    commands_file = os.path.join(lib.tex_path, "commands_acsgc_generated.tex")
    with open(commands_file, "w") as f:
        f.writelines(latex_commands)
    if isinstance(lib.dropbox_path, str):
        db_path = os.path.join(lib.dropbox_path, "commands")
        if os.path.isdir(lib.dropbox_path):
            shutil.copy(commands_file, db_path)

    fig = plt.figure(figsize=(0.6 * pl.textwidths["mqthesis"], 0.6 * pl.textwidths["mqthesis"]))

    tbl_craft = craft_galfit
    tbl_acs =  acsgc_v_mag

    name_col = "object_name"

    label_1 = "CRAFT"
    label_2 = "ACS-GC"

    colname = "galfit_cos_inclination"
    stats, _, _, _ = lib.compare_cat(
        col_1=colname, col_2="COS_I_GALFIT",
        tbl_1=tbl_craft, tbl_2=tbl_acs,
        label_1=label_1, label_2=label_2,
        clean_min=0., clean_max=1.,
        fig=fig,
        filename=False,
        do_hist=False,
        n=3,
        legend=False,
        name_col_1=name_col,
        n_rows=1,
        n_cols=1
    )

    lib.savefig(fig=fig, filename="CDF_cosi", subdir="acsgc", tight=True)


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
