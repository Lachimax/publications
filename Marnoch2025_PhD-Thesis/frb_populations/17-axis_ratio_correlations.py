#!/usr/bin/env python
# Code by Lachlan Marnoch, 2024

import os
import shutil

import numpy as np

from astropy import units

import frb.dm.igm as igm

import astropy.table as table
from astropy.cosmology import Planck18

import craftutils.utils as u
import craftutils.params as p
import craftutils.plotting as pl

from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
from matplotlib import pyplot as plt

import lib
from craftutils.utils import dequantify

from inspect import currentframe, getframeinfo

description = """
Performs and plots statistical tests for correlations between CRAFT FRB host properties.
"""

commands = []

def lts_3d(
        col_x_1, col_x_2,
        tbl_name,
        tbl,
        col_y: str = "dm_excess_rest",
        n_grid=100
):
    global commands
    fig, ax, fit_plane_1, props, commands = lib.lts_prop(
        col_x=[col_x_1, col_x_2],
        col_y=col_y,
        tbl_name=tbl_name,
        tbl=tbl,
        command_lines=commands
    )
    plt.close(fig)

    # Set up 3D Plot

    tbl = tbl[np.isfinite(tbl[col_x_1])]
    tbl = tbl[np.isfinite(tbl[col_x_2])]

    yy, equstrings = lib.y_from_lts(
        x_col=[col_x_1, col_x_2],
        y_data_col=col_y,
        tbl=tbl,
        f=fit_plane_1,
        n_grid=n_grid
    )

    fig = plt.figure(
        figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"])
    )

    # Add an axes
    ax = fig.add_subplot(111, projection='3d')

    x_col_1 = dequantify(tbl[col_x_1])
    x_col_2 = dequantify(tbl[col_x_2])
    y_un = tbl[col_y].unit
    if y_un is None:
        y_un = 1.
    y_col = dequantify(tbl[col_y])
    y_model_col = dequantify(tbl[col_y + "_model"])

    delta_1 = np.abs(np.max(x_col_1) - np.min(x_col_1))
    delta_2 = np.abs(np.max(x_col_2) - np.min(x_col_2))

    xx1, xx2 = np.meshgrid(
        np.linspace(
            np.min(x_col_1) - 0.1 * delta_1,
            np.max(x_col_1) + 0.1 * delta_1,
            n_grid
        ),
        np.linspace(
            np.min(x_col_2) - 0.1 * delta_2,
            np.max(x_col_2) + 0.1 * delta_2, n_grid
        )
    )

    print("EQUATION")
    # xx1 = np.linspace(craft_gI["g-I"].min(), craft_gI["g-I"].max(), 10000)
    # xx2 = np.linspace(craft_gI["galfit_axis_ratio"].min(), craft_gI["galfit_axis_ratio"].max(), 10000)

    print(equstrings["equation"])
    a = np.round(fit_plane_1.coef[0], 2)
    b_0 = np.round(fit_plane_1.coef[1], 2)
    p_0 = np.round(u.dequantify(np.median(x_col_1)), 1)
    b_1 = np.round(fit_plane_1.coef[2], 2)
    p_1 = np.round(u.dequantify(np.median(x_col_2)), 1)

    yy = a + b_0 * (xx1 - p_0) + b_1 * (xx2 - p_1)

    # # plot the surface
    ax.plot_surface(xx1, xx2, yy, alpha=0.3, cmap="Purples")
    ax.view_init(elev=30, azim=-45)

    # and plot the point
    ax.scatter(x_col_1, x_col_2, y_col, color='green')
    ax.scatter(x_col_1, x_col_2, y_model_col, color='black', marker="x")
    for i, row in enumerate(tbl):
        x_1 = x_col_1[i]
        x_2 = x_col_2[i]
        y = y_col[i]
        y_model = u.dequantify(row[col_y + "_model"])
        ax.plot(
            (x_1, x_1),
            (x_2, x_2),
            (y, y_model),
            c="black",
            ls=":",
            alpha=0.8
        )
    ax.set_xlabel(lib.nice_axis_label(col_x_1, tbl))
    ax.set_ylabel(lib.nice_axis_label(col_x_2, tbl))
    ax.set_zlabel(lib.nice_axis_label(col_y, tbl))
    fig.suptitle(equstrings["equation_latex"], fontsize=9)

    lib.savefig(
        fig,
        f"3D_lts_{col_y}+{col_x_1}+{col_x_2}_{tbl_name}",
        "correlations",
        tight=False
    )

    return fig, ax, fit_plane_1


def main(
        output_dir: str,
        input_dir: str,
        test_plot: bool = False,
        do_axis_ratios: bool = False,
        do_inclinations: bool = False,
        do_colours: bool = False,
        do_local_colours: bool = False,
        do_offsets: bool = False,
        do_dm_z: bool = False,
        do_3d: bool = False,
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    global commands

    do = [
        do_axis_ratios,
        do_inclinations,
        do_colours,
        do_local_colours,
        do_offsets,
        do_dm_z,
        # do_3d
    ]
    do_all = sum(do) == 0
    variables = locals().copy()
    print(f"{test_plot=}")
    for name, var in variables.items():
        if name.startswith("do_"):
            print(name, ":", var)

    n_cut = 2

    craft_galfit = table.QTable.read(lib.craft_galfit_path(0.2))

    craft_galfit_z = craft_galfit[craft_galfit["z"] > 0]

    # craft_galfit_z["dm_excess_rest_err"] = craft_galfit_z["dm_excess_err"] * (1 + craft_galfit_z["z"])

    craft_galfit_z["log_dm_excess_rest"] = np.log10(
        craft_galfit_z["dm_excess_rest"] / lib.dm_units
    )
    craft_galfit_z["log_dm_excess_rest_err"] = u.uncertainty_log10(
        craft_galfit_z["dm_excess_rest"] / lib.dm_units,
        craft_galfit_z["dm_excess_rest_err"] / lib.dm_units
    )

    craft_galfit_z["dm_exgal_ymw16"] = craft_galfit_z["dm"] - craft_galfit_z["dm_ism_ymw16"] - craft_galfit_z[
        "dm_mw_halo_pz19"]
    craft_galfit_z["dm_exgal_ymw16_err"] = np.sqrt(craft_galfit_z["dm_err"] ** 2 + craft_galfit_z["dm_ism_delta"] ** 2)
    craft_galfit_z["dm_excess_ymw16"] = craft_galfit_z["dm_exgal_ymw16"] - craft_galfit_z["dm_cosmic_avg"]
    craft_galfit_z["dm_excess_ymw16_err"] = craft_galfit_z["dm_exgal_ymw16_err"].copy()
    craft_galfit_z["dm_excess_rest_ymw16"] = craft_galfit_z["dm_excess_ymw16"] * (1 + craft_galfit_z["z"])
    craft_galfit_z["dm_excess_rest_ymw16_err"] = craft_galfit_z["dm_excess_ymw16_err"] * (1 + craft_galfit_z["z"])

    craft_galfit_tau = craft_galfit_z[craft_galfit_z["tau"] > 0]

    # AXIS RATIOS
    # ============================================================================================

    fig, ax, fit_1, props, commands = lib.lts_prop(
        "galfit_axis_ratio",
        "dm_excess_rest", craft_galfit_z,
        command_lines=commands
    )

    fig.savefig(
        os.path.join(
            lib.output_path, "correlations", "dmexcess_v_axisratio_accumulation",
            f"dmexcess_v_axisratio_n{len(craft_galfit_z)}.png"
        ),
        bbox_inches="tight"
    )


    plt.close(fig)
    lib.y_from_lts(
        x_col="galfit_axis_ratio",
        y_data_col="dm_excess_rest",
        tbl=craft_galfit_z,
        f=fit_1
    )
    lib.y_from_lts(
        x_col="galfit_axis_ratio",
        y_data_col="dm_excess_rest",
        tbl=craft_galfit,
        f=fit_1
    )

    craft_galfit["dm_excess_model"] = craft_galfit["dm_excess_rest_model"] / (1 + craft_galfit["z"])
    craft_galfit_z["dm_excess_model"] = craft_galfit_z["dm_excess_rest_model"] / (1 + craft_galfit_z["z"])

    craft_galfit["dm_inclination_corrected"] = craft_galfit["dm_exgal"] - craft_galfit["dm_excess_model"]
    craft_galfit_z["dm_inclination_corrected"] = craft_galfit_z["dm_exgal"] - craft_galfit_z["dm_excess_model"]

    craft_galfit_z["dm_minus_cosmic"] = craft_galfit_z["dm"] - craft_galfit_z["dm_cosmic_avg"]
    craft_galfit_z["dm_minus_cosmic_err"] = craft_galfit_z["dm_err"]

    craft_galfit_mag = craft_galfit_z.copy()
    craft_galfit_mag = craft_galfit_mag[craft_galfit_mag["galfit_mag"] < 21. * units.mag]

    craft_galfit_n = craft_galfit_z.copy()
    craft_galfit_n = craft_galfit_n[craft_galfit_n["galfit_n"] < n_cut]

    suspect = ["20220610A", "20190611B"]
    minus_suspect = craft_galfit_z[[n not in suspect for n in craft_galfit_z["name"]]]

    craft_g = lib.cut_to_band(tbl=craft_galfit_z, fil_name="g_HIGH")
    craft_gI = lib.cut_to_band(tbl=craft_g, fil_name="I_BESS")

    craft_gI["g"] = craft_gI["mag_best_vlt-fors2_g-HIGH"] - craft_gI["ext_gal_vlt-fors2_g-HIGH"]
    craft_gI["I"] = craft_gI["mag_best_vlt-fors2_I-BESS"] - craft_gI["ext_gal_vlt-fors2_I-BESS"]
    craft_gI["g-I"] = craft_gI["g"] - craft_gI["I"]
    craft_gI["g-I_err"] = np.sqrt(
        craft_gI["mag_best_vlt-fors2_g-HIGH_err"] ** 2 + craft_gI["mag_best_vlt-fors2_I-BESS_err"] ** 2
    )
    craft_gI["g_absolute"] = craft_gI["g"] - craft_gI["mu"]
    craft_gI["I_absolute"] = craft_gI["I"] - craft_gI["mu"]

    craft_gI["g-I_local"] = craft_gI["transient_position_surface_brightness_vlt-fors2_g-HIGH"] - craft_gI[
        "transient_position_surface_brightness_vlt-fors2_I-BESS"]
    craft_gI["g-I_local"] *= units.arcsec ** 2
    craft_gI["g-I_local_err"] = np.sqrt(
        craft_gI["transient_position_surface_brightness_vlt-fors2_g-HIGH_err"] ** 2 + craft_gI[
            "transient_position_surface_brightness_vlt-fors2_I-BESS_err"] ** 2
    ) * units.arcsec ** 2

    craft_gI.sort("g-I")
    craft_gI.write(os.path.join(lib.table_path, "craft_gI.ecsv"), overwrite=True)
    craft_gI.write(os.path.join(lib.table_path, "craft_gI.csv"), overwrite=True)

    craft_R = lib.cut_to_band(tbl=craft_galfit_z, fil_name="R_SPECIAL")
    craft_RK = lib.cut_to_band(tbl=craft_R, fil_name="Ks", instrument="vlt-hawki")
    print(len(craft_gI))

    craft_RK["R"] = craft_RK["mag_best_vlt-fors2_R-SPECIAL"] - craft_RK["ext_gal_vlt-fors2_R-SPECIAL"]
    craft_RK["K"] = craft_RK["mag_best_vlt-hawki_Ks"] - craft_RK["ext_gal_vlt-hawki_Ks"]
    craft_RK["R-K"] = craft_RK["R"] - craft_RK["K"]
    craft_RK["R-K_err"] = np.sqrt(
        craft_RK["mag_best_vlt-fors2_R-SPECIAL_err"] ** 2 + craft_RK["mag_best_vlt-hawki_Ks_err"] ** 2
    )
    craft_RK["R_absolute"] = craft_RK["R"] - craft_RK["mu"]
    craft_RK["K_absolute"] = craft_RK["K"] - craft_RK["mu"]

    craft_RK["R-K_local"] = craft_RK["transient_position_surface_brightness_vlt-fors2_R-SPECIAL"] - craft_RK[
        "transient_position_surface_brightness_vlt-hawki_Ks"]
    craft_RK["R-K_local"] *= units.arcsec ** 2
    craft_RK["R-K_local_err"] = np.sqrt(
        craft_RK["transient_position_surface_brightness_vlt-fors2_R-SPECIAL_err"] ** 2 + craft_RK[
            "transient_position_surface_brightness_vlt-hawki_Ks_err"] ** 2
    ) * units.arcsec ** 2

    craft_RK.sort("R-K")
    craft_RK.write(os.path.join(lib.table_path, "craft_RK.ecsv"), overwrite=True)
    craft_RK.write(os.path.join(lib.table_path, "craft_RK.csv"), overwrite=True)

    flimflam = craft_galfit_z[craft_galfit_z["in_flimflam_dr1"]]
    flimflam.sort("z")
    f_gas = 0.55
    f_igm = 0.59
    flimflam["dm_halos"] = [22.6, 13.4, 30.5, 120., 360.7, 321.4, 11.5, 37.4] * lib.dm_units * f_gas
    flimflam["dm_halos_err"] = [13.2, 15.7, 20.4, 33.1, 72.4, 92.1, 10.6, 23.5] * lib.dm_units * f_gas
    flimflam["dm_igm"] = [61.6, 171.2, 89.9, 116.1, 466.3, 326.5, 329.2, 559.9] * lib.dm_units * f_igm
    flimflam["dm_igm_err"] = [32., 106.4, 12., 5.6, 170.4, 94.7, 91., 202.8] * lib.dm_units * f_igm
    flimflam["dm_host_halo"] = [24.1, 35.8, 46.1, 24.2, 58.1, 37.4, 44., 45.8] * lib.dm_units
    flimflam["dm_host_halo_err"] = [5.5, 8.1, 10.4, 5.7, 13., 8.7, 10.4, 10.8] * lib.dm_units
    flimflam["dm_residual_flimflam"] = flimflam["dm_exgal"] - flimflam["dm_halos"] - flimflam["dm_igm"] - flimflam["dm_host_halo"]
    flimflam["dm_residual_flimflam_2"] = flimflam["dm_exgal"] - flimflam["dm_halos"] - flimflam["dm_igm"] #- flimflam["dm_host_halo"]
    flimflam["dm_residual_flimflam_err"] = np.sqrt(
        flimflam["dm_exgal_err"] ** 2 + flimflam["dm_halos_err"] ** 2 + flimflam["dm_igm_err"] ** 2 + flimflam[
            "dm_host_halo_err"] ** 2
    )
    flimflam["dm_residual_flimflam_2_err"] = np.sqrt(
        flimflam["dm_exgal_err"] ** 2 + flimflam["dm_halos_err"] ** 2 + flimflam["dm_igm_err"] ** 2
    )
    flimflam["dm_residual_flimflam_rest"] = flimflam["dm_residual_flimflam"] * (1 + flimflam["z"])
    flimflam["dm_residual_flimflam_rest_err"] = flimflam["dm_residual_flimflam_err"] * (1 + flimflam["z"])

    flimflam["dm_residual_flimflam_2_rest"] = flimflam["dm_residual_flimflam_2"] * (1 + flimflam["z"])
    flimflam["dm_residual_flimflam_2_rest_err"] = flimflam["dm_residual_flimflam_2_err"] * (1 + flimflam["z"])


    craft_mstar = craft_galfit_z.copy()
    craft_mstar = craft_mstar[craft_mstar["mass_stellar"] > 0.]
    craft_mstar["dm_excess_halo"] = craft_mstar["dm_excess"] - craft_mstar["dm_host_halo"]
    craft_mstar["dm_excess_halo_rest"] = craft_mstar["dm_excess_halo"] * (1 + craft_mstar["z"])
    craft_mstar["dm_excess_halo_rest_err"] = 10 * lib.dm_units

    bad_bulges = ["20190608B", "20220725A", "20231226A"]
    craft_wo_bulges = craft_galfit_z.copy()[[n not in bad_bulges for n in craft_galfit_z["name"]]]



    fig, ax, _, props, commands = lib.lts_prop(
        "galfit_axis_ratio",
        "log_dm_excess_rest",
        craft_galfit_z,
        command_lines=commands
        # clip=2.5
    )
    plt.close(fig)

    fig, ax, _, props, commands = lib.lts_prop(
        "galfit_axis_ratio",
        "log_dm_excess_rest",
        craft_galfit_n,
        tbl_name=f"n-cut-{n_cut}",
        command_lines=commands
        # clip=2.5
    )
    plt.close(fig)

    fig, ax, _, props, commands = lib.lts_prop(
        "galfit_axis_ratio",
        "log_dm_excess_rest",
        craft_galfit_mag,
        tbl_name="mag-cut-21",
        command_lines=commands
        # clip=2.5
    )
    plt.close(fig)

    if test_plot:
        exit()

    if do_all or do_axis_ratios:
        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_excess_rest",
            craft_wo_bulges,
            tbl_name="minus-bad-bulges",
            command_lines=commands
            # clip=2.5
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_excess_rest",
            craft_galfit_mag,
            tbl_name="mag-cut-21",
            command_lines=commands
            # clip=2.5
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_excess_rest",
            craft_galfit_n,
            tbl_name=f"n-cut-{n_cut}",
            command_lines=commands
            # clip=2.5
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_excess_rest",
            minus_suspect,
            tbl_name="minus-suspects",
            command_lines=commands
            # clip=2.5
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_excess_rest",
            craft_galfit_n[craft_galfit_n["galfit_mag"] < 21. * units.mag],
            tbl_name="n-cut-mag-cut",
            command_lines=commands
            # clip=2.5
        )
        plt.close(fig)

        # Miscellaneous DM variants

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_excess",
            craft_galfit_z,
            tbl_name="craft_galfit",
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_excess_rest_ymw16",
            craft_galfit_z,
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_exgal",
            craft_galfit,
            tbl_name="craft_galfit",
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "dm",
            craft_galfit_z,
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_minus_cosmic",
            craft_galfit_z,
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "galfit_mag",
            craft_galfit_z,
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_residual_flimflam_rest",
            flimflam,
            command_lines=commands
        )
        plt.close(fig)

        # fig, ax, _, props, commands = lib.lts_prop(
        #     "galfit_axis_ratio",
        #     "dm_residual_flimflam_2_rest",
        #     flimflam,
        #     command_lines=commands
        # )
        # plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_excess_halo_rest",
            craft_mstar,
            command_lines=commands
        )
        plt.close(fig)

    lib.scatter_prop(
        "galfit_axis_ratio",
        "dm_excess_rest",
        craft_galfit_z,
        fit=True,
    )

    lib.scatter_prop(
        "galfit_axis_ratio",
        "dm",
        craft_galfit_z,
        fit=True,
    )
    plt.close()

    lib.scatter_prop(
        "galfit_axis_ratio",
        "dm_minus_cosmic",
        craft_galfit_z,
        fit=True,
    )
    plt.close()

    # INCLINATION
    # ============================================================================================

    if do_all or do_inclinations:

        craft_galfit_13 = table.QTable.read(lib.craft_galfit_path(0.13))
        craft_galfit_13 = craft_galfit_13[craft_galfit_13["z"] > 0]

        craft_galfit_13["dm_excess_rest_err"] = craft_galfit_13["dm_excess_err"] * (1 + craft_galfit_13["z"])

        craft_galfit_13["dm_exgal_ymw16"] = craft_galfit_13["dm"] - craft_galfit_z["dm_ism_ymw16"] - craft_galfit_13[
            "dm_mw_halo_pz19"]
        craft_galfit_13["dm_exgal_ymw16_err"] = np.sqrt(
            craft_galfit_13["dm_err"] ** 2 + craft_galfit_13["dm_ism_delta"] ** 2)
        craft_galfit_13["dm_excess_ymw16"] = craft_galfit_13["dm_exgal_ymw16"] - craft_galfit_13["dm_cosmic_avg"]
        craft_galfit_13["dm_excess_ymw16_err"] = craft_galfit_13["dm_exgal_ymw16_err"].copy()
        craft_galfit_13["dm_excess_rest_ymw16"] = craft_galfit_13["dm_excess_ymw16"] * (1 + craft_galfit_13["z"])
        craft_galfit_13["dm_excess_rest_ymw16_err"] = craft_galfit_13["dm_excess_ymw16_err"] * (1 + craft_galfit_13["z"])

        for tbl in (craft_galfit_13, craft_galfit):

            tbl_mag = tbl[tbl["galfit_mag"] < 21. * units.mag]

            if tbl is craft_galfit_13:
                tbl_name = "q13"
                mag_tbl_name = "mag-cut-21" + tbl_name
                commands_ = None
            else:
                tbl_name = None
                mag_tbl_name = "mag-cut-21"
                commands_ = commands

            fig, ax, _, props, _ = lib.lts_prop(
                "galfit_inclination",
                "dm_excess_rest",
                tbl,
                tbl_name=tbl_name,
                command_lines=commands_
            )
            plt.close(fig)

            fig, ax, _, props, _ = lib.lts_prop(
                "galfit_inclination",
                "dm_excess_rest",
                tbl_mag,
                tbl_name=mag_tbl_name,
                command_lines=commands_
                # clip=2.5
            )
            plt.close(fig)

            # lib.scatter_prop("galfit_inclination", "dm_excess_rest", craft_galfit_z)
            # plt.close()

            # COS(I)
            # ============================================================================================

            fig, ax, _, props, _ = lib.lts_prop(
                "galfit_cos_inclination",
                "dm_excess_rest",
                tbl,
                tbl_name=tbl_name,
                command_lines=commands_
                # clip=2.5
            )
            plt.close(fig)

            fig, ax, _, props, _ = lib.lts_prop(
                "galfit_cos_inclination",
                "dm_excess_rest",
                tbl_mag,
                tbl_name=mag_tbl_name,
                command_lines=commands_
                # clip=2.5
            )
            plt.close(fig)

            # 1 / COS(I)
            # ============================================================================================

            # fig, ax, _, props, _ = lib.lts_prop(
            #     "galfit_1-cos_inclination",
            #     "dm_excess_rest",
            #     tbl[tbl["galfit_1-cos_inclination_err"] < 50],
            #     command_lines=commands_
            # )
            # plt.close()


    # OFFSETS
    # ============================================================================================

    if do_all or do_offsets:
        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_offset",
            "dm_excess_rest",
            craft_galfit_z,
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_offset_norm",
            "dm_excess_rest",
            craft_galfit_z[craft_galfit_z["galfit_offset_norm_err"] < 10.],
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_offset_proj",
            "dm_excess_rest",
            craft_galfit_z,
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_offset_deproj",
            "dm_excess_rest",
            craft_galfit_z[craft_galfit_z["galfit_offset_deproj_err"] < 100. * units.arcsec],
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_offset_norm_deproj",
            "dm_excess_rest",
            craft_galfit_z[craft_galfit_z["galfit_offset_norm_deproj_err"] < 100.],
            command_lines=commands
        )
        plt.close(fig)

    # DM-Z
    # ============================================================================================

    if do_all or do_dm_z:

        # All on the same plot

        fig, ax = plt.subplots()

        dm_cosmic, z_cosmic = igm.average_DM(z=1.1, cosmo=Planck18, cumul=True)
        cosmic_dict = {
            "dm": dm_cosmic,
            "dm_exgal": dm_cosmic,
            "dm_inclination_corrected": dm_cosmic,
            "dm_cosmic_nominal": dm_cosmic,
            "dm_excess": np.zeros(dm_cosmic.shape),
            "dm_excess_rest": np.zeros(dm_cosmic.shape),
            "dm_residual": np.zeros(dm_cosmic.shape)
        }

        scatter_props = {
            "marker": "o",
            # "facecolor": "none",
            # "edgecolor": "black",
            # "lw": 2,
            "alpha": 1,
        }

        ax.scatter(
            craft_galfit_z["z"],
            craft_galfit_z["dm_exgal"],
            edgecolor="purple",
            facecolor="none",
            label="CRAFT",
            **scatter_props
        )

        scatter_props["marker"] = "x"

        ax.scatter(
            craft_galfit_z["z"],
            craft_galfit_z["dm_inclination_corrected"],
            facecolor="darkgreen",
            label="CRAFT",
            **scatter_props
        )
        ax.plot(
            z_cosmic,
            cosmic_dict["dm_inclination_corrected"],
            label="$\mathrm{<DM_{cosmic}>}$",
            c="black"
        )
        ax.set_ylabel(
            lib.nice_axis_label(
                "dm_exgal",
                craft_galfit_z
            )
        )
        ax.set_xlabel(
            lib.nice_axis_label(
                "z",
                craft_galfit_z
            )
        )
        lib.savefig(fig=fig, filename="dm_exgal_corrected_combined", subdir="dm-z")

        fig, ax, _ = lib.scatter_prop(
            "z",
            "dm_exgal",
            craft_galfit_z
        )
        plt.close(fig)

        # On separate subplots

        fig = plt.figure(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] / 2))

        dm_cosmic, z_cosmic = igm.average_DM(z=0.8, cosmo=Planck18, cumul=True)
        cosmic_dict = {
            "dm": dm_cosmic,
            "dm_exgal": dm_cosmic,
            "dm_inclination_corrected": dm_cosmic,
            "dm_cosmic_nominal": dm_cosmic,
            "dm_excess": np.zeros(dm_cosmic.shape),
            "dm_excess_rest": np.zeros(dm_cosmic.shape),
            "dm_residual": np.zeros(dm_cosmic.shape)
        }

        scatter_props = {
            "marker": "o",
            # "facecolor": "none",
            # "edgecolor": "black",
            # "lw": 2,
            "alpha": 1,
        }

        craft_galfit_z["delta_dm_exgal_macquart"] = craft_galfit_z["dm_exgal"] - craft_galfit_z["dm_cosmic_avg"]
        rms_dm_exgal = np.sqrt(np.sum(craft_galfit_z["delta_dm_exgal_macquart"]**2) / len(craft_galfit_z))
        print("rms_dm_exgal ==", rms_dm_exgal)
        craft_galfit_z["delta_dm_inclination_macquart"] = craft_galfit_z["dm_inclination_corrected"] - craft_galfit_z["dm_cosmic_avg"]
        rms_dm_inclination = np.sqrt(np.sum(craft_galfit_z["delta_dm_inclination_macquart"] ** 2) / len(craft_galfit_z))
        print("rms_dm_inclination ==", rms_dm_inclination)

        ax = fig.add_subplot(1, 2, 1)
        ax.scatter(
            craft_galfit_z["z"],
            craft_galfit_z["dm_exgal"],
            edgecolor="purple",
            facecolor="none",
            label="CRAFT",
            **scatter_props
        )
        ax.text(
            x=0.02, y=0.99,
            s=f"RMSE $= {int(rms_dm_exgal.round().value)}" + r"\ \mathrm{pc}\ \mathrm{cm}^{-3}$",
            transform=ax.transAxes,
            verticalalignment="top",
        )
        ax.plot(
            z_cosmic,
            cosmic_dict["dm_inclination_corrected"],
            label="$\mathrm{<DM_{cosmic}>}$",
            c="black"
        )
        ax.set_ylabel(
            lib.nice_axis_label(
                "dm_exgal",
                craft_galfit_z
            )
        )
        ax.set_xlabel(
            lib.nice_axis_label(
                "z",
                craft_galfit_z
            )
        )
        ax.tick_params("both", labelsize=12)

        scatter_props["marker"] = "x"
        ax = fig.add_subplot(1, 2, 2)
        ax.scatter(
            craft_galfit_z["z"],
            craft_galfit_z["dm_inclination_corrected"],
            facecolor="darkgreen",
            label="CRAFT",
            **scatter_props
        )
        ax.text(
            x=0.02, y=0.99,
            s=f"RMSE $= {int(rms_dm_inclination.round().value)}" + r"\ \mathrm{pc}\ \mathrm{cm}^{-3}$",
            verticalalignment="top",
            transform=ax.transAxes
        )
        ax.plot(
            z_cosmic,
            cosmic_dict["dm_inclination_corrected"],
            label="$\mathrm{<DM_{cosmic}>}$",
            c="black"
        )
        ax.set_ylabel(
            " "
        )
        ax.set_xlabel(
            lib.nice_axis_label(
                "z",
                craft_galfit_z
            )
        )
        ax.set_yticklabels([], visible=False)
        ax.tick_params("both", labelsize=12)
        fig.subplots_adjust(wspace=0.)
        lib.savefig(fig=fig, filename="dm_exgal_corrected", subdir="dm-z", tight=True)

        fig, ax, _ = lib.scatter_prop(
            "z",
            "dm_exgal",
            craft_galfit_z
        )
        plt.close(fig)

        # lib.lts_prop("z", "dm_exgal", craft_galfit_z)
        # plt.close()

        # lib.scatter_prop("z", "dm", craft_galfit_z)
        # plt.close()
        # lib.lts_prop("z", "dm", craft_galfit_z)
        # plt.close()

    # COLOUR
    # ============================================================================================

    print(f"{do_all=}, {do_colours=}, {do_3d=}")
    if do_all or do_colours:
        print("\n", "=" * 50)
        print("COLOUR")
        print("=" * 50, "\n")

        fig, ax, _, props, commands = lib.lts_prop(
            "g-I",
            "dm_excess_rest",
            craft_gI,
            tbl_name="craft_gI",
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "g-I", "dm_excess_rest_model_res",
            craft_gI,
            tbl_name="craft_gI",
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "R-K", "dm_excess_rest",
            craft_RK,
            tbl_name="craft_RK",
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "R-K", "dm_excess_rest_model_res",
            craft_RK,
            tbl_name="craft_RK",
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "g-I",
            craft_gI,
            tbl_name="craft_gI",
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "galfit_axis_ratio",
            "R-K",
            craft_RK,
            tbl_name="craft_RK",
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "g-I",
            "galfit_n",
            craft_gI,
            tbl_name="craft_gI",
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "R-K",
            "galfit_n",
            craft_RK,
            tbl_name="craft_RK",
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, fit_plane_1, props = lib.lts_prop(
            col_x=["g-I", "galfit_axis_ratio"],
            col_y="dm_excess_rest",
            tbl_name="craft_gI",
            tbl=craft_gI,
        )
        plt.close(fig)


        # Set up 3D Plot

        yy, _ = lib.y_from_lts(
            x_col=["g-I", "galfit_axis_ratio"],
            y_data_col="dm_excess_rest",
            tbl=craft_gI,
            f=fit_plane_1
        )

        fig = plt.figure(
            figsize=(10, 8)
        )

        # Add an axes
        ax = fig.add_subplot(111, projection='3d')

        xx1, xx2 = np.meshgrid(
            np.linspace(craft_gI["g-I"].min().value - 0.1, craft_gI["g-I"].max().value + 0.1, 10000),
            np.linspace(craft_gI["galfit_axis_ratio"].min(), craft_gI["galfit_axis_ratio"].max(), 10000)
        )
        # xx1 = np.linspace(craft_gI["g-I"].min(), craft_gI["g-I"].max(), 10000)
        # xx2 = np.linspace(craft_gI["galfit_axis_ratio"].min(), craft_gI["galfit_axis_ratio"].max(), 10000)

        yy = lib.dm_units * (126.32 + -352.77 * (xx1 - 1.2) + -352.77 * (xx2 - 0.5))

        # # plot the surface
        ax.plot_surface(xx1, xx2, yy, alpha=0.2)

        # and plot the point
        ax.scatter(craft_gI["g-I"], craft_gI["galfit_axis_ratio"], craft_gI["dm_excess_rest"], color='green')
        for row in craft_gI:
            ax.plot(
                (row["g-I"].value, row["g-I"].value),
                (row["galfit_axis_ratio"], row["galfit_axis_ratio"]),
                (row["dm_excess_rest_model"].value, row["dm_excess_rest"].value),
                c="black")
        ax.set_xlabel(lib.nice_axis_label("g-I", craft_gI))
        ax.set_ylabel(lib.nice_axis_label("galfit_axis_ratio", craft_gI))
        ax.set_zlabel(lib.nice_axis_label("dm_excess_rest", craft_gI))

        lib.savefig(fig, "3D", "correlations", tight=True)

        if do_3d:
            print("\n", "=" * 50)
            print("IN 3-D!")
            print("=" * 50, "\n")

            fig, ax, _ = lts_3d(
                col_x_1="g-I",
                col_x_2="galfit_axis_ratio",
                tbl_name="craft_gI",
                tbl=craft_gI,
                col_y="dm_excess_rest"
            )
            plt.close(fig)

            lib.scatter_prop(
                "g-I",
                "dm_excess_rest",
                craft_gI,
                tbl_name="craft_gI",
            )
            plt.close()

            lib.scatter_prop(
                "g-I",
                "dm_excess_rest_res",
                craft_gI,
                tbl_name="craft_gI",
            )
            plt.close()

            lib.scatter_prop(
                "g-I", "galfit_axis_ratio",
                craft_gI, tbl_name="craft_gI"
            )
            plt.close()

            mk = "o"
            fig = plt.figure(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] / 2))
            ax_gI = fig.add_subplot(1, 2, 1)
            ax_gI.hist2d(
                np.array(primus_gI["I_absolute"]),
                np.array(primus_gI["g-I"]),
                bins=(100, 100),
                cmap='binary',
                norm=ImageNormalize(stretch=SqrtStretch())
            )

            lib.print_stats("g-I", craft_gI)
            lib.print_stats("g-I_err", craft_gI)
            lib.print_stats("g_absolute", craft_gI)
            lib.print_stats("mag_best_vlt-fors2_g-HIGH_err", craft_gI)
            lib.print_stats("I_absolute", craft_gI)
            lib.print_stats("mag_best_vlt-fors2_I-BESS_err", craft_gI)

            ax_gI.errorbar(
                craft_gI["I_absolute"],
                craft_gI["g-I"],
                xerr=craft_gI["mag_best_vlt-fors2_I-BESS_err"],
                yerr=craft_gI["g-I_err"],
                fmt=".",
                c="black",
                capsize=1,
                zorder=-1
            )
            ax_gI.scatter(
                craft_gI["I_absolute"],
                craft_gI["g-I"],
                marker=mk,
                c=craft_gI["z"],
                vmin=np.min(craft_galfit_z["z"]),
                vmax=np.max(craft_galfit_z["z"])
            )
            ax_gI.invert_xaxis()
            # ax_gI.invert_yaxis()

            lib.print_stats("R-K", craft_RK)
            lib.print_stats("R-K_err", craft_RK)
            lib.print_stats("R", craft_RK)
            lib.print_stats("mag_best_vlt-fors2_R-SPECIAL_err", craft_RK)
            lib.print_stats("K", craft_RK)
            lib.print_stats("mag_best_vlt-hawki_Ks_err", craft_RK)

            ax_RK = fig.add_subplot(1, 2, 2)
            cax = ax_RK.scatter(
                craft_RK["K_absolute"],
                craft_RK["R-K"],
                marker=mk,
                c=craft_RK["z"],
                vmin=np.min(craft_galfit_z["z"]),
                vmax=np.max(craft_galfit_z["z"])
            )
            ax_RK.invert_xaxis()
            # ax_RK.invert_yaxis()
            fig.colorbar(cax)
            lib.savefig(fig, "CMD")

    # Local Colour
    # ============================================================================================

    if do_all or do_local_colours:
        fig, ax, _, props, commands = lib.lts_prop(
            "g-I_local",
            "dm_excess_rest",
            craft_gI,
            tbl_name="craft_gI",
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "g-I_local",
            "g-I",
            craft_gI,
            tbl_name="craft_gI",
            command_lines=commands
        )
        plt.close(fig)

        fig, ax, _, props, commands = lib.lts_prop(
            "R-K_local",
            "dm_excess_rest",
            craft_RK,
            tbl_name="craft_RK",
            command_lines=commands
        )
        plt.close(fig)

        if do_3d:
            fig, ax, _ = lts_3d(
                col_x_1="g-I_local",
                col_x_2="galfit_axis_ratio",
                tbl_name="craft_gI",
                tbl=craft_gI,
                col_y="dm_excess_rest"
            )
            plt.close(fig)

        # for row in craft_gI:
        #     print(row["name"], row["g-I"], row["g-I_local"])

        # fig, ax = plt.subplots()

    if do_all:
        commands_file = os.path.join(lib.tex_path, "commands_lts_generated.tex")
        with open(commands_file, "w") as f:
            f.writelines(commands)
        db_path = os.path.join(lib.dropbox_path, "commands")
        if os.path.isdir(lib.dropbox_path):
            shutil.copy(commands_file, db_path)

    new_table_dir = os.path.join(output_dir, "tables", "downstream")
    os.makedirs(new_table_dir, exist_ok=True)

    craft_galfit.write(os.path.join(new_table_dir, "craft_galfit.ecsv"), overwrite=True)
    craft_galfit_z.write(os.path.join(new_table_dir, "craft_galfit_z.ecsv"), overwrite=True)
    craft_galfit_mag.write(os.path.join(new_table_dir, "craft_galfit_mag.ecsv"), overwrite=True)
    craft_galfit_n.write(os.path.join(new_table_dir, "craft_galfit_n.ecsv"), overwrite=True)
    minus_suspect.write(os.path.join(new_table_dir, "minus_suspect.ecsv"), overwrite=True)
    craft_gI.write(os.path.join(new_table_dir, "craft_gI.ecsv"), overwrite=True)
    craft_RK.write(os.path.join(new_table_dir, "craft_RK.ecsv"), overwrite=True)
    flimflam.write(os.path.join(new_table_dir, "flimflam.ecsv"), overwrite=True)
    craft_mstar.write(os.path.join(new_table_dir, "craft_mstar.ecsv"), overwrite=True)
    craft_wo_bulges.write(os.path.join(new_table_dir, "craft_wo_bulges.ecsv"), overwrite=True)


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
        "--test_plot",
        help="Do a test plot.",
        action="store_true"
    )

    parser.add_argument(
        "--do_axis_ratios",
        action="store_true"
    )

    parser.add_argument(
        "--do_inclinations",
        action="store_true"
    )

    parser.add_argument(
        "--do_colours",
        action="store_true"
    )

    parser.add_argument(
        "--do_local_colours",
        action="store_true"
    )

    parser.add_argument(
        "--do_offsets",
        action="store_true"
    )

    parser.add_argument(
        "--do_dm-z",
        action="store_true"
    )

    parser.add_argument(
        "--do_3d",
        action="store_true"
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        test_plot=args.test_plot,
        do_axis_ratios=args.do_axis_ratios,
        do_inclinations=args.do_inclinations,
        do_colours=args.do_colours,
        do_local_colours=args.do_local_colours,
        do_offsets=args.do_offsets,
        do_dm_z=args.do_dm_z,
        do_3d=args.do_3d
    )
