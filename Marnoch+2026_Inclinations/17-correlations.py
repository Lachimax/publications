#!/usr/bin/env python
# Code by Lachlan Marnoch, 2024

import os
import shutil

import numpy as np

from astropy import units

try:
    import frb.dm.igm as igm

    frb_code = True
except:
    frb_code = False
    print("frb code not installed or not functioning; some of this script will be skipped.")

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
Performs and plots statistical tests for correlations between properties of CRAFT FRBs and their hosts.
"""

commands = []


def lts_3d(
        col_x_1,
        col_x_2,
        tbl_name,
        tbl,
        col_y: str = "dm_excess_rest",
        n_grid=100,
        latex_table=None
):
    global commands
    fig, ax, fit_plane_1, props, commands, tbl_str = lib.lts_prop(
        col_x=[col_x_1, col_x_2],
        col_y=col_y,
        tbl_name=tbl_name,
        tbl=tbl,
        command_lines=commands
    )
    latex_table += tbl_str
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
        do_non_paper: bool = False,
        do_dm_igm_err: bool = True
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
    if do_dm_igm_err:
        craft_galfit["dm_igm_err"] = 0.2 * craft_galfit["dm_igm_avg"]
        craft_galfit["dm_excess_err"] = np.sqrt(
            craft_galfit["dm_err"] ** 2 + craft_galfit["dm_mw_err"] ** 2 + craft_galfit["dm_igm_err"] ** 2)
        craft_galfit["dm_excess_rest_err"] = craft_galfit["dm_excess_err"] * (1 + craft_galfit["z"])

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

    latex_table = []

    fig, ax, fit_1, props, commands, tbl_str = lib.lts_prop(
        "galfit_axis_ratio",
        "dm_excess_rest",
        craft_galfit_z,
        command_lines=commands
    )
    # latex_table += tbl_str

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
    flimflam["dm_residual_flimflam"] = flimflam["dm_exgal"] - flimflam["dm_halos"] - flimflam["dm_igm"] - flimflam[
        "dm_host_halo"]
    flimflam["dm_residual_flimflam_2"] = flimflam["dm_exgal"] - flimflam["dm_halos"] - flimflam[
        "dm_igm"]  # - flimflam["dm_host_halo"]
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

    kovacs = table.QTable.read(os.path.join(input_dir, "Kovacs+24.csv"))
    kovacs["inclination"] *= units.deg
    kovacs["dm_host"] *= lib.dm_units
    ax.plot(kovacs["inclination"], kovacs["dm_host"], label="Kovacs+24", ls="-.", lw="1.5", c="green")
    lib.savefig(fig=fig, filename="kovacs_lts_dm_excess_rest-v-galfit_inclination", subdir="correlations")

    plt.close(fig)

    if test_plot:
        with open(os.path.join(output_path, "correlations.tex"), "w") as f:
            f.writelines(latex_table)
        exit()

    samples = {
        "full": craft_galfit_z,
        "mag-cut-21": craft_galfit_mag,
        f"n-cut-{n_cut}": craft_galfit_n,
        "minus-bad-bulges": craft_wo_bulges,
    }

    for name, tbl in samples.items():

        latex_table = []

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_excess",
            tbl=tbl,
            tbl_name=name,
            command_lines=commands,
            narrow=True,
            # clip=2.5
        )
        latex_table += tbl_str
        plt.close(fig)

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_excess_rest",
            tbl=tbl,
            tbl_name=name,
            command_lines=commands,
            narrow=True,
            # clip=2.5
        )
        latex_table += tbl_str
        plt.close(fig)

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_excess_rest_ymw16",
            tbl=tbl,
            tbl_name=name,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_exgal",
            tbl=tbl,
            tbl_name=name,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_axis_ratio",
            "dm",
            tbl=tbl,
            tbl_name=name,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_axis_ratio",
            "dm_minus_cosmic",
            tbl=tbl,
            tbl_name=name,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_axis_ratio",
            "galfit_mag",
            tbl=tbl,
            tbl_name=name,
            command_lines=commands
        )

        fig, ax, _, props, _, tbl_str = lib.lts_prop(
            "galfit_inclination",
            "dm_excess_rest",
            tbl=tbl,
            tbl_name=name,
            narrow=True,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)

        # COS(I)
        # ============================================================================================

        fig, ax, _, props, _, tbl_str = lib.lts_prop(
            "galfit_cos_inclination",
            "dm_excess_rest",
            tbl,
            tbl_name=name,
            narrow=True,
            command_lines=commands
            # clip=2.5
        )
        latex_table += tbl_str
        plt.close(fig)

        # OFFSETS
        # ============================================================================================

        # if (do_all or do_offsets) and do_non_paper:
        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_offset",
            "dm_excess_rest",
            tbl=tbl,
            tbl_name=name,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_offset_norm",
            "dm_excess_rest",
            tbl[tbl["galfit_offset_norm_err"] < 10.],
            tbl_name=name,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_offset_proj",
            "dm_excess_rest",
            tbl=tbl,
            tbl_name=name,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_offset_deproj",
            "dm_excess_rest",
            tbl[tbl["galfit_offset_deproj_err"] < 100. * units.arcsec],
            tbl_name=name,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_offset_norm_deproj",
            "dm_excess_rest",
            tbl=tbl[tbl["galfit_offset_norm_deproj_err"] < 100.],
            tbl_name=name,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)


        with open(os.path.join(output_path, f"correlations_{name}.tex"), "w") as f:
            f.writelines(latex_table)

    # DM-Z
    # ============================================================================================

    if frb_code:
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

        fig = plt.figure(figsize=(pl.textwidths["PASA"] / 2, pl.textwidths["PASA"]))

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
        rms_dm_exgal = np.sqrt(np.sum(craft_galfit_z["delta_dm_exgal_macquart"] ** 2) / len(craft_galfit_z))
        print("rms_dm_exgal ==", rms_dm_exgal)
        craft_galfit_z["delta_dm_inclination_macquart"] = craft_galfit_z["dm_inclination_corrected"] - craft_galfit_z[
            "dm_cosmic_avg"]
        rms_dm_inclination = np.sqrt(np.sum(craft_galfit_z["delta_dm_inclination_macquart"] ** 2) / len(craft_galfit_z))
        print("rms_dm_inclination ==", rms_dm_inclination)

        ax = fig.add_subplot(2, 1, 1)
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
            " "
        )
        ax.set_xticklabels([], visible=False)

        scatter_props["marker"] = "x"
        ax = fig.add_subplot(2, 1, 2)
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

    samples = {
        "gI": craft_gI,
        "RK": craft_RK
    }

    c_keys = {
        "gI": "g-I",
        "RK": "R-K"
    }

    print(f"{do_all=}, {do_colours=}, {do_3d=}")

    # if (do_all or do_colours) and do_non_paper:

    for name, tbl in samples.items():

        latex_table = []

        c_key = c_keys[name]

        print("\n", "=" * 50)
        print("COLOUR")
        print("=" * 50, "\n")

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            c_key,
            "dm_excess_rest",
            tbl=tbl,
            tbl_name=name,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)

        # fig, ax, _, props, commands, tbl_str = lib.lts_prop(
        #     c_key,
        #     "dm_excess_rest_model_res",
        #     tbl=tbl,
        #     tbl_name=name,
        #     command_lines=commands
        # )
        # # latex_table += tbl_str
        # plt.close(fig)

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            "galfit_axis_ratio",
            c_key,
            tbl=tbl,
            tbl_name=name,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)

        fig, ax, _, props, commands, tbl_str = lib.lts_prop(
            c_key,
            "galfit_n",
            tbl=tbl,
            tbl_name=name,
            command_lines=commands
        )
        latex_table += tbl_str
        plt.close(fig)

    # FLIMFLAM ONLY

    latex_table = []

    fig, ax, _, props, commands, tbl_str = lib.lts_prop(
        "galfit_axis_ratio",
        "dm_residual_flimflam_rest",
        flimflam,
        command_lines=commands
    )
    latex_table += tbl_str
    plt.close(fig)

    # MSTAR ONLY

    fig, ax, _, props, commands, tbl_str = lib.lts_prop(
        "galfit_axis_ratio",
        "dm_excess_halo_rest",
        craft_mstar,
        command_lines=commands
    )
    latex_table += tbl_str
    plt.close(fig)

    with open(os.path.join(output_path, "correlations_misc.tex"), "w") as f:
        f.writelines(latex_table)


    if do_all and do_non_paper:
        commands_file = os.path.join(lib.tex_path, "commands_lts_generated.tex")
        with open(commands_file, "w") as f:
            f.writelines(commands)
        if isinstance(lib.dropbox_path, str):
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

    parser.add_argument(
        "--do_non_paper",
        action="store_true"
    )

    parser.add_argument(
        "--no_dm_igm_err",
        action="store_false"
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
        do_3d=args.do_3d,
        do_non_paper=args.do_non_paper,
        do_dm_igm_err=args.no_dm_igm_err
    )
