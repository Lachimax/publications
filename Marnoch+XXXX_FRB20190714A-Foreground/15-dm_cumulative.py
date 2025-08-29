#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np
import matplotlib.pyplot as plt

from astropy import units, table
from astropy.cosmology import Planck18

from frb.dm.igm import average_DM
import craftutils.utils as u
import craftutils.plotting as pl

import lib

description = """
Generates the cumulative DM figures.
"""

frb = lib.fld.frb
step_size_halo = 1 * units.kpc


def cumulative_dm_tbl(cosmic_tbl, property_dict, halo_tbl, smhm="K18"):
    cosmic_tbl = cosmic_tbl["z", "comoving_distance", "dm_cosmic_avg", "dm_igm"]
    cosmic_tbl["dm_halos_emp"] = np.zeros(len(cosmic_tbl)) * lib.dm_units
    cosmic_tbl["dm_halos_emp_var"] = np.zeros(len(cosmic_tbl)) * lib.dm_units ** 2
    cosmic_tbl["dm_halos_emp_var_plus"] = np.zeros(len(cosmic_tbl)) * lib.dm_units ** 2
    cosmic_tbl["dm_halos_emp_var_minus"] = np.zeros(len(cosmic_tbl)) * lib.dm_units ** 2
    halo_tbl.sort("z")
    for row in halo_tbl:
        z = row['z']
        to_change = cosmic_tbl["z"] >= z
        dm = row[f"dm_halo_mc_{smhm}"]
        name = row["id_short"]
        cosmic_tbl["dm_halos_emp"][to_change] += dm
        cosmic_tbl["dm_halos_emp_var"][to_change] += row[f"dm_halo_mc_{smhm}_err"] ** 2
        cosmic_tbl["dm_halos_emp_var_plus"][to_change] += row[f"dm_halo_mc_mean_{smhm}_err_plus"] ** 2
        cosmic_tbl["dm_halos_emp_var_minus"][to_change] += row[f"dm_halo_mc_mean_{smhm}_err_minus"] ** 2
        print("\t", name, z, dm, "+", row[f"dm_halo_mc_mean_{smhm}_err_plus"], "/ -",
              row[f"dm_halo_mc_mean_{smhm}_err_minus"])
    cosmic_tbl["dm_halos_emp_err"] = np.sqrt(cosmic_tbl["dm_halos_emp_var"])
    cosmic_tbl["dm_halos_emp_err_minus"] = np.sqrt(cosmic_tbl["dm_halos_emp_var_minus"])
    cosmic_tbl["dm_halos_emp_err_plus"] = np.sqrt(cosmic_tbl["dm_halos_emp_var_plus"])

    print(
        f'Total DM_halos = {cosmic_tbl["dm_halos_emp"][-1]} + {cosmic_tbl["dm_halos_emp_err_plus"][-1]} - {cosmic_tbl["dm_halos_emp_err_minus"][-1]}')

    print(
        f'Using DM_MW = {property_dict["dm_ism_mw_ne2001"]} + {property_dict["dm_halo_mw_pz19"]} = {property_dict["dm_mw"]}'
    )
    cosmic_tbl["dm_mw"] = np.ones(len(cosmic_tbl)) * property_dict["dm_ism_mw_ne2001"] + property_dict[
        "dm_halo_mw_pz19"]
    cosmic_tbl["dm_mw"][0] = 0 * lib.dm_units
    cosmic_tbl["dm_mw_var"] = np.ones(len(cosmic_tbl)) * (property_dict["dm_ism_mw_err"] ** 2)

    weight = lib.dm_igm_flimflam / property_dict["dm_igm_avg"]
    cosmic_tbl["dm_igm_flimflam"] = cosmic_tbl["dm_igm"] * weight
    weight_err_plus = lib.dm_igm_flimflam_err_plus / property_dict["dm_igm_avg"]
    cosmic_tbl["dm_igm_flimflam_err_plus"] = cosmic_tbl["dm_igm"] * weight_err_plus
    weight_err_minus = lib.dm_igm_flimflam_err_minus / property_dict["dm_igm_avg"]
    cosmic_tbl["dm_igm_flimflam_err_minus"] = cosmic_tbl["dm_igm"] * weight_err_minus

    cosmic_tbl["dm_all"] = cosmic_tbl["dm_igm"] + cosmic_tbl["dm_halos_emp"] + cosmic_tbl[
        "dm_mw"]  # + cosmic_tbl["dm_halo_host"]
    cosmic_tbl["dm_all"][-1] += property_dict["dm_host_ism"]
    cosmic_tbl["dm_all_flimflam"] = cosmic_tbl["dm_igm_flimflam"] + cosmic_tbl["dm_halos_emp"] + cosmic_tbl[
        "dm_mw"]  # + cosmic_tbl["dm_halo_host"]
    cosmic_tbl["dm_all_flimflam"][-1] += property_dict["dm_host_ism"]

    cosmic_tbl["dm_all_var"] = cosmic_tbl["dm_mw_var"] + cosmic_tbl["dm_halos_emp_var"] + (
                cosmic_tbl["dm_igm"] * 0.2) ** 2
    cosmic_tbl["dm_all_var"][-1] += property_dict["dm_host_ism_err"] ** 2
    cosmic_tbl["dm_all_err"] = np.sqrt(cosmic_tbl["dm_all_var"])

    cosmic_tbl["dm_all_var_plus"] = cosmic_tbl["dm_mw_var"] + cosmic_tbl["dm_halos_emp_var_plus"]
    cosmic_tbl["dm_all_var_plus"][-1] += property_dict["dm_host_ism_err"] ** 2
    cosmic_tbl["dm_all_err_plus"] = np.sqrt(cosmic_tbl["dm_all_var_plus"])
    cosmic_tbl["dm_all_var_minus"] = cosmic_tbl["dm_mw_var"] + cosmic_tbl["dm_halos_emp_var_minus"]
    cosmic_tbl["dm_all_var_minus"][-1] += property_dict["dm_host_ism_err"] ** 2
    cosmic_tbl["dm_all_err_minus"] = np.sqrt(cosmic_tbl["dm_all_var_minus"])

    cosmic_tbl["dm_all_flimflam_var"] = cosmic_tbl["dm_all_var"] + cosmic_tbl["dm_igm_flimflam_err_plus"] ** 2
    cosmic_tbl["dm_all_flimflam_err"] = np.sqrt(cosmic_tbl["dm_all_flimflam_var"])

    cosmic_tbl["dm_all_flimflam_var_plus"] = cosmic_tbl["dm_all_var_plus"] + cosmic_tbl["dm_igm_flimflam_err_plus"] ** 2
    cosmic_tbl["dm_all_flimflam_err_plus"] = np.sqrt(cosmic_tbl["dm_all_flimflam_var_plus"])
    cosmic_tbl["dm_all_flimflam_var_minus"] = cosmic_tbl["dm_all_var_minus"] + cosmic_tbl[
        "dm_igm_flimflam_err_minus"] ** 2
    cosmic_tbl["dm_all_flimflam_err_minus"] = np.sqrt(cosmic_tbl["dm_all_flimflam_var_minus"])
    return cosmic_tbl


#
# def cumulative_dm_tbl_interp(cosmic_tbl, property_dict, rmax=1.):
#     dm_halo_mw = frb.dm_mw_halo_cum(
#         rmax=rmax,
#         step_size=step_size_halo
#     )
#     cosmic_tbl["dm_halo_mw"] = np.interp(
#         cosmic_tbl["comoving_distance"],
#         dm_halo_mw["d"],
#         dm_halo_mw["DM"]
#     )
#
#     dm_ism_mw = frb.dm_mw_ism_cum(
#         max_distance=10 * units.kpc,
#         step_size=step_size_halo,
#     )
#     cosmic_tbl["dm_ism_mw_ne2001"] = np.interp(
#         cosmic_tbl["comoving_distance"],
#         dm_ism_mw["d"],
#         dm_ism_mw["DM"]
#     )
#     weight = lib.dm_igm_flimflam / cosmic_tbl["dm_igm"][-1]
#     cosmic_tbl["dm_igm_flimflam"] = cosmic_tbl["dm_igm"] * weight
#     cosmic_tbl["dm_mw"] = cosmic_tbl["dm_ism_mw_ne2001"] + cosmic_tbl["dm_halo_mw"]
#     cosmic_tbl["dm_all"] = cosmic_tbl["dm_cosmic_emp"] + cosmic_tbl["dm_mw"] + cosmic_tbl["dm_halo_host"]
#     cosmic_tbl["dm_all"][-1] += property_dict["dm_host_ism"]
#
#     cosmic_tbl["dm_all_flimflam"] = cosmic_tbl["dm_igm_flimflam"] + cosmic_tbl["dm_halos_emp"] + cosmic_tbl["dm_mw"] + cosmic_tbl["dm_halo_host"]
#     cosmic_tbl["dm_all_flimflam"][-1] += property_dict["dm_host_ism"]
#
#     return cosmic_tbl


def cumulative_dm_plot(
        cosmic_tbl,
        halo_tbl,
        filename,
        rel: str,
        legend=True,
        dm_frb=True,
        dm_total_modelled=True,
        dm_flimflam=True,
        kwargs_modelled={
            "c": "purple",
            "lw": 3
        },
        kwargs_cosmic={
            "c": "red",
            "lw": 1,
        },
        kwargs_frb={
            "c": "red",
            "ls": "--"
        },
        kwargs_flimflam={
            "c": "green",
            "lw": 8
        },
        kwargs_legend={
            # "loc": "upper center",
            "loc": "best",
            # "bbox_to_anchor": (0.5, -0.1),
            "fontsize": 11
        }
):
    if dm_flimflam:
        key = "dm_all_flimflam"
    else:
        key = "dm_all"

    fig = plt.figure(figsize=(lib.figwidth, lib.figwidth * 5 / 4))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])

    dm_modelled = cosmic_tbl[key][-1].value  # + properties["dm_host_ism"].value
    print(f"Total modelled DM: {cosmic_tbl[key][-1]} +/- {cosmic_tbl[key + '_err'][-1]}")

    # Cosmic expectation
    ax1.plot(
        cosmic_tbl["z"],
        cosmic_tbl["dm_cosmic_avg"] + cosmic_tbl["dm_mw"],
        label="DM$_\mathrm{MW} + \langle \mathrm{DM_{cosmic}} \\rangle$",
        **kwargs_cosmic
    )
    # Modelled
    ax1.plot(
        cosmic_tbl["z"],
        # cosmic_tbl["dm_halos_emp"],
        cosmic_tbl[key],
        label="Modelled DM",
        **kwargs_modelled
    )
    # Uncertainty
    bound_upper = cosmic_tbl[key] + cosmic_tbl[key + "_err_plus"]
    bound_lower = cosmic_tbl[key] - cosmic_tbl[key + "_err_minus"]
    # ax1.plot(
    #     cosmic_tbl["z"],
    #     bound_upper,
    #     c="violet"
    # )
    # ax1.plot(
    #     cosmic_tbl["z"],
    #     bound_lower,
    #     c="violet"
    # )
    ax1.fill_between(
        cosmic_tbl["z"],
        bound_lower,
        bound_upper,
        color=kwargs_modelled["c"],
        alpha=0.2,
        lw=0
    )

    # if dm_flimflam:
    #     # FLIMFLAM
    #     bound_upper = cosmic_tbl["dm_all_flimflam"] + cosmic_tbl["dm_all_flimflam_err"]
    #     bound_lower = cosmic_tbl["dm_all_flimflam"] - cosmic_tbl["dm_all_flimflam_err"]
    #     ax1.plot(
    #         cosmic_tbl["z"],
    #         cosmic_tbl["dm_all_flimflam"],
    #         label="Modelled DM (FLIMFLAM)",
    #         **kwargs_flimflam
    #     )
    #     ax1.plot(
    #         cosmic_tbl["z"],
    #         bound_upper,
    #         c="green"
    #     )
    #     ax1.plot(
    #         cosmic_tbl["z"],
    #         bound_lower,
    #         c="green"
    #     )

    if dm_total_modelled:
        # ax1.plot(
        #     (-1, 1),
        #     (dm_modelled, dm_modelled),
        #     ls=":",
        #     label="Total modelled DM",
        #     c=kwargs_modelled["c"],
        #     lw=kwargs_modelled["lw"]
        # )
        yerr = np.zeros(shape=(2, 1)) * lib.dm_units
        yerr[0, 0] = cosmic_tbl[key + '_err_minus'][-1]
        yerr[1, 0] = cosmic_tbl[key + '_err_plus'][-1]
        print(f"{yerr=}")
        ax1.errorbar(
            cosmic_tbl["z"][-1],
            cosmic_tbl[key][-1],
            yerr=yerr,
            marker="x",
            c="black",
            lw=1,
            markersize=7,
            capsize=6.,
            label="Total modelled DM",
        )
        # if dm_flimflam:
        #     ax1.plot(
        #         (0, frb.host_galaxy.z),
        #         (dm_modelled_ff, dm_modelled_ff),
        #         ls=":",
        #         label="Total modelled DM (FLIMFLAM)",
        #         c=kwargs_flimflam["c"],
        #         lw=4
        #     )

    if dm_frb:
        ax1.plot(
            (-1, 1),
            (frb.dm.value, frb.dm.value),
            label="Total measured DM$_\mathrm{FRB}$",
            **kwargs_frb
        )
    #         ax1.scatter(frb.host_galaxy.z, frb.dm.value, label="Measured DM$_\mathrm{FRB}$", **kwargs_frb)

    z_ticks = np.linspace(0, frb.host_galaxy.z, 6)
    d_ticks = Planck18.comoving_distance(z_ticks)

    delta = frb.host_galaxy.z * 1.02 - frb.host_galaxy.z
    xlim = -delta, frb.host_galaxy.z + delta
    ax1.set_xlim(xlim)
    ylim_candidates = [frb.dm.value, dm_modelled, bound_upper[-1].value]
    ax1.set_ylim(0, np.max(ylim_candidates) + 10)  # max(cosmic_tbl["dm_all"]))

    print("Upper bound:", bound_upper[-3:])

    ax1.set_xticks(z_ticks)

    ax2 = ax1.twiny()
    ax2.set_xticks(z_ticks)
    ax2.set_xticklabels(d_ticks.round(1).value)
    ax2.set_xlim(xlim)

    ax1.tick_params(axis="both", labelsize=pl.tick_fontsize)
    ax1.set_xlabel("Redshift", size=12)
    ax2.tick_params(axis="both", labelsize=pl.tick_fontsize)
    ax2.set_xlabel("Comoving distance (Mpc)", size=12)

    ax1.set_ylabel("Cumulative observer-frame DM\ \ [pc\ cm$^{-3}$]", size=12)

    if legend:
        ax1.legend(**kwargs_legend)

    ax3 = fig.add_subplot(gs[1])
    x = halo_tbl["x_"]
    z = halo_tbl["z"]
    ax3.scatter(halo_tbl["z"], x, marker="x", c="black")
    ax3.set_ylabel("$x$ [kpc]", labelpad=-1)
    ax3.set_xlabel("")
    ax3.xaxis.set_label_position('top')
    ax3.xaxis.tick_top()
    # ax3.set_ylabel("$z$")

    ax3.plot((0, 0.2365), (0, 0), c="red")

    for i, row in enumerate(halo_tbl):
        x_ = x[i].value
        z_ = z[i]
        r200 = row[f"r_200_mc_{rel}"].to("kpc").value
        if r200 > np.abs(x_):
            c = "limegreen"
            zorder = -1
        else:
            c = "blue"
            zorder = -2
        ax3.plot(
            (z_, z_),
            (x_ - r200, x_ + r200),
            c=c,
            zorder=zorder
        )
        # ax1.text(
        #     z_, x_ + 10,
        #     row["letter"],
        #     horizontalalignment="left",
        #     verticalalignment="center",
        # )

    ax3.set_xticks(z_ticks)
    ax3.set_xlim(-delta, frb.host_galaxy.z + delta)
    ax3.set_ylim(-510, 510)
    ax3.tick_params(axis="both", labelsize=pl.tick_fontsize)

    fig.tight_layout(h_pad=0.4)
    if dm_flimflam:
        filename += "_flimflam"
    lib.savefig(fig=fig, filename=filename, subdir="cumulative_dm", tight=False)
    print()

    return ax1, ax2, fig


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)
    halo_tbl = lib.read_master_table()
    properties = lib.read_master_properties()
    for rel in ["K18", "M13"]:
        print(f"Generating figure for {rel} SMHM")
        cosmic_tbl = table.QTable.read(
            os.path.join(lib.output_path, rel, "cumulative_DM_tables", "cumulative_DM_table_1.0.ecsv")
        )

        cosmic_tbl = cumulative_dm_tbl(
            cosmic_tbl=cosmic_tbl,
            property_dict=properties,
            halo_tbl=halo_tbl,
            smhm=rel
        )
        for dm_flimflam in True, False:
            if dm_flimflam:
                print(f"Using FLIMFLAM value for DM_IGM ({lib.dm_igm_flimflam})")
            else:
                print(f"Using expectation values for DM_IGM.")
            cosmic_tbl.write(os.path.join(lib.output_path, f"dm_cumulative_{rel}.ecsv"), overwrite=True)

            cumulative_dm_plot(
                cosmic_tbl=cosmic_tbl,
                filename=f"dm_cumulative_{rel}",
                dm_flimflam=dm_flimflam,
                halo_tbl=halo_tbl,
                rel=rel
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
