#!/usr/bin/env python
# Code by Lachlan Marnoch, 2024

import os
import shutil

import craftutils.utils as u

import lib

description = """
Generates DM-z figures from tables of CRAFT and other FRBs.
"""

import matplotlib.pyplot as plt

import craftutils.observation.objects as objects
import craftutils.observation.field as field
import craftutils.observation.image as image
import craftutils.observation.epoch as epoch
import craftutils.plotting as pl
import craftutils.utils as u

import astropy.units as units
import frb.dm.igm as igm
from astropy.cosmology import Planck18
from astropy import table
from astropy.modeling import models, fitting
from astropy.coordinates import SkyCoord

import lib

import numpy as np

pl.latex_setup()

dm_units = units.pc / units.cm ** 3

dm_cosmic, z_cosmic = igm.average_DM(z=1.1, cosmo=Planck18, cumul=True)
cosmic_dict = {
    "dm": dm_cosmic,
    "dm_exgal": dm_cosmic,
    "dm_cosmic_nominal": dm_cosmic,
    "dm_excess": np.zeros(dm_cosmic.shape),
    "dm_excess_rest": np.zeros(dm_cosmic.shape),
    "dm_residual": np.zeros(dm_cosmic.shape)
}
dm_halos, z_halos = igm.average_DMhalos(z=1.1, cosmo=Planck18, cumul=True)
halos_dict = {
    "dm_exgal": dm_halos,
    "dm_cosmic_nominal": dm_halos,
    "dm_excess": dm_halos - dm_cosmic,
    "dm_residual": dm_halos - dm_cosmic
}

dm_igm = dm_cosmic - dm_halos
igm_dict = {
    "dm_exgal": dm_igm,
    "dm_cosmic_nominal": dm_igm,
    "dm_excess": dm_igm - dm_cosmic,
    "dm_residual": dm_igm - dm_cosmic
}

dm_cliff = 500 * z_cosmic * dm_units
cliff_dict = {
    "dm_exgal": dm_cliff,
    "dm_cosmic_nominal": dm_cliff,
    "dm_excess": dm_cliff - dm_cosmic,
    "dm_residual": dm_cliff - dm_cosmic
}


def main(
        output_dir: str,
        input_dir: str,
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb_table_big = lib.load_frb_table()

    frb_table_z = frb_table_big[frb_table_big["z"] > 0]

    plot_dir = os.path.join(output_dir, "dm-z_plots")
    os.makedirs(plot_dir, exist_ok=True)

    def macquart_plot(
            frb_table: table.QTable,
            dm_colname: str,
            y_name: str = None,
            suptitle: str = None,
            draw_names: bool = False,
            askap_only: bool = False,
            draw_cosmic: bool = True,
            draw_igm: bool = True,
            draw_halos: bool = False,
            draw_fitted: bool = False,
            draw_cliff: bool = False,
            draw_histograms: bool = False,
            legend_loc='lower right',
            separate_repeaters: bool = True,
            name: str = None,
            all_teams: bool = False
    ):

        hist_props = {
            "edgecolor": "black",
            "lw": 2,
            "alpha": 1,
        }
        scatter_props = {
            "marker": "o",
            "facecolor": "none",
            # "edgecolor": "black",
            # "lw": 2,
            "alpha": 1,
        }
        scatter_repeat = {
            "marker": "s",
            "facecolor": "none",
            # "edgecolor": "black",
            # "lw": 2,
            "alpha": 1,
        }

        all_hist_color = "darkorange"

        craft = frb_table[frb_table["team"] == "CRAFT"]
        noncraft = frb_table[frb_table["team"] != "CRAFT"]
        stub = dm_colname.replace("dm", "")
        if askap_only:
            sample = craft
        else:
            sample = frb_table

        fig = plt.figure(figsize=(pl.textheights["mqthesis"], pl.textwidths["mqthesis"]))
        gs = fig.add_gridspec(
            nrows=2, ncols=2,
            width_ratios=[2, 1], height_ratios=[1, 2]
        )

        other = {
            "DSA": "cyan",
            "CHIME/FRB": "red",
            "MeerTRAP": "violet"
        }
        misc = noncraft[[not lib.isreal(n) or n not in other for n in noncraft["team"]]]

        if draw_histograms:
            ax_dm_hist = fig.add_subplot(gs[1, 1])
            if not askap_only:
                n, bins, _ = ax_dm_hist.hist(
                    sample[dm_colname],
                    orientation="horizontal",
                    label="All hosts",
                    bins="auto",
                    color=all_hist_color,
                    **hist_props
                )
            else:
                bins = "auto"
            ax_dm_hist.hist(
                craft[dm_colname],
                bins=bins,
                orientation="horizontal",
                label="CRAFT",
                color="purple",
                **hist_props
            )
            ax_dm_hist.tick_params(axis="y", direction="in")
            ax_dm_hist.set_xlabel("N")
            ax_dm_hist.tick_params(labelsize=10)
            fig.subplots_adjust(wspace=0.)

            ax_z_hist = fig.add_subplot(gs[0, 0])
            if not askap_only:
                n, bins, _ = ax_z_hist.hist(
                    sample["z"],
                    bins="auto",
                    # label="Full host sample",
                    color=all_hist_color,
                    **hist_props
                )
            else:
                bins = "auto"
            ax_z_hist.hist(
                craft["z"],
                bins=bins,
                # label="ASKAP FRB hosts",
                color="purple",
                **hist_props
            )

            ax_z_hist.tick_params(axis="x", direction="in")
            ax_z_hist.set_ylabel("N")
            ax_z_hist.tick_params(labelsize=10)
            fig.subplots_adjust(hspace=0.)
            # ax_z_hist.legend(
            #     loc=(1.005, .68),
            #     fontsize=12
            # )
            xlim = ax_z_hist.get_xlim()

            ax_1 = fig.add_subplot(gs[1, 0])
            ax_1.set_xlim(xlim)
        else:
            ax_1 = fig.add_subplot(111)

        if suptitle is not None:
            fig.suptitle(suptitle)
        if not askap_only:
            if all_teams:
                tbl_rep = misc[misc["repeater"]]
                tbl_non = misc[[not r for r in misc["repeater"]]]
                ax_1.scatter(
                    tbl_non["z"],
                    tbl_non[dm_colname],
                    edgecolors="green",
                    label="Other",
                    **scatter_props
                )
                ax_1.scatter(
                    tbl_rep["z"],
                    tbl_rep[dm_colname],
                    edgecolors="green",
                    # label="Other",
                    **scatter_repeat
                )
                for team, c in other.items():
                    team_hosts = frb_table[frb_table["team"] == team]
                    team_hosts_rep = team_hosts[team_hosts["repeater"]]
                    team_hosts_non = team_hosts[[not r for r in team_hosts["repeater"]]]
                    ax_1.scatter(
                        team_hosts_non["z"],
                        team_hosts_non[dm_colname],
                        edgecolors=c,
                        label=team,
                        **scatter_props
                    )
                    ax_1.scatter(
                        team_hosts_rep["z"],
                        team_hosts_rep[dm_colname],
                        edgecolors=c,
                        # label=team,
                        **scatter_repeat
                    )
            else:
                tbl_rep = noncraft[noncraft["repeater"]]
                tbl_non = noncraft[[not r for r in noncraft["repeater"]]]
                ax_1.scatter(
                    tbl_non["z"],
                    tbl_non[dm_colname],
                    edgecolors="green",
                    label="Non-CRAFT",
                    **scatter_props
                )
                ax_1.scatter(
                    tbl_rep["z"],
                    tbl_rep[dm_colname],
                    edgecolors="green",
                    # label="Non-ASKAP FRB hosts",
                    **scatter_repeat
                )
            if draw_names:
                for row in noncraft:
                    plt.text(row["z"], row[dm_colname], row["name"], alpha=0.5)

        craft_rep = craft[craft["repeater"]]
        craft_non = craft[[not r for r in craft["repeater"]]]

        ax_1.scatter(
            craft_non["z"],
            craft_non[dm_colname],
            edgecolors="purple",
            label="CRAFT",
            **scatter_props
        )
        ax_1.scatter(
            craft_rep["z"],
            craft_rep[dm_colname],
            edgecolors="purple",
            # label="CRAFT",
            **scatter_repeat
        )

        if draw_names:
            for row in frb_table[frb_table["team"] == "CRAFT"]:
                plt.text(row["z"], row[dm_colname], row["name"], alpha=0.5)

        if draw_cosmic and dm_colname in cosmic_dict:
            ax_1.plot(
                z_cosmic,
                cosmic_dict[dm_colname],
                label="$\mathrm{<DM_{cosmic}>}$",
                c="black"
            )
        if draw_halos and dm_colname in halos_dict:
            ax_1.plot(
                z_halos,
                halos_dict[dm_colname],
                label="$\mathrm{<DM_{halos}>}$",
                c="violet"
            )
        if draw_igm and dm_colname in igm_dict:
            ax_1.plot(
                z_halos,
                cliff_dict[dm_colname],
                label="$\mathrm{<DM_{igm}>}$",
                c="red"
            )
        if draw_cliff and dm_colname in cliff_dict:
            ax_1.plot(
                z_halos,
                igm_dict[dm_colname],
                label="$\mathrm{<DM_{cliff}>}$",
                c="cyan"
            )
        if draw_fitted:
            line = models.Linear1D(slope=900)
            fitter = fitting.LinearLSQFitter()
            fitted = fitter(line, sample["z"], sample[dm_colname].value)
            sample["dm_fitted" + stub] = fitted(sample["z"])
            dm_fitted = fitted(z_cosmic)
            print(fitted)
            ax_1.plot(z_cosmic, dm_fitted, label="Fitted DM-z relation", c="black")

        # scatter_repeat.update({"alpha": 0})
        ax_1.scatter([], [], edgecolors="black", label="Non-repeaters", **scatter_props)
        ax_1.scatter([], [], edgecolors="black", label="Repeaters", **scatter_repeat)
        # ax_1.legend(
        #     loc=(1.005, 1.005),  # legend_loc,
        #     fontsize=12,
        #     ncol=2
        # )
        fig.legend(
            loc=(0.695, 0.67),#"upper right",
            fontsize=12,
            # ncol=2
        )

        if y_name is None:
            y_name = lib.nice_axis_label(colname=dm_colname, tbl=frb_table)
        ax_1.set_ylabel(y_name)
        ax_1.set_xlabel(f"Redshift")
        ax_1.tick_params(labelsize=10)

        if name is None:
            name = f"{dm_colname}-z"
        if draw_histograms:
            name += "_hist"
        if draw_cosmic:
            name += "_cosmic"
        if all_teams:
            name += "_all-teams"
        fig.savefig(os.path.join(plot_dir, name) + ".png", bbox_inches="tight", dpi=200)
        pdf_path = os.path.join(plot_dir, name + ".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        db_path = os.path.join(lib.dropbox_path, "figures", "dm-z")
        os.makedirs(db_path, exist_ok=True)
        if os.path.isdir(db_path):
            shutil.copy(pdf_path, db_path)
        return ax_1, fig

    # Bog-standard Total DM-z
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm",
        # y_name="$\mathrm{DM_{FRB}}$",
        draw_names=False,
        askap_only=False,
        draw_cosmic=False,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False,
        draw_histograms=True,
        all_teams=False
    )
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm",
        # y_name="$\mathrm{DM_{FRB}}$",
        draw_names=False,
        askap_only=False,
        draw_cosmic=False,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False,
        draw_histograms=True,
        all_teams=True
    )
    # Ditto, without histograms
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm",
        # y_name="$\mathrm{DM_{FRB}}$",
        draw_names=False,
        askap_only=False,
        draw_cosmic=False,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False
    )
    # Ditto, with DM_cosmic expectation
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm",
        # y_name="$\mathrm{DM_{FRB}}$",
        draw_names=False,
        askap_only=False,
        draw_cosmic=True,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False,
        draw_histograms=True
    )
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm",
        # y_name="$\mathrm{DM_{FRB}}$",
        draw_names=False,
        askap_only=False,
        draw_cosmic=True,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False,
        draw_histograms=True,
        all_teams=True
    )

    # Extragalactic DM
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm_exgal",
        # y_name="$\mathrm{DM_{exgal}}$",
        draw_names=False,
        askap_only=False,
        draw_cosmic=True,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False,
        draw_histograms=True,
        all_teams=False
    )
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm_exgal",
        # y_name="$\mathrm{DM_{exgal}}$",
        draw_names=False,
        askap_only=False,
        draw_cosmic=True,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False,
        draw_histograms=True,
        all_teams=True
    )
    # DM_ISM
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm_ism_ne2001",
        # y_name="$\mathrm{DM_{MW,ISM,NE2001}}$",
        legend_loc="upper right",
        draw_names=False,
        askap_only=False,
        draw_cosmic=True,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False,
        draw_histograms=True
    )
    # DM excess
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm_excess",
        # suptitle="Excess DM",
        # y_name="$\mathrm{DM_{excess}}$",
        draw_names=False,
        askap_only=False,
        draw_cosmic=True,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False,
        draw_cliff=False,
        draw_histograms=True,
        legend_loc="upper right",
        all_teams=False
    )
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm_excess",
        # suptitle="Excess DM",
        # y_name="$\mathrm{DM_{excess}}$",
        draw_names=False,
        askap_only=False,
        draw_cosmic=True,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False,
        draw_cliff=False,
        draw_histograms=True,
        legend_loc="upper right",
        all_teams=True
    )
    # DM excess, rest
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm_excess_rest",
        # suptitle="Excess DM (host rest frame)",
        # y_name=r"$\mathrm{DM\prime_{excess}}(1+z_\mathrm{host})$",
        draw_names=False,
        askap_only=False,
        draw_cosmic=True,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False,
        draw_cliff=False,
        draw_histograms=True,
        legend_loc=(1.005, 1.1)  # "upper right",
    )
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm_excess_rest",
        # suptitle="Excess DM (host rest frame)",
        # y_name=r"$\mathrm{DM_{excess}}(1+z_\mathrm{host})$",
        draw_names=False,
        askap_only=False,
        draw_cosmic=True,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False,
        draw_cliff=False,
        draw_histograms=True,
        legend_loc=(1.005, 1.1),  # "upper right",
        all_teams=True
    )
    # DM residuals
    macquart_plot(
        frb_table=frb_table_z,
        dm_colname="dm_residual",
        # suptitle="DM Residuals",
        y_name="Residuals for $\mathrm{DM_{FRB}}$",
        legend_loc="upper right",
        draw_names=False,
        askap_only=False,
        draw_cosmic=False,
        draw_igm=False,
        draw_halos=False,
        draw_fitted=False,
        draw_cliff=False,
        draw_histograms=True
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
