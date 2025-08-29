#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from astropy import units, table
from astropy.coordinates import SkyCoord
from astropy.cosmology import z_at_value, Planck18

import craftutils.utils as u
import craftutils.astrometry as a
import craftutils.plotting as pl

import lib

description = """
Generates miscellaneous figures.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb = lib.fld.frb
    halos = lib.read_master_table()

    s23 = table.QTable.read(os.path.join(lib.input_path, "Simha+2023", "object_catalogs", "FRB20190714A_fg_stellar_mass.csv"))
    s23["ra"] *= units.deg
    s23["dec"] *= units.deg

    matches, matches_s23, distance = a.match_catalogs(
        halos, s23,
        ra_col_1="ra", dec_col_1="dec",
        tolerance=3 * units.arcsec,
    )
    matches_both = table.hstack([matches, matches_s23], table_names=["", "S23"])
    matches_both.write(os.path.join(output_dir, "join_S23.csv"), overwrite=True)

    # ================================================================
    # Circle plot

    x_lim = (-510 * units.kpc, 510 * units.kpc)

    # halos["x_cart"] = halos["coord"].

    frb_pos = frb.position
    senses = []
    for row in halos:
        if row["ra"] < frb_pos.ra:
            sense = 1
        else:
            sense = -1
        senses.append(sense)

    x = halos["x_"] = halos["r_perp_mc_K18"] * senses
    y = halos["y_"] = halos["distance_comoving"].to("Mpc")
    y_frb = halos[-1]["y_"]
    x_frb = 0 * units.kpc

    fig = plt.figure(figsize=(lib.figwidth, lib.figwidth * 1.3))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot((x_frb.value, x_frb.value), (y_frb.value, 0), c="black")

    ax1.scatter(x, y, marker="x", c="violet")
    d_ticks = np.linspace(10e-5 * units.Mpc, y_frb, 6)
    z_ticks = z_at_value(func=Planck18.comoving_distance, fval=d_ticks)

    # ax1.set_yticks(d_ticks)
    ax1.set_xlabel("$x$ (kpc)")
    ax1.set_ylabel("Comoving distance (Mpc)")
    # ax1.set_aspect("equal")
    ax1.set_ylim(0 * units.Mpc, 1020 * units.Mpc)

    for i, row in enumerate(halos):
        x_ = x[i]
        y_ = y[i]
        r200 = row["r_200_mc_K18"]
        if r200 > np.abs(x_):
            c = "limegreen"
            zorder = -1
        else:
            c = "blue"
            zorder = -2
        ax1.plot(
            (x_.value - r200.to("kpc").value, x_.value + r200.to("kpc").value),
            (y_.value, y_.value),
            c=c,
            zorder=zorder
        )
        # e = Ellipse(
        #     xy=(row["x_"], row["y_"]),
        #     width=2 * r200.value,
        #     height=2 * r200.to("Mpc").value,
        #     angle=0.,
        #     color="green",
        # )
        # ax1.add_patch(e)
        ax1.text(
            x_, y_ + 5 * units.Mpc,
            row["letter"],
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    ax1.set_yticks(d_ticks.value.round())
    ax2 = ax1.twinx()
    ax2.set_yticks(d_ticks.value.round())
    # z_ticks[0] = 0.
    z_ticks = np.round(z_ticks, decimals=4).value

    ax2.set_yticklabels(z_ticks)
    ax2.set_ylim(0 * units.Mpc, 1000 * units.Mpc)
    ax2.set_xlim(x_lim)
    ax2.set_ylabel("Redshift", rotation=-90)
    ax1.tick_params(axis="both", labelsize=pl.tick_fontsize)
    ax2.tick_params(axis="both", labelsize=pl.tick_fontsize)
    plt.tight_layout(w_pad=10)

    figname = "sightline"
    lib.savefig(fig, figname, tight=False)

    # fig = plt.figure(figsize=(10, 100))
    # ax1 = fig.add_subplot(1, 1, 1)
    # ax1.plot((x_frb.value, x_frb.value), (y_frb.to(10 * units.kpc).value, 0), c="red")
    # ax1.scatter(halos["x_"], halos["y_"].to(10 * units.kpc), marker="x")
    # ax1.set_xlabel("x (kpc)")
    # ax1.set_ylabel("Comoving distance (100 kpc)")
    # ax1.set_aspect("equal")
    # lib.savefig(fig, "sightline_scale", tight=False)

    halos["x_"] = halos["r_perp_mc_K18"] * senses
    halos["y_"] = halos["distance_comoving"].to("Mpc")
    y_frb = halos[0]["y_"]
    x_frb = 0 * units.kpc

    # Redshift histogram

    fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth * 2 / 3))
    ax.hist(halos["z"], bins="auto")

    lib.savefig(fig=fig, filename="hist_z", tight=True)

    # ================================================================
    # Completeness curve

    halos.sort("offset_angle")
    fig = plt.figure(figsize=(lib.figwidth, lib.figwidth * 2 / 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.step(
        halos["offset_angle"],
        np.cumsum(halos["dm_halo_mc_K18"]),
        label="K+18",
        lw=2,
        c="purple",
    )
    ax.step(
        halos["offset_angle"],
        np.cumsum(halos["dm_halo_mc_M13"]),
        label="M+13",
        lw=2,
        c="green",
    )
    ax.legend()
    ax.tick_params(axis="both", labelsize=pl.tick_fontsize)
    ax.set_xlabel(f"Radius of inclusion ({units.arcsec.to_string('latex')})")
    ax.set_ylabel("$\mathrm{DM_{halos}}$")
    lib.savefig(fig=fig, filename="completeness", tight=True)

    lib.write_master_table(halos)


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
