#!/usr/bin/env python
# Code by Lachlan Marnoch, 2025

import os

import numpy as np
import matplotlib.pyplot as plt
from ligo.skymap.tool.matplotlib import figwidth

import craftutils.utils as u

import lib

from frb.halos.hmf import halo_incidence

description = """
Does some calculations of FRB probabilities of intersecting halos at mass thresholds.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    halo_tbl = lib.read_master_table()

    frb = lib.fld.frb

    m_highs = []
    m_lows = []
    int_partitions = []
    int_greater = []

    for rel in ("K18", "M13"):
        for row in halo_tbl:
            print("\t\tCalculating intersection probabilities.")
            log_mass_halo = row[f"log_mass_halo_mc_{rel}"]
            m_low = 10 ** (np.floor(log_mass_halo))
            m_high = 10 ** (np.ceil(log_mass_halo))
            if m_low < 2e10:
                m_high += 2e10 - m_low
                m_low = 2e10

            # halo_info["mass_halo_partition_high"] = m_high * units.solMass
            # halo_info["mass_halo_partition_low"] = m_low * units.solMass

            m_highs.append(np.log10(m_high))
            m_lows.append(np.log10(m_low))

            int_partitions.append(halo_incidence(
                Mlow=m_low,
                Mhigh=m_high,
                zFRB=row["z"],
                radius=row["r_perp_mc_K18"]
            ))
            int_greater.append(halo_incidence(
                Mlow=10 ** log_mass_halo,
                zFRB=row["z"],
                radius=row["r_perp_mc_K18"]
            ))

    halo_tbl["log_mass_halo_partition_high"] = m_highs
    halo_tbl["log_mass_halo_partition_low"] = m_lows

    halo_tbl["n_intersect_partition"] = int_partitions
    halo_tbl["n_intersect_greater"] = int_greater

    fga = halo_tbl[1]

    n_int = []
    masses = np.logspace(10.4, 16, 50)
    for i, j in masses:
        print(f"{i} M_sol (10e{np.log10(i)}), {j} / {len(masses)}")
        n_int.append(
            halo_incidence(
                Mlow=i,  # fg_row["mass_halo"].value,
                Mhigh=i + 1,
                zFRB=frb.host_galaxy.z,
                radius=fga["r_perp"]
            )
        )

    log_masses = np.log10(masses)
    fig = plt.figure(figsize=(lib.figwidth, lib.figwidth * 2 / 3))
    ax = fig.add_subplot()
    ax.plot(log_masses, n_int)
    # ax.xscale("log")
    ax.set_xlabel(r"$\log{(\frac{M_\mathrm{halo}}{\mathrm{M_\odot}})}$")
    ax.set_ylabel("Probability of intersection")
    lib.savefig(fig, "incidence_probability", tight=False)


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
