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
Uses figures from Bhardwaj et al 2024 as templates for CRAFT FRB hosts.
"""

latex_commands = []

wspace = 0.05
tight = True


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)
    subdir = "reproduce_bhardwaj2024"

    craft_galfit = table.QTable.read(lib.craft_galfit_path())

    n_cut = 4.
    pox_min = 0.75
    mag_max = 21. * units.mag

    # craft_galfit = craft_galfit[craft_galfit["path_pox"] >= pox_min]
    craft_galfit_n = craft_galfit[craft_galfit["galfit_n"] < n_cut]
    craft_galfit_z = craft_galfit[craft_galfit["z"] > 0]
    craft_galfit_mag = craft_galfit_z[craft_galfit_z["galfit_mag"] < mag_max]

    figsize = (0.9 * pl.textwidths["mqthesis"], 0.9 * pl.textheights["mqthesis"] * 2 / 3)

    def hist_box(suffix, tbl=craft_galfit, bins="auto", xlabelsize=10):

        plt.clf()
        fig = plt.figure(figsize=figsize)
        ncols = 2
        nrows = 3

        ax = fig.add_subplot(nrows, ncols, 1)

        counts, bins_hist, _ = ax.hist(
            tbl["galfit_inclination"].value,
            bins=bins,
            color="purple",
            linewidth=2.,
            edgecolor="black"
        )

        # if isinstance(bins, str):
        bins = u.check_quantity(bins_hist, units.deg)

        # bins *= units.deg
        # midpoints = (bins[1:] + bins[:-1]) / 2
        ticks = [int(b.value) for b in bins]
        # ticks = np.sort(np.concatenate([bins, midpoints]))
        ax.set_xticks(ticks)

        ax.set_xlabel(lib.nice_axis_label("galfit_inclination", tbl=tbl))
        ax.set_ylabel("$N$")
        ax.set_yticks(range(0, int(np.max(counts) + 1), 2))
        ax.tick_params(axis="y", labelsize=11)
        ax.tick_params(axis="x", labelsize=xlabelsize)#, labelrotation=90)
        # lib.savefig(fig=fig, filename=f"hist_inclination_bins_{suffix}", subdir=subdir)

        bin_tbl = tbl.copy()

        binned = []
        labels = []
        for i, left in enumerate(bins):
            if i < len(bins) - 1:
                j = i + 1
                right = bins[j]
                if j == len(bins) - 1:
                    lbl = f"[{int(left.value)}, {int(right.value)}]"
                    right += 1 * units.deg
                else:
                    lbl = f"[{int(left.value)}, {int(right.value)})"
                labels.append(lbl)
                bin_x = bin_tbl.copy()
                bin_x = bin_x[bin_x["galfit_inclination"] >= left]
                bin_x = bin_x[bin_x["galfit_inclination"] < right]
                binned.append(bin_x)
                print(f"Bin {lbl}:", list(bin_x["name"]))

        for i, var in enumerate(("z", "dm_excess_rest", "dm_excess")):
            ax = fig.add_subplot(nrows, ncols, i + 2)
            ax.boxplot(x=[b[var] for b in binned], labels=labels)
            ax.set_ylabel(lib.nice_axis_label(var, tbl=craft_galfit))
            ax.tick_params(axis="x", labelsize=xlabelsize, labelrotation=45)
            ax.tick_params(axis="y", labelsize=11)

        fig.tight_layout()
        lib.savefig(filename=f"bhardwaj2024_style_{suffix}", fig=fig, subdir=subdir)

    hist_box("bhardwaj_all", craft_galfit_z, [0, 42, 60, 76, 90])
    hist_box("bhardwaj_mag_cut", craft_galfit_mag, [0, 42, 60, 76, 90])
    hist_box("any_all", craft_galfit_z)  # , xlabelsize=6)
    hist_box("any_all_mag_cut", craft_galfit_mag)  # , xlabelsize=6)


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
