#!/usr/bin/env python
# Code by Lachlan Marnoch, 2025

import os

from lalinference.tiger.postproc import fontsize
from matplotlib import pyplot as plt
import numpy as np

from astropy import table, units

import craftutils.utils as u
import craftutils.plotting as pl

import lib

description = """
Generates the figures for the Validation section.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    pl.latex_setup()

    validation_catalogue = table.QTable.read(os.path.join(lib.input_path, "validation_catalogue_generated.ecsv"))
    fil = "g_HIGH"
    tolerance = 0.5 * units.arcsec

    ticksize=12
    labelsize=13

    axes = []

    fig_comp = plt.figure(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] * 0.4))

    for i, mag_type in enumerate(("PSF", "AUTO")):
        cat = validation_catalogue[validation_catalogue[f"{fil}_mag_{mag_type}"] < 100 * units.mag]
        cat = cat[cat[f"{fil}_mag_{mag_type}"] > -100 * units.mag]
        cat = cat[cat[f"separation_{fil}"] < tolerance]
        x = cat[f"mag"].value
        y = cat[f"{fil}_mag_{mag_type}"].value
        y_err = cat[f"{fil}_mag_{mag_type}_err"].value

        if i > 0:
            sy = axes[0]
        else:
            sy = None
        ax = fig_comp.add_subplot(1, 2, i + 1, sharey=sy)

        ax.errorbar(x, y, yerr=y_err, c="black", ls="none")
        ax.set_xlim(17.5, 26)
        ax.set_ylim(17.5, 26)
        print(np.min(x), np.max(x))
        ax.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], c="red")
        c = ax.scatter(x, y, marker="x", c=cat[f"separation_{fil}"].value, s=10, zorder=10)
        ax.set_xlabel("Inserted object (mag)", fontsize=labelsize)
        ax.set_xticks([19, 21, 23, 25])
        ax.tick_params(axis="x", labelsize=ticksize)
        ax.grid(False)
        if i == 0:
            ax.set_ylabel("Extracted object (mag)", fontsize=labelsize)
            ax.tick_params(axis="y", labelsize=ticksize)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(f"{mag_type} mag")
        axes.append(ax)

        fig, ax = plt.subplots()
        ax.hist(cat[f"{fil}_mag_{mag_type}_delta"], bins="auto")
        ax.set_xlabel(r"$m_\mathrm{inserted} - m_\mathrm{extracted}$")
        fig.savefig(os.path.join(lib.output_path, f"delta-m_{mag_type}.pdf"))
        fig.clear()
        plt.close(fig)

    # cax = fig_comp.add_subplot(1, 3, 3)
    cbar = fig_comp.colorbar(c, ax=axes, location="right")
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label(r"Separation ($\prime\prime$)", fontsize=labelsize)
    # fig_comp.subplots_adjust(hspace=1)
    fig_comp.savefig(os.path.join(lib.output_path, f"validation_comparison.pdf"))
    fig_comp.clear()
    plt.close(fig_comp)


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
