#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np
import matplotlib.pyplot as plt

from astropy import units, table
from astropy.visualization import (
    ImageNormalize,
    LogStretch
)

import craftutils.utils as u
import craftutils.params as p
from craftutils.observation import objects
from craftutils.plotting import latex_setup, textwidths, textheights

import lib

description = """
Uses the framework of Cordes et al 2022 to estimate the host dispersion
measure contribution from the scattering timescale.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)
    fld = lib.fld
    fld.frb._dm_mw_ism = 30.9 * objects.dm_units

    # Set up a range of values for redshift and AFG
    z_host = np.arange(0., 2., 0.01)
    afg = np.logspace(-2, 1, 100) * lib.afg_units
    dm_arrays = []
    for AFG_val in afg:
        dm = fld.frb.dm_host_from_tau(
            z_host=z_host,
            afg=AFG_val,
        )
        dm_arrays.append(dm)

    values = {}
    values["tau_mw"] = fld.frb.tau_mw()
    values["tau_exgal"] = fld.frb.tau - values["tau_mw"]
    values["DM_host_0.1_max_obs"] = (fld.frb.dm_host_from_tau(
        z_host=z_host,
        afg=0.1 * lib.afg_units,
    ) / (1 + z_host)).max()
    dm_arrays = np.array(dm_arrays)

    labels = [
        "Host frame",
        "Observer frame"
    ]

    output = os.path.join(lib.output_path, "scattering")
    u.mkdir_check(output)

    latex_setup()

    cordes_2022_tbl3 = table.Table.read(os.path.join(lib.input_path, "cordes+2022", "cordes+2022_table-3.ecsv"))

    fig = plt.figure(figsize=(textwidths["mqthesis"], textwidths["mqthesis"] * 0.5))
    for i, dm_array in enumerate((dm_arrays, dm_arrays / (1 + z_host))):
        ax = fig.add_subplot(1, 2, i + 1)
        c = ax.pcolor(
            z_host,
            afg,
            dm_array,
            cmap='viridis',
            alpha=1,
            norm=ImageNormalize(
                data=dm_array,
                stretch=LogStretch()
            )
        )
        cbar = fig.colorbar(
            c,
            location="bottom",
            label="$\mathrm{DM_{host}}$ (pc cm$^{-3}$)",
            pad=0.2,
            ticks=np.logspace(
                np.log10(np.nanmin(dm_array[np.isfinite(dm_array)])),
                np.log10(np.floor(np.nanmax(dm_array[np.isfinite(dm_array)]))),
                7
            ).round(),
        )
        plt.scatter(
            cordes_2022_tbl3["z_h"][~cordes_2022_tbl3["AFG_limit"]],
            cordes_2022_tbl3["AFG"][~cordes_2022_tbl3["AFG_limit"]],
            marker="x",
            c="black"
        )
        plt.scatter(
            cordes_2022_tbl3["z_h"][cordes_2022_tbl3["AFG_limit"]],
            cordes_2022_tbl3["AFG"][cordes_2022_tbl3["AFG_limit"]],
            marker="v",
            c="black"
        )
        # cbar.set_label(
        # )
        ax.set_yscale("log")
        if i == 0:
            ax.set_ylabel("$A_\\tau \widetilde{F} G$")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("$z$")
        ax.set_title(labels[i])
    fig.subplots_adjust(hspace=0.2)
    fig.savefig(os.path.join(output, f"dm_scattering.png"), bbox_inches="tight")
    fig.savefig(os.path.join(output, f"dm_scattering.pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"tau_mw ==", values["tau_mw"])
    print(f"tau_exgal ==", values["tau_exgal"])
    print(f"Max DM_host for AFG=0.1:", values['DM_host_0.1_max_obs'])
    p.save_params(os.path.join(output, "values.yaml"), values)


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
