#!/usr/bin/env python
# Code by Lachlan Marnoch, 2025

import os

import matplotlib.pyplot as plt

# import astropy.
import astropy.units as units
import astropy.table as table
from astropy.visualization import ImageNormalize, LogStretch
from mpl_toolkits.axes_grid1 import Grid, ImageGrid, AxesGrid

import craftutils.utils as u
import craftutils.observation.field as field
import craftutils.params as p
import numpy as np

import lib

description = """
Analyses the fhot-Rmax grid models.
"""


def main(
        output_dir: str,
        input_dir: str,
        rel: str,
        # n_real: float
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb190714 = lib.fld

    # rel = "K18"
    rmaxes = lib.rmaxes
    fs_hot = lib.fs_hot

    index_f75 = np.argmin(np.abs(fs_hot - 0.75))
    index_r1 = np.argmin(np.abs(rmaxes - 1.))
    index_r2 = np.argmin(np.abs(rmaxes - 2.))
    index_r3 = np.argmin(np.abs(rmaxes - 3.))

    print("Closest")
    print(f"fhot: {fs_hot[index_f75]}, {rmaxes[index_r1]}, {rmaxes[index_r2]}, {rmaxes[index_r3]}")

    commands_dict = {}

    print()

    absmin = np.inf

    for rel in ("K18", "M13"):

        rel_ = {"M13": "Moster", "K18": "Kravtsov"}[rel]
        print(f"{rel} ({rel_})")

        dms_halos = np.load(lib.constraints_npy_path(relationship=rel)) * lib.dm_units

        props_fiducial = lib.read_properties(rmax=1., fhot=0.75, relationship=rel)
        # dm_exgal = props_fiducial["dm_exgal"]
        dm_igm_avg = props_fiducial["dm_igm"]
        dm_mw_ism = props_fiducial["dm_ism_mw_ne2001"]

        for n, dm_igm in {
            "avg": dm_igm_avg,
            "flimflam": lib.dm_igm_flimflam
        }.items():
            print("=" * 50)
            print(f"{rel}, {n}")

            dm_budget = frb190714.frb.dm - dm_mw_ism - dm_igm - lib.dm_host_ism_lb

            # Colour with DM_halos

            fig, ax = plt.subplots()

            c = ax.pcolor(
                rmaxes,
                fs_hot,
                dms_halos.value,
                cmap='cmr.cosmic_r',
                alpha=1,
                norm=ImageNormalize(
                    data=dms_halos.value,
                    # stretch=LogStretch(),
                    vmax=dm_budget.value,
                    vmin=np.min(dms_halos.value),
                )
            )
            ax.set_xlim(np.min(rmaxes), np.max(rmaxes))
            ax.set_ylim(np.min(fs_hot), np.max(fs_hot))
            ax.set_xlabel(r"$R_\mathrm{max} / R_\mathrm{200}$")
            ax.set_ylabel(r"$f_\mathrm{hot}$")
            ax.tick_params(axis='both', labelsize=12)

            cbar = fig.colorbar(c)
            cbar.set_label(
                "$\mathrm{DM_{halos}}$ (pc\,cm$^{-3}$)",
                # labelpad=-4
            )
            lib.savefig(fig, f"Rmax-fhot_{n}_{rel}", subdir="rmax-fhot", png_to_db=True)

            # Colour with DM_halos, but with max at DM_FRB

            fig, ax = plt.subplots()

            c = ax.pcolor(
                rmaxes,
                fs_hot,
                dms_halos.value,
                cmap='cmr.cosmic_r',
                alpha=1,
                norm=ImageNormalize(
                    data=dms_halos.value,
                    # stretch=LogStretch(),
                    vmax=lib.fld.frb.dm.value,
                    vmin=np.min(dms_halos.value),
                )
            )
            ax.set_xlim(np.min(rmaxes), np.max(rmaxes))
            ax.set_ylim(np.min(fs_hot), np.max(fs_hot))
            ax.set_xlabel(r"$R_\mathrm{max} / R_\mathrm{200}$")
            ax.set_ylabel(r"$f_\mathrm{hot}$")
            ax.tick_params(axis='both', labelsize=12)

            cbar = fig.colorbar(c)
            cbar.set_label(
                "$\mathrm{DM_{halos}}$ (pc\,cm$^{-3}$)",
                # labelpad=-4
            )
            lib.savefig(fig, f"Rmax-fhot_maxfrb_{n}_{rel}", subdir="rmax-fhot", png_to_db=True)

            # Colour with DM_total

            dms_all = dm_mw_ism + dm_igm + lib.dm_host_ism_lb + dms_halos

            fig, ax = plt.subplots()

            c = ax.pcolor(
                rmaxes,
                fs_hot,
                dms_all.value,
                cmap='cmr.cosmic_r',
                alpha=1,
                norm=ImageNormalize(
                    data=dms_all.value,
                    # stretch=LogStretch(),
                    vmax=lib.fld.frb.dm.value,
                    vmin=np.min(dms_all.value),
                )
            )
            absmin = np.min((absmin, np.min(dms_all.value)))
            ax.set_xlim(np.min(rmaxes), np.max(rmaxes))
            ax.set_ylim(np.min(fs_hot), np.max(fs_hot))
            ax.set_xlabel(r"$R_\mathrm{max} / R_\mathrm{200}$")
            ax.set_ylabel(r"$f_\mathrm{hot}$")
            ax.tick_params(axis='both', labelsize=12)

            cbar = fig.colorbar(c)
            cbar.set_label(
                "$\mathrm{DM_\mathrm{modelled}}$ (pc\,cm$^{-3}$)",
                # labelpad=-4
            )
            lib.savefig(fig, f"Rmax-fhot_total_{n}_{rel}", subdir="rmax-fhot", png_to_db=True)

            # No black

            fig, ax = plt.subplots()

            c = ax.pcolor(
                rmaxes,
                fs_hot,
                dms_all.value,
                cmap='cmr.cosmic_r',
                alpha=1,
                # norm=ImageNormalize(
                #     data=dms_all.value,
                #     # stretch=LogStretch(),
                #     vmax=lib.fld.frb.dm.value,
                #     vmin=np.min(dms_all.value),
                # )
            )
            ax.set_xlim(np.min(rmaxes), np.max(rmaxes))
            ax.set_ylim(np.min(fs_hot), np.max(fs_hot))
            ax.set_xlabel(r"$R_\mathrm{max} / R_\mathrm{200}$")
            ax.set_ylabel(r"$f_\mathrm{hot}$")
            ax.tick_params(axis='both', labelsize=12)

            cbar = fig.colorbar(c)
            cbar.set_label(
                "$\mathrm{DM_\mathrm{modelled}}$ (pc\,cm$^{-3}$)",
                # labelpad=-4
            )
            lib.savefig(fig, f"Rmax-fhot_noblack_{n}_{rel}", subdir="rmax-fhot", png_to_db=True)

            ## Numbers

            slice_75 = dms_halos[index_f75, :]
            max_rmax_avg = rmaxes[np.max(np.argwhere(slice_75 < dm_budget))]
            commands_dict[f"RmaxMax{n}{rel_}"] = np.round(max_rmax_avg, 2)

            max_fhot_avg_1 = fs_hot[np.max(np.argwhere(dms_halos[:, index_r1] < dm_budget))]
            max_fhot_avg_2 = fs_hot[np.max(np.argwhere(dms_halos[:, index_r2] < dm_budget))]
            max_fhot_avg_3 = fs_hot[np.max(np.argwhere(dms_halos[:, index_r3] < dm_budget))]
            commands_dict[f"FHotMaxOne{n}{rel_}"] = np.round(max_fhot_avg_1, 2)
            commands_dict[f"FHotMaxTwo{n}{rel_}"] = np.round(max_fhot_avg_2, 2)
            commands_dict[f"FHotMaxThree{n}{rel_}"] = np.round(max_fhot_avg_3, 2)

            print()
            print(f"{dm_igm=}")
            print(f"{dm_budget=}")
            print(f"{np.max(dms_all)=}")
            print(f"{np.min(dms_all)=}")
            print(f"{np.max(dms_halos)=}")
            print(f"{np.min(dms_halos)=}")
            print()

        # for command, value in commands_dict.items():
        #     print(f"{command}: {value}")

    fig = plt.figure(
        figsize=(lib.figwidth, lib.figwidth * 0.5)
    )

    grid = AxesGrid(
        fig,
        111,
        nrows_ncols=(2, 2),
        axes_pad=0.2,
        cbar_mode='single',
        cbar_location='right',
        cbar_pad=0.1,
        # aspect=1.
    )

    rel = "K18"
    dms_halos = np.load(lib.constraints_npy_path(relationship=rel)) * lib.dm_units

    ax = grid[0]
    dm_igm = dm_igm_avg
    dms_all = dm_mw_ism + dm_igm + lib.dm_host_ism_lb + dms_halos
    c = ax.pcolor(
        rmaxes,
        fs_hot,
        dms_all.value,
        cmap='cmr.cosmic_r',
        alpha=1,
        norm=ImageNormalize(
            data=dms_all.value,
            # stretch=LogStretch(),
            vmax=lib.fld.frb.dm.value,
            vmin=absmin,
        )
    )
    ax.set_xlim(np.min(rmaxes), np.max(rmaxes))
    ax.set_ylim(np.min(fs_hot), np.max(fs_hot))
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel(r"K+18")
    ax.set_xlabel(r" ")
    ax.set_aspect(1.)
    # ax.axis("equal")
    ax.set_title(r"$\langle \mathrm{DM_{IGM}} \rangle$", fontsize=12)

    ax = grid[1]
    dm_igm = lib.dm_igm_flimflam
    dms_all = dm_mw_ism + dm_igm + lib.dm_host_ism_lb + dms_halos
    c = ax.pcolor(
        rmaxes,
        fs_hot,
        dms_all.value,
        cmap='cmr.cosmic_r',
        alpha=1,
        norm=ImageNormalize(
            data=dms_all.value,
            # stretch=LogStretch(),
            vmax=lib.fld.frb.dm.value,
            vmin=absmin,
        )
    )
    ax.set_xlim(np.min(rmaxes), np.max(rmaxes))
    ax.set_ylim(np.min(fs_hot), np.max(fs_hot))
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel(r" ")
    ax.set_ylabel(r" ")
    ax.set_aspect(1.)
    # ax.axis("equal")
    ax.set_title("$\mathrm{DM^{IGM}_{FLIMFLAM}}$", fontsize=12)

    rel = "M13"
    dms_halos = np.load(lib.constraints_npy_path(relationship=rel)) * lib.dm_units

    ax = grid[2]
    dm_igm = dm_igm_avg
    dms_all = dm_mw_ism + dm_igm + lib.dm_host_ism_lb + dms_halos
    c = ax.pcolor(
        rmaxes,
        fs_hot,
        dms_all.value,
        cmap='cmr.cosmic_r',
        alpha=1,
        norm=ImageNormalize(
            data=dms_all.value,
            # stretch=LogStretch(),
            vmax=lib.fld.frb.dm.value,
            vmin=absmin,
        )
    )
    ax.set_xlim(np.min(rmaxes), np.max(rmaxes))
    ax.set_ylim(np.min(fs_hot), np.max(fs_hot))
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel(r"M+13")
    ax.set_xlabel(r" ")
    ax.set_aspect(1.)
    # ax.axis("equal")
    # ax.set_title("M+13, Avg")

    ax = ax = grid[3]
    dm_igm = lib.dm_igm_flimflam
    dms_all = dm_mw_ism + dm_igm + lib.dm_host_ism_lb + dms_halos
    c = ax.pcolor(
        rmaxes,
        fs_hot,
        dms_all.value,
        cmap='cmr.cosmic_r',
        alpha=1,
        norm=ImageNormalize(
            data=dms_all.value,
            # stretch=LogStretch(),
            vmax=lib.fld.frb.dm.value,
            vmin=absmin,
        )
    )
    ax.set_xlim(np.min(rmaxes), np.max(rmaxes))
    ax.set_ylim(np.min(fs_hot), np.max(fs_hot))
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel(r" ")
    ax.set_ylabel(r" ")
    ax.set_aspect(1.)
    # ax.axis("equal")
    # ax.set_title("M+13, FLIMFLAM")

    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel(r"$R_\mathrm{max} / R_\mathrm{200}$", labelpad=-5, fontsize=16)
    ax.set_ylabel(r"$f_\mathrm{hot}$", labelpad=20, fontsize=18)

    # cbar = fig.colorbar(c, label="$\mathrm{DM_\mathrm{modelled}}$ (pc\,cm$^{-3}$)")
    cbar = grid.cbar_axes[0].colorbar(c, label="$\mathrm{DM_\mathrm{modelled}}$ (pc\,cm$^{-3}$)")
    # cbar.set_label(
    #     ,
    #     # labelpad=-4
    # )

    # fig.tight_layout()
    lib.savefig(fig, f"Rmax-fhot_combined", subdir="rmax-fhot", png_to_db=True)

    print(f"{dm_mw_ism=}")
    print(f"{lib.dm_host_ism_lb=}")
    lib.add_commands(commands_dict)


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
        "-r",
        help="SMHM relationship.",
        type=str,
        default="K18"
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        rel=args.r,
    )
