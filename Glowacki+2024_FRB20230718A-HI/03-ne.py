#!/usr/bin/env python
# Code by Lachlan Marnoch, 2024

import os

import matplotlib.pyplot as plt
import numpy as np

from astropy import units
from astropy.coordinates import SkyCoord, Galactocentric
from astropy.visualization import ImageNormalize, LogStretch
import pygedm

import craftutils.utils as u
import craftutils.plotting as pl

import lib

description = """
Generates the figures showcasing the Milky Way DM models with this sightline.
"""


def rotate(x, y, theta):
    theta = u.dequantify(theta, units.rad)
    x_prime = x * np.cos(theta) + y * np.sin(theta)
    y_prime = - x * np.sin(theta) + y * np.cos(theta)
    # print(x_prime, y_prime)
    return x_prime, y_prime


def formatter(x, pos):
    return str(int(x) // 1000)


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb20230718 = lib.fld

    pl.latex_setup()

    xmin = -20e3
    xmax = 20e3
    ymin = -20e3
    ymax = 20e3
    zmin = -5e3
    zmax = 5e3

    figsize = pl.textwidths["mqthesis"]

    frb = frb20230718.frb.position.galactic

    print("Generating n_e maps:")

    theta = 270 * units.deg - frb.l
    theta_z = frb.b

    for typ in ["rotated"]:  # , "flat":

        fig_panels = plt.figure(figsize=(figsize, figsize))
        gs = fig_panels.add_gridspec(
            nrows=3, ncols=2,
            height_ratios=(0.2, 4, 1),
            width_ratios=(3, 3)
        )

        for k, method in enumerate(["ymw16", "ne2001"]):
            print(f"{method.upper()}...")
            sun_x, sun_y, sun_z = pygedm.convert_lbr_to_xyz(
                gl=0, gb=0,
                dist=0,
                method=method
            )
            sun_x_prime, sun_z_prime = rotate(sun_x, sun_z, -theta_z)
            sun_x_prime, sun_y_prime = rotate(sun_x_prime, sun_y, -theta)
            print(sun_x_prime, sun_z, -theta_z)
            print("\t Sun:", sun_x, sun_y, sun_z)
            print("\t\t", sun_x_prime, sun_y_prime, sun_z_prime)

            frb_x, frb_y, frb_z = pygedm.convert_lbr_to_xyz(
                gl=frb.l, gb=frb.b,
                dist=frb20230718.frb.D_comoving,
                method=method
            )
            frb_x_prime, frb_y_prime = rotate(frb_x, frb_y, -theta)
            frb_x_prime, frb_z_prime = rotate(frb_x_prime, frb_z, -theta_z)
            print("\t FRB:", frb_x, frb_y, frb_z)
            print("\t\t", frb_x_prime, frb_y_prime, frb_z_prime)
            print("\t\t", frb.l.value, frb.b.to("deg").value)

            central_x, central_y, central_z = pygedm.convert_lbr_to_xyz(
                gl=0, gb=0, dist=27000 * units.lightyear, method=method
            )
            central_x_prime, central_y_prime = rotate(central_x, central_y, -theta)
            central_x_prime, central_z_prime = rotate(central_x_prime, central_z, -theta_z)
            print("\t Sag A*:", central_x, central_y, central_z)
            print("\t\t", central_x_prime, central_y_prime, central_z_prime)

            X = np.arange(xmin, xmax, 1e2)
            Y = np.arange(ymin, ymax, 1e2)
            Z = np.arange(zmin, zmax, 1e2)

            ne_image_top = np.zeros((len(Y), len(X))) * 1 / units.cm ** 3

            for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    if typ == "flat":
                        ne = pygedm.calculate_electron_density_xyz(x, y, 0, method=method)
                        ne_image_top[j, i] = ne
                    else:
                        x_prime, y_prime = rotate(x, y, theta)
                        x_prime, z_prime = rotate(x_prime, 0, theta_z)
                        ne = pygedm.calculate_electron_density_xyz(x_prime, y_prime, 0, method=method)
                        ne_image_top[j, i] = ne

            ne_image_side = np.zeros((len(Z), len(X))) * 1 / units.cm ** 3

            for i, x in enumerate(X):
                for j, z in enumerate(Z):
                    if typ == "flat":
                        ne = pygedm.calculate_electron_density_xyz(x, sun_y, z, method=method)
                        ne_image_side[j, i] = ne
                    else:
                        x_prime, y_prime = rotate(x, sun_y.value, theta)
                        x_prime, z_prime = rotate(x_prime, z, theta_z)
                        ne = pygedm.calculate_electron_density_xyz(x_prime, y_prime, z_prime, method=method)
                        ne_image_side[j, i] = ne

            fig_top_solo, ax_top_solo = plt.subplots(figsize=(figsize, figsize))
            n = k + 1
            ax_top_panel = fig_panels.add_subplot(gs[1, k])
            # ax_top_panel = fig_panels.add_subplot(2, 2, n)
            ax_top_panel.get_xaxis().set_major_formatter(formatter=formatter)
            ax_top_panel.get_yaxis().set_major_formatter(formatter=formatter)

            for ax_top in (ax_top_solo, ax_top_panel):
                c = ax_top.pcolor(
                    X,
                    Y,
                    ne_image_top,
                    cmap='plasma',
                    alpha=1,
                    norm=ImageNormalize(
                        data=ne_image_top.value,
                        stretch=LogStretch(),
                        vmax=5,
                        vmin=0
                    )
                )
                ax_top.set_xlim(xmin, xmax)
                ax_top.set_ylim(ymin, ymax)
                ax_top.set_aspect("equal")

                if typ == "rotated":
                    ax_top.scatter(sun_x_prime.value, sun_y_prime.value, marker="x", c="black")
                    ax_top.plot(
                        [sun_x_prime.value, frb_x_prime.value],
                        [sun_y_prime.value, frb_y_prime.value],
                        c="black", ls=":",
                        label="FRB\,20230718A LOS"
                    )
                    # ax.plot(
                    #     [sun_x.value, central_x.value],
                    #     [sun_y.value, central_y.value],
                    #     c="black", ls=":",
                    #     label="Centre LOS"
                    # )
                else:
                    ax_top.scatter(sun_x.value, sun_y.value, marker="x", c="black")
                    ax_top.plot(
                        [sun_x.value, frb_x.value],
                        [sun_y.value, frb_y.value],
                        c="black", ls=":",
                        label="FRB\,20230718A LOS"
                    )
                    # ax.plot(
                    #     [sun_x_prime.value, central_x_prime.value],
                    #     [sun_y_prime.value, central_y_prime.value],
                    #     c="black", ls=":",
                    #     label="Centre LOS"
                    # )

                ax_top.get_xaxis().set_tick_params(
                    labelbottom=False,
                    direction="out"
                )

                ax_top.set_title(method.upper())

                if n == 2:
                    ax_top.get_yaxis().set_tick_params(
                        labelleft=False,
                        direction="out"
                    )
                else:
                    ax_top.get_yaxis().set_tick_params(
                        labelsize=pl.tick_fontsize,
                        direction="out"
                    )
                    ax_top.set_ylabel(
                        "$y$ (kpc)",
                        fontsize=pl.axis_fontsize
                    )

                # else:
                #     ax_top.set_yticks(
                #         list(range(int(ymin), int(ymax) + 10000, 10000)),
                #         list(range(-20, 30, 10)),
                #         fontsize=pl.tick_fontsize
                #     )

            # cbar = fig.colorbar(c)
            fig_top_solo.savefig(os.path.join(lib.output_path, f"FRB20230718A_{method}_{typ}_top.pdf"))

            fig_side_solo, ax_side_solo = plt.subplots(figsize=(figsize, figsize))
            n = k + 3
            ax_side_panel = fig_panels.add_subplot(gs[2, k])
            # ax_side_panel = fig_panels.add_subplot(2, 2, n)
            ax_side_panel.get_xaxis().set_major_formatter(formatter=formatter)
            ax_side_panel.get_yaxis().set_major_formatter(formatter=formatter)

            for ax_side in (ax_side_solo, ax_side_panel):
                c = ax_side.pcolor(
                    X,
                    Z,
                    ne_image_side,
                    cmap='plasma',
                    alpha=1,
                    norm=ImageNormalize(
                        data=ne_image_side.value,
                        stretch=LogStretch(),
                        vmax=5,
                        vmin=0
                    )
                )
                ax_side.set_xlim(xmin, xmax)
                ax_side.set_ylim(zmin, zmax)
                ax_side.set_aspect("equal")

                if typ == "flat":
                    ax_side.scatter(sun_x.value, sun_z.value, marker="x", c="black")
                    ax_side.plot(
                        [sun_x.value, frb_x.value],
                        [sun_z.value, frb_z.value],
                        c="black", ls=":",
                        label="FRB\,20230718A LOS"
                    )
                else:
                    ax_side.scatter(sun_x_prime.value, sun_z_prime.value, marker="x", c="black")
                    ax_side.plot(
                        [sun_x_prime.value, frb_x_prime.value],
                        [sun_z_prime.value, frb_z_prime.value],
                        c="black", ls=":",
                        label="FRB\,20230718A LOS"
                    )

                ax_side.get_xaxis().set_tick_params(
                    labelsize=pl.tick_fontsize,
                    direction="out"
                )

                if n == 4:
                    ax_side.get_yaxis().set_tick_params(
                        labelleft=False,
                        direction="out"
                    )
                else:
                    ax_side.get_yaxis().set_tick_params(
                        labelsize=pl.tick_fontsize,
                        direction="out"
                    )
                    ax_side.set_ylabel(
                        "$z$ (kpc)",
                        fontsize=pl.axis_fontsize
                    )

                # ax_side.set_xticks(
                #     list(range(-20, 30, 10)),
                #     list(range(int(xmin), int(xmax) + 10000, 10000)),
                #     fontsize=pl.tick_fontsize
                # )
                #
                # if ax_side is ax_side_panel and n == 4:
                #     ax_side.set_yticks(
                #         list(range(int(zmin), int(zmax) + 10000, 10000)),
                #         labels=list(range(-10, 20, 10)),
                #         visible=False
                #     )
                # else:
                #     ax_side.set_yticks(
                #         list(range(int(zmin), int(zmax) + 10000, 10000)),
                #         labels=list(range(-10, 20, 10)),
                #         fontsize=pl.tick_fontsize
                #     )

            # cbar = fig.colorbar(c)
            fig_side_solo.savefig(os.path.join(lib.output_path, f"FRB20230718A_{method}_{typ}_side.pdf"))

        ax = fig_panels.add_subplot(gs[1:3, :])
        ax.set_frame_on(False)
        ax.tick_params(left=False, right=False, top=False, bottom=False)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        # ax.set_ticks([])
        # ax.set_aspect("equal")
        ax.set_xlabel(
            "$x$ (kpc)",
            labelpad=-5,
            fontsize=pl.axis_fontsize
        )

        ax_cbar = fig_panels.add_subplot(gs[0, :])

        cbar = fig_panels.colorbar(
            c,
            cax=ax_cbar,
            shrink=0.7,
            orientation="horizontal",
            ticklocation="top"
        )
        # cbar.set_tick
        cbar.set_label("$n_\mathrm{e}$ (cm$^{-3}$)", labelpad=-4, )

        fig_panels.subplots_adjust(wspace=0.1, hspace=-0.6)
        fig_panels.savefig(
            os.path.join(lib.output_path, f"FRB20230718A_GEDM_{typ}_panels.png"),
            dpi=200,
            # bbox_inches="tight",
            # pad_inches=-0.1
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
