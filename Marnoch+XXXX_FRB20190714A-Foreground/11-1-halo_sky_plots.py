#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import copy
import os

import numpy as np
from matplotlib import pyplot as plt

import astropy.units as units
from astropy.cosmology import Planck18
from astropy.visualization import quantity_support
from astropy import table

import craftutils.plotting as pl
from craftutils import params as p
import craftutils.observation.objects as objects

import lib

description = """
Generates the figures showing sky projections of DM_halos.
"""

frb190714 = lib.fld

params = p.plotting_params()
pl.latex_setup()
size_font = params['size_font']
size_label = params['size_label']
size_legend = params['size_legend']
weight_line = params['weight_line']
width = params['a4_width']

objects.cosmology = Planck18

dmunits = lib.dm_units

quantity_support()

step_size_halo = 0.1 * units.kpc
rmax = 1.


def halo_plots_grid(
        filename,
        tbl,
        base_img,
        rel: str,
        padding=10 * units.arcsec,
        n_y=None,
        n_x=2,
        frb_ellipse: bool = False,
        frb_colour: str = "limegreen",
        corners=None,
        fig_factor=3.,
):
    if n_y is None:
        n_y = int(np.ceil(len(tbl) / n_x))

    fig = plt.figure(figsize=(lib.figwidth, lib.figwidth * n_y / fig_factor))

    plt_grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(n_y, n_x),
        axes_pad=0.,
        cbar_mode='single',
        cbar_location='right',
        cbar_pad=0.1
    )

    if corners is None:
        corners = lib.construct_corners(tbl, img=base_img, padding=padding)
    left = corners["left"]
    right = corners["right"]
    bottom = corners["bottom"]
    top = corners["top"]

    halo_grids = {row['id']: np.load(lib.halo_npy_path(row['id'], rel=rel)) for row in tbl}
    if "HG20190714A" in halo_grids:
        halo_grids["HG20190714A"] /= 2

    for ax in plt_grid:
        ax.xaxis.set_tick_params(color="none")
        ax.xaxis.set_ticklabels([])
        #     ax.xaxis.set_ticklabel_params(color="none")
        ax.yaxis.set_tick_params(color="none")
        ax.yaxis.set_ticklabels([])
    #     ax.yaxis.set_ticklabel_params(color="none")

    vmax = np.max(np.max(list(halo_grids.values())))
    for i, row in enumerate(tbl):
        grid = halo_grids[row['id']]

        #     ax = fig.add_subplot(n_y, n_x, n, projection=img_fors2_g.wcs)
        ax = plt_grid[i]

        #     ax.set_axis_off()

        #     ra = ax.coords[0]
        #     dec = ax.coords[1]
        #     if n%2 == 0:
        #         dec.set_ticklabel_visible(False)
        #     if n in (1,2):
        #         ra.set_ticklabel_position('t')
        #     elif n in (6,7):
        #         ra.set_ticklabel_position('b')
        #     else:
        #         ra.set_ticklabel_visible(False)
        #     ra.set_
        frb_x, frb_y = frb190714.frb.position.to_pixel(base_img.wcs[0])
        ims = ax.imshow(
            grid,
            cmap='plasma',
            origin='lower',
            vmax=vmax,
            vmin=0
        )
        if frb_ellipse:
            frb190714.frb_ellipse_to_plot(ax, base_img, colour=frb_colour)
        else:
            ax.scatter(frb_x, frb_y, color=frb_colour, marker="x")

        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

        ax.set_title(row['id_short'], y=0., x=0.05, c="white", ha='left')

    cbar = ax.cax.colorbar(ims, label="$\mathrm{DM_{halos}}$ (pc\,cm$^{-3}$)")
    # cbar = grid.cbar_axes[0].colorbar(ax)
    fig.tight_layout(pad=0.8)
    lib.savefig(fig, filename, subdir="imaging", tight=True)
    return fig, plt_grid


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    img_fors2_g = lib.load_image("vlt-fors2_g")

    for rel in ("K18", "M13"):

        full_grid = np.load(lib.halo_npy_path("all", rel=rel))

        halos_img = img_fors2_g.copy(destination=os.path.join(lib.halo_npy_dir, "halos_img"))
        halos_img.data[0] = full_grid * lib.dm_units
        halos_img.write_fits_file()
        halos_img.load_wcs()

        properties = lib.read_master_properties()
        halo_tbl = lib.read_master_table()

        lib.label_objects(
            tbl=halo_tbl[halo_tbl["dm_halo_K18"] > 0],
            img=halos_img,
            do_ellipses=True,
            ellipse_colour="white",
            output=f"halos_all_{rel}",
            frb_cross=True,
            short_labels=True,
            text_colour="black",
            frb_kwargs={"colour": "limegreen"},
            imshow_kwargs={"cmap": "plasma"},
            normalize_kwargs={
                "stretch": "linear",
                # "vmin": 0,
                "vmax": None
            },
            figsize=(0.75 * lib.figwidth_sideways, 0.75 * lib.figwidth_sideways * 2 / 3),
            do_colorbar=True
        )

        lib.label_objects(
            tbl=halo_tbl[halo_tbl["sample"] == "MUSE"],
            img=halos_img,
            do_ellipses=True,
            ellipse_colour="white",
            short_labels=True,
            text_colour="black",
            output=f"halos_muse_{rel}",
            imshow_kwargs={"cmap": "plasma"},
            normalize_kwargs={"stretch": "linear", "vmin": None, "vmax": None},
            do_colorbar=True,
            frb_kwargs={"colour": "limegreen"},
            figsize=(lib.figwidth, lib.figwidth * 2 / 3)
        )

        # for k, row in enumerate(halo_tbl):
        #     obj_name = row["id"]
        #     this_grid, other = halo_plot(
        #         row=row,
        #         img=img_fors2_g,
        #     )

        # import frb.halos.models as models
        #
        # fake_smc_obj = copy.deepcopy(frb190714.objects_dict["FGa_20190714A"])
        # fake_smc_obj.halo_mnfw = models.SMC()
        # fake_smc_obj.name = "SMC"
        #
        # plot_halo(
        #     this_obj=fake_smc_obj,
        #     img=img_fors2_g,
        #     left=left,
        #     right=right,
        #     bottom=bottom,
        #     top=top,
        #     output=plot_path + f"halo_smc.npy",
        #     rmax=rmax,
        #     s_factor=1,
        # )
        #
        # fake_lmc_obj = copy.deepcopy(frb190714.objects_dict["FGa_20190714A"])
        # fake_lmc_obj.halo_mnfw = models.LMC()
        # fake_lmc_obj.name = "LMC"
        #
        # plot_halo(
        #     this_obj=fake_lmc_obj,
        #     img=img_fors2_g,
        #     left=left,
        #     right=right,
        #     bottom=bottom,
        #     top=top,
        #     output=plot_path + f"halo_lmc.npy",
        #     rmax=rmax,
        #     s_factor=1,
        # )

        # ra_max = halo_tbl["ra"].max() + 10 * units.arcsec
        # ra_min = halo_tbl["ra"].min() - 10 * units.arcsec
        # dec_max = halo_tbl["dec"].max() + 10 * units.arcsec
        # dec_min = halo_tbl["dec"].min() - 10 * units.arcsec
        #
        # corner_1 = SkyCoord(ra_max, dec_max)
        # corner_2 = SkyCoord(ra_min, dec_min)
        #
        # corner_x_1, corner_y_1 = corner_1.to_pixel(img_fors2_g.wcs)
        # corner_x_2, corner_y_2 = corner_2.to_pixel(img_fors2_g.wcs)
        #
        # left = int(np.min([corner_x_1, corner_x_2])) #+ 170
        # right = int(np.max([corner_x_1, corner_x_2])) #- 80
        # bottom = int(np.min([corner_y_1, corner_y_2])) #+ 100
        # top = int(np.max([corner_y_1, corner_y_2])) #- 60
        #
        # plot_halos(
        #     halo_tbl=halo_tbl,
        #     img=img_fors2_g,
        #     left=left,
        #     right=right,
        #     bottom=bottom,
        #     top=top,
        #     output=plot_path + "small",
        #     s_factor=1,
        #     rmax=1.,
        # )

        halo_plots_grid(
            filename=f"halos_grid_{rel}",
            tbl=halo_tbl[halo_tbl["dm_halo_K18"] > 0],
            base_img=halos_img,
            fig_factor=5.5,
            rel=rel
        )
        halos_muse = halo_tbl[halo_tbl["sample"] == "MUSE"]
        halo_plots_grid(
            filename=f"halos_grid_muse_{rel}",
            tbl=halos_muse,
            base_img=halos_img,
            fig_factor=3.5,
            rel=rel
        )

        fake_tbl = table.QTable.read(os.path.join(output_dir, "fictitious_halos.ecsv"))
        print(fake_tbl['mass_halo_mc_K18'])
        print(halo_tbl['mass_halo_mc_K18'])
        fake_tbl.add_row(halo_tbl[1][fake_tbl.colnames])

        halo_plots_grid(
            filename=f"halos_fake_{rel}",
            tbl=fake_tbl,
            base_img=halos_img,
            corners=lib.construct_corners(tbl=halos_muse, img=halos_img),
            rel=rel
        )

from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid

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
