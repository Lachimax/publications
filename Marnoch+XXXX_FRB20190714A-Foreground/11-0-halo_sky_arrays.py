#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import copy
import os

import numpy as np

import astropy.units as units
from astropy.cosmology import Planck18
from astropy.visualization import quantity_support, make_lupton_rgb
from astropy import table

import craftutils.observation.field as field
import craftutils.plotting as pl
from craftutils import params as p
import craftutils.observation.objects as objects
import craftutils.utils as u
import craftutils.observation.image as image

import lib

description = """
Generates the sky grids of DM_halos.
"""

frb190714 = lib.fld

params = p.plotting_params()
pl.latex_setup()
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


def halo_array(
        row,
        img,
        left,
        right,
        bottom,
        top,
        rel: str,
        rmax=1.,
        s_factor=1,
        this_obj=None,
):
    """
    The Covenant has the Index. Prepare to fire.

    :param row:
    :param img:
    :param left:
    :param right:
    :param bottom:
    :param top:
    :param rel:
    :param rmax:
    :param s_factor:
    :param this_obj:
    :return:
    """
    name_full = row["id"]
    if this_obj is None:
        this_obj = lib.fld.objects[name_full]

    this_grid = np.zeros((top - bottom, right - left))
    pix_x, pix_y = img.extract_pixel_scale()
    pix_scale = (1 * units.pix).to(units.radian, pix_y) / (s_factor * units.radian)
    physical_scale = (pix_scale * this_obj.angular_size_distance()).to(units.kpc)

    x_pix, y_pix = this_obj.position.to_pixel(img.wcs[0], 0)

    x_pix = x_pix * s_factor - left
    y_pix = y_pix * s_factor - bottom

    x_physical = x_pix * physical_scale
    y_physical = y_pix * physical_scale


    print(row["id_short"], ":", row[f"log_mass_halo_mc_{rel}"], row[f"mass_halo_mc_{rel}"])
    this_obj.log_mass_halo = row[f"log_mass_halo_mc_{rel}"]
    this_obj.mass_halo = row[f"mass_halo_mc_{rel}"]
    halo = this_obj.halo_model_mnfw()

    print(row["id_short"], ":", halo.log_Mhalo, halo.M_halo)

    for i in range(this_grid.shape[1]):
        percent_this = 100 * i / this_grid.shape[1]
        x = i * physical_scale
        print(f"{this_obj.name}: {x}, {i + 1}/{this_grid.shape[1]}; {percent_this}%")
        for j in range(this_grid.shape[0]):
            y = j * physical_scale
            r = np.sqrt((x - x_physical) ** 2 + (y - y_physical) ** 2)
            if r < rmax * halo.r200:
                this_grid[j, i] += halo.Ne_Rperp(r, rmax=rmax).value / (1 + this_obj.z)

    r200_pix = halo.r200 / (physical_scale * s_factor)

    # fig, ax = plt.subplots()
    #
    # ims = ax.imshow(this_grid, cmap='plasma', origin='lower')
    # plt.colorbar(ims, ax=ax)
    # ax.scatter(
    #     x_pix,
    #     y_pix,
    #     marker="x")
    # plt.xlabel("")
    # plt.ylabel("")
    # e = Ellipse(
    #     xy=(x_pix, y_pix),
    #     width=2 * r200_pix,
    #     height=2 * r200_pix,
    #     angle=0.0
    # )
    # e.set_facecolor('none')
    # e.set_edgecolor("white")
    # ax.add_artist(e)
    # fig.show()

    np.save(lib.halo_npy_path(name_full, rel=rel), this_grid)

    other = {
        "x_pix": x_pix,
        "y_pix": y_pix,
        "x_physical": x_physical,
        "y_physical": y_physical,
        "r200_pix": r200_pix,
        "pix_scale": pix_scale,
        "physical_scale": physical_scale
    }

    return this_grid, other


def halos_array(
        halo_tbl,
        img,
        left,
        right,
        bottom,
        top,
        rel: str,
        rmax=1.,
        s_factor=1,
        objs: list = None,
        suffix: str = "",
):
    grids = {}
    grid_list = []
    others = {}
    for k, row in enumerate(halo_tbl):
        obj_name = row["id"]
        if objs is None:
            this_obj = None
        else:
            this_obj = objs[k]
        this_grid, other = halo_array(
            row=row,
            img=img,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            rmax=rmax,
            s_factor=s_factor,
            this_obj=this_obj,
            rel=rel,
        )
        if obj_name.startswith("HG"):
            this_grid /= 2.
        others[obj_name] = other
        grids[obj_name] = this_grid
        grid_list.append(this_grid)
    grid_all = np.sum(grid_list, 0)

    np.save(lib.halo_npy_path("all" + suffix, rel=rel), grid_all)

    # fig, ax = plt.subplots()
    # ims = ax.imshow(grid_all, cmap='plasma', origin='lower')
    # plt.scatter(halo_tbl["x"]-left, halo_tbl["y"]-bottom, marker="x")
    # plt.xlabel("")
    # plt.ylabel("")
    # plt.colorbar(ims, ax=ax)
    #
    # for obj_name in r200_pixs:
    #     properties = others[obj_name]
    #     r200 = properties["r200_pix"]
    #     e = Ellipse(
    #         xy=(properties["x_pix"], properties["y_pix"]),
    #         width=r200,
    #         height=r200,
    #         angle=0.0,
    #     )
    #     e.set_facecolor('none')
    #     e.set_edgecolor("white")
    #     ax.add_artist(e)
    #
    # fig.show()
    return grid_all, grids, grid_list, others


def main(
        output_dir: str,
        input_dir: str,
        fictitious_only: bool,
        tables_only: bool
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    img_fors2_g = lib.load_image("vlt-fors2_g")
    left = 0
    right = img_fors2_g.data[0].shape[1]
    bottom = 0
    top = img_fors2_g.data[0].shape[0]

    properties = lib.read_master_properties()
    halo_tbl = lib.read_master_table()

    for rel in ("K18", "M13"):

        if not fictitious_only and not tables_only:
            halos_array(
                halo_tbl=halo_tbl,
                img=img_fors2_g,
                left=left,
                right=right,
                bottom=bottom,
                top=top,
                s_factor=1,
                rmax=rmax,
                rel=rel
            )

        import frb.halos.models as models

        fake_smc_obj = copy.deepcopy(frb190714.objects["FGa_20190714A"])
        fake_smc_obj.halo_mnfw = models.SMC()
        fake_smc_obj.name = "SMC"

        fake_lmc_obj = copy.deepcopy(frb190714.objects["FGa_20190714A"])
        fake_lmc_obj.halo_mnfw = models.LMC()
        fake_lmc_obj.name = "LMC"

        fake_tbl = table.QTable(halo_tbl[[1, 1]])
        fake_tbl["id"] = fake_tbl["id_short"] = ["SMC", "LMC"]
        fake_tbl["log_mass_halo_mc_K18"] = fake_tbl["log_mass_halo_mc_M13"] = [
            fake_smc_obj.halo_mnfw.log_Mhalo,
            fake_lmc_obj.halo_mnfw.log_Mhalo,
        ]
        fake_tbl["mass_halo_mc_K18"] = fake_tbl["mass_halo_mc_M13"] = [
            fake_smc_obj.halo_mnfw.M_halo.to(units.solMass),
            fake_lmc_obj.halo_mnfw.M_halo.to(units.solMass),
        ]
        fake_tbl.write(os.path.join(output_dir, "fictitious_halos.ecsv"), overwrite=True)
        fake_tbl.write(os.path.join(output_dir, "fictitious_halos.csv"), overwrite=True)

        if not tables_only:
            halos_array(
                halo_tbl=fake_tbl,
                img=img_fors2_g,
                left=left,
                right=right,
                bottom=bottom,
                top=top,
                s_factor=1,
                rmax=rmax,
                objs=[fake_smc_obj, fake_lmc_obj],
                suffix="_fake",
                rel=rel
            )

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
        "-f",
        help="Skip real halos and only do the fake ones.",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        help="Only construct tables, not arrays.",
        action="store_true",
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        fictitious_only=args.f,
        tables_only=args.t
    )
