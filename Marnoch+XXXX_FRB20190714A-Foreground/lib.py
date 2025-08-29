import os
import shutil

from typing import List

import numpy as np
import matplotlib.pyplot as plt

from astropy import table
from astropy import units
from astropy import constants
from astropy.stats import sigma_clip

import craftutils.utils as u
import craftutils.params as p
from craftutils.observation import objects, field, instrument, image
from craftutils import plotting as pl

pl.latex_setup()

params = p.plotting_params()
pl.latex_setup()
size_font = params['size_font']
size_label = params['size_label']
size_legend = params['size_legend']
weight_line = params['weight_line']
width = params['a4_width']

script_dir = os.path.dirname(__file__)

params = p.load_params(os.path.join(script_dir, "params.yaml"))
_, paper_name = os.path.split(script_dir)

top_data_dir = p.data_dir
if top_data_dir is None:
    top_data_dir = os.path.join(
        os.path.expanduser("~"),
        "Data",
    )

default_data_dir = os.path.join(
    top_data_dir,
    "publications",
    paper_name
)

default_output_path = os.path.join(
    default_data_dir, "output"
)
output_path = default_output_path
os.makedirs(default_output_path, exist_ok=True)


def set_output_path(path):
    global output_path
    output_path = path
    u.mkdir_check_nested(output_path, False)


default_input_path = os.path.join(
    default_data_dir, "input"
)
input_path = default_input_path
data_path = os.path.join(script_dir, "data")
param_path = os.path.join(script_dir, "param")


def set_input_path(path):
    global input_path
    input_path = path
    u.mkdir_check_nested(input_path, False)


f_igm = 0.59
f_igm_plus_err = 0.11
f_igm_minus_err = 0.10

f_gas = 0.55
f_gas_plus_err = 0.26
f_gas_minus_err = 0.29

dm_units = units.pc / (units.cm ** 3)
dm_host_ism_lb = 64.8 * dm_units  # 67 * dm_units
dm_host_ism_lb_err = 1.1 * dm_units

dm_igm_ff_fid = 326.5 * dm_units
dm_igm_err_fid = 94.7 * dm_units
dm_igm_flimflam = dm_igm_ff_fid * f_igm
dm_igm_flimflam_err_plus = u.uncertainty_product(
    dm_igm_flimflam,
    (dm_igm_ff_fid, dm_igm_err_fid),
    (f_igm, f_igm_plus_err),
)
dm_igm_flimflam_err_minus = u.uncertainty_product(
    dm_igm_flimflam,
    (dm_igm_ff_fid, dm_igm_err_fid),
    (f_igm, f_igm_minus_err),
)

dm_halos_ff_fid = 321.4 * dm_units
dm_halos_err_fid = 92.1 * dm_units
dm_halos_flimflam = dm_halos_ff_fid * f_gas
dm_halos_flimflam_err_plus = u.uncertainty_product(
    dm_halos_flimflam,
    (dm_halos_ff_fid, dm_halos_err_fid),
    (f_gas, f_gas_plus_err),
)
dm_halos_flimflam_err_minus = u.uncertainty_product(
    dm_halos_flimflam,
    (dm_halos_ff_fid, dm_halos_err_fid),
    (f_gas, f_gas_minus_err),
)

dm_excess_inclination_model = 266.54039408767164 * dm_units
dm_inclination_corrected = 159.36835682785568 * dm_units

fld = field.Field.from_params("FRB20190714A")
fld.frb.get_host()

plot_dir = os.path.join(output_path, "plots")
os.makedirs(plot_dir, exist_ok=True)
dropbox_path = None

if dropbox_path is not None:
    dropbox_figs = os.path.join(dropbox_path, "figures")
    latex_table_path_db = os.path.join(dropbox_path, "tables")
    commands_path_db = os.path.join(dropbox_path, "commands_frb190714_generated_2.tex")


else:
    dropbox_figs = None
    latex_table_path_db = None
    commands_path_db = None

commands_dict = {}
commands_dict_path = os.path.join(output_path, "commands.yaml")
commands_tex_path = os.path.join(output_path, "commands.tex")

latex_table_path = os.path.join(output_path, "latex_tables")
os.makedirs(latex_table_path, exist_ok=True)
os.makedirs(latex_table_path_db, exist_ok=True)

halo_npy_dir = os.path.join(output_path, "halo_arrays")
os.makedirs(halo_npy_dir, exist_ok=True)

rmaxes = np.linspace(0.5, 3, 100)
fs_hot = np.linspace(0, 1, 100)


def latex_gal_id(gal_id: str):
    if gal_id.startswith("HG"):
        gal_id = "HG"
    else:
        gal_id = gal_id[:3]
    return gal_id


def read_commands():
    global commands_dict
    commands_dict = p.load_params(commands_dict_path)
    if commands_dict is None:
        commands_dict = {}
    return commands_dict


def write_commands():
    global commands_dict
    p.save_params(commands_dict_path, commands_dict)
    with open(commands_tex_path, "w") as f:
        lines = []
        for cmd, val in commands_dict.items():
            line = u.latex_command(
                command=cmd,
                value=val
            )
            lines.append(line)
        f.writelines(lines)
    shutil.copy(commands_tex_path, commands_path_db)


def add_commands(new_commands: dict):
    read_commands()
    global commands_dict
    commands_dict.update(new_commands)
    sorted_dict = {}
    for key in sorted(commands_dict.keys()):
        sorted_dict[key] = commands_dict[key]
    commands_dict = sorted_dict
    write_commands()


def savefig(fig, filename, subdir=None, tight=True, png_to_db=False):
    output_this = plot_dir
    if dropbox_figs is not None:
        db_this = str(dropbox_figs)
    else:
        db_this = None
    if subdir is not None:
        output_this = os.path.join(output_this, subdir)
        if db_this is not None:
            db_this = os.path.join(db_this, subdir)
    os.makedirs(output_this, exist_ok=True)
    if db_this is not None:
        os.makedirs(db_this, exist_ok=True)
    output = os.path.join(output_this, filename)
    print("Saving figure to ", output + ".pdf")
    if tight:
        bb = "tight"
    else:
        bb = None
    fig.savefig(output + ".pdf",
                # bbox_inches=bb
                )
    fig.savefig(output + ".png",
                # bbox_inches=bb,
                dpi=400)
    if db_this is not None:
        db_output = os.path.join(db_this, filename)
        print("Saving figure to Dropbox: ", db_output + ".pdf")
        if png_to_db:
            fig.savefig(db_output + ".png", bbox_inches=bb, dpi=400)
        else:
            fig.savefig(db_output + ".pdf", bbox_inches=bb)


##############################################################################

def constraints_npy_path(relationship: str):
    subdir = os.path.join(output_path, relationship, "rmax-fhot")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, "fhot_rmax_dm_halos.npy")


def halo_npy_path(gal_id: str, rel: str):
    path = os.path.join(halo_npy_dir, rel)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"halo_{gal_id}.npy")


##############################################################################

def halo_table_path(rmax: float, relationship: str, fhot: float, fmt: str = "ecsv"):
    subdir = os.path.join(output_path, relationship, "galaxy_halo_tables")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"galaxy_halo_table_{rmax}_{fhot}.{fmt}")


def write_halo_table(tbl: table.QTable, relationship: str, rmax: float, fhot: float = 0.75):
    tbl.write(halo_table_path(rmax, relationship, fhot, fmt="ecsv"), format="ascii.ecsv", overwrite=True)
    tbl.write(halo_table_path(rmax, relationship, fhot, fmt="csv"), format="ascii.csv", overwrite=True)


def read_halo_table(rmax: float, relationship: str, fhot: float = 0.75):
    return table.QTable.read(halo_table_path(rmax, relationship, fhot=fhot), format="ascii.ecsv")


##############################################################################
##############################################################################

def master_table_path(fmt: str = "ecsv"):
    return os.path.join(output_path, f"master_halo_table.{fmt}")


main_cols = [
        "id", "id_short", "letter", "sample", "z", "ra", "dec", 'offset_angle', 'r_perp',
        'id_cat', 'ra_cat', "dec_cat", 'offset_cat', 'log_mass_stellar', 'log_mass_stellar_err'
    ]

def write_master_table(tbl: table.Table):
    tbl.sort("id")
    tbl = tbl[main_cols + sorted(list(set(tbl.colnames) - set(main_cols)))]
    tbl.write(master_table_path(fmt="ecsv"), overwrite=True)
    tbl.write(master_table_path(fmt="csv"), overwrite=True)


def read_master_table():
    print("Loading Master Table...")
    path = master_table_path(fmt="ecsv")
    if os.path.exists(path):
        tbl = table.QTable.read(path)
        tbl.sort("id")
    else:
        tbl = table.QTable()
    print("Complete")
    return tbl


##############################################################################

def master_properties_path():
    return os.path.join(output_path, f"master_properties.yaml")


def write_master_properties(prop_dict: dict):
    p.save_params(master_properties_path(), prop_dict)


def read_master_properties():
    props = p.load_params(master_properties_path())
    if props is None:
        return {}
    else:
        return props


##############################################################################

def halo_table_mc_path(n: int, relationship: str, fmt: str = "ecsv"):
    subdir = os.path.join(output_path, relationship, "MC", "galaxy_halo_tables")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"galaxy_halo_table_mc_{n}.{fmt}")


def write_halo_table_mc(tbl: table.QTable, relationship: str, n: int):
    tbl.write(halo_table_mc_path(n, relationship, fmt="ecsv"), format="ascii.ecsv", overwrite=True)


def read_halo_table_mc(relationship: str, n: int):
    return table.QTable.read(halo_table_mc_path(n, relationship, fmt="ecsv"))


##############################################################################

def halo_individual_table_collated_mc_path(obj_id: str, relationship: str, fmt: str = "ecsv"):
    subdir = os.path.join(output_path, relationship, "MC", "collated_individual_halo_tables")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"{obj_id}_mc.{fmt}")


def write_halo_individual_table_collated_mc(tbl: table.QTable, obj_id: str, relationship: str):
    tbl.write(
        halo_individual_table_collated_mc_path(obj_id=obj_id, relationship=relationship, fmt="ecsv"),
        format="ascii.ecsv",
        overwrite=True
    )


def read_halo_individual_table_collated_mc(relationship: str, obj_id: str):
    return table.QTable.read(
        halo_individual_table_collated_mc_path(obj_id=obj_id, relationship=relationship, fmt="ecsv")
    )


##############################################################################


def halo_table_mc_collated_path(relationship: str, fmt: str = "ecsv"):
    subdir = os.path.join(output_path, relationship, "MC", "galaxy_halo_tables")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"galaxy_halo_table_mc_collated.{fmt}")


def write_halo_table_mc_collated(tbl: table.QTable, relationship: str):
    tbl.write(halo_table_mc_collated_path(relationship, fmt="ecsv"), format="ascii.ecsv", overwrite=True)
    tbl.write(halo_table_mc_collated_path(relationship, fmt="csv"), format="ascii.csv", overwrite=True)


def read_halo_table_mc_collated(relationship: str):
    return table.QTable.read(halo_table_mc_collated_path(relationship, fmt="ecsv"))


##############################################################################


def dm_table_path(rmax: float, relationship: str, fmt: str = "ecsv"):
    subdir = os.path.join(output_path, relationship, "cumulative_DM_tables")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"cumulative_DM_table_{rmax}.{fmt}")


def write_dm_table(tbl: table.QTable, rmax: float, relationship: str):
    tbl.write(dm_table_path(rmax, relationship, fmt="ecsv"), format="ascii.ecsv", overwrite=True)
    tbl.write(dm_table_path(rmax, relationship, fmt="csv"), format="ascii.csv", overwrite=True)


##############################################################################


def dm_table_mc_path(n: int, relationship: str, fmt: str = "ecsv"):
    subdir = os.path.join(output_path, relationship, "MC", "cumulative_DM_tables")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"cumulative_DM_table_{n}.{fmt}")


def write_dm_table_mc(tbl: table.QTable, relationship: str, n: int):
    tbl.write(dm_table_mc_path(n, relationship, fmt="ecsv"), format="ascii.ecsv", overwrite=True)


##############################################################################


def dm_gal_path(galaxy_name: str, rmax: float, relationship: str, fmt: str = "ecsv"):
    subdir = os.path.join(output_path, relationship, "cumulative_DM_tables", f"individual_Rmax_{int(rmax)}")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"{galaxy_name}.{fmt}")


def write_dm_gal_table(galaxy_name: str, tbl: table.QTable, rmax: float, relationship: str):
    tbl.write(dm_gal_path(galaxy_name=galaxy_name, rmax=rmax, fmt="ecsv", relationship=relationship),
              format="ascii.csv", overwrite=True)
    tbl.write(dm_gal_path(galaxy_name=galaxy_name, rmax=rmax, fmt="csv", relationship=relationship),
              format="ascii.ecsv", overwrite=True)


##############################################################################

def properties_path(rmax: float, relationship: str, fhot: float, simple: bool):
    subdir = os.path.join(output_path, relationship, "run_properties")
    os.makedirs(subdir, exist_ok=True)
    if simple:
        path = os.path.join(subdir, f"properties_simple_{rmax}_{fhot}.yaml")
    else:
        path = os.path.join(subdir, f"properties_{rmax}_{fhot}.yaml")
    # print(simple, path)
    return path


def write_properties(prop_dict, rmax: float, relationship: str, fhot: float = 0.75, simple=False):
    p.save_params(properties_path(rmax, relationship, fhot, simple=simple), prop_dict)


def read_properties(rmax: float, relationship: str, fhot: float = 0.75, simple=False):
    return p.load_params(properties_path(rmax, relationship, fhot, simple))


##############################################################################

def properties_path_mc(n: int, relationship: str):
    subdir = os.path.join(output_path, relationship, "MC", "run_properties")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"properties_{n}.yaml")


def write_properties_mc(prop_dict, n: int, relationship: str):
    p.save_params(properties_path_mc(n, relationship), prop_dict)


def read_properties_mc(relationship: str, n: int):
    return p.load_params(properties_path_mc(n, relationship))


##############################################################################


def properties_table_path_mc(relationship: str, fmt="ecsv"):
    subdir = os.path.join(output_path, relationship, "MC", "run_properties")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"properties_all.{fmt}")


def write_properties_table_mc(tbl: table.QTable, relationship: str):
    tbl.write(properties_table_path_mc(relationship, fmt="ecsv"), format="ascii.ecsv", overwrite=True)
    tbl.write(properties_table_path_mc(relationship, fmt="csv"), format="ascii.csv", overwrite=True)


def read_properties_table_mc(relationship: str):
    return table.QTable.read(properties_table_path_mc(relationship, fmt="ecsv"))


##############################################################################


imaging = os.path.join(input_path, "imaging")
img_fnames = {
    "vlt-fors2_g": "FRB20190714A_VLT-FORS2_g-HIGH_combined.fits",
    "vlt-fors2_I": "FRB20190714A_VLT-FORS2_I-BESS_combined.fits",
    "hst-ir": "FRB20190714A_HST-WFC3-IR_F160W_2020-09-29.fits",
    "hst-uvis": "FRB20190714A_HST-WFC3-UVis_F300X_2020-10-28.fits"
}

img_dict = {}

figwidth = pl.textwidths["mqthesis"]
figwidth_sideways = pl.textwidths["mqthesis_sideways"]


def load_imaging(force: bool = False):
    global img_dict

    for band, fname in img_fnames.items():
        load_image(band_name=band, force=force)

    return img_dict


def load_image(band_name: str, force: bool = False):
    fname = img_fnames[band_name]
    img_path = os.path.join(imaging, fname)
    if force or band_name not in img_dict:
        if band_name.startswith("vlt-fors2"):
            cls = image.FORS2CoaddedImage
        else:
            cls = image.HubbleImage
        img = cls(path=img_path)
        img.load_wcs()
        img.load_data()
        img_dict[band_name] = img
    else:
        img = img_dict[band_name]
    print("Loaded image", band_name)
    if img.instrument is not None:
        print("With instrument", img.instrument.name)
    else:
        print("No instrument")
    return img


simha2023_path = os.path.join(input_path, "Simha+2023", "object_catalogs")


def add_labels(
        ax: plt.Axes,
        img,
        tbl,
        short_labels,
        factor,
        height_factor,
        text_colour
):
    for i, row in enumerate(tbl):
        width = row["a"].to(units.pix, img.pixel_scale_y).value * row["kron_radius"]
        height = row["b"].to(units.pix, img.pixel_scale_y).value * row["kron_radius"]
        if short_labels:
            label = row["letter"]
        else:
            label = row['id_short']
        ax.text(
            row["x"] + factor * width,
            row["y"] + height * height_factor,
            label,
            c=text_colour,
            fontsize=10
        )
    return ax


def add_ellipses(
        img,
        ax,
        tbl,
        colour="red",
        linestyle="-",
        linewidth=2,
        cbar_label="Galaxy redshift",
        shrink=0.8
):
    from matplotlib.patches import Ellipse

    cmap = plt.cm.bwr
    do_col = False
    colour_name = ""

    if colour in tbl.colnames:
        vmax = None
        vmin = None
        if np.sum(tbl[colour] < 0) > 0:
            vmax = max(np.abs(tbl[colour].min().value), np.abs(tbl[colour].max().value))
            vmin = -vmax
        colour_name = f"cmap_{colour}"
        norm = plt.Normalize(
            vmax=vmax,
            vmin=vmin
        )
        tbl[colour_name] = cmap(
            norm(
                tbl[colour],

            ),
        )
        do_col = True
    else:
        norm = plt.Normalize()

    for i, row in enumerate(tbl):
        # obj = fld.objects[row["id"]]
        width = row["a"].to(units.pix, img.pixel_scale_y).value * row["kron_radius"]
        height = row["b"].to(units.pix, img.pixel_scale_y).value * row["kron_radius"]
        rotation_angle = img.extract_rotation_angle()
        theta = row["theta"].to(units.deg) - rotation_angle
        e = Ellipse(
            xy=(row["x"], row["y"]),
            width=2 * width,
            height=2 * height,
            angle=-theta.value,
        )

        if do_col:
            colour_this = row[colour_name]
        else:
            colour_this = colour

        e.set_facecolor('none')
        e.set_edgecolor(colour_this)
        e.set_linestyle(linestyle)
        e.set_linewidth(linewidth)
        ax.add_artist(e)

    if do_col:
        cmp = cmap(norm(tbl[colour]))
        # plt.colorbar(cmp, shrink=0.8)

        plt.scatter(tbl["ra"], tbl["dec"], c=tbl[colour], cmap="bwr", norm=norm)
        cbar = plt.colorbar(
            shrink=shrink,
            location="top",
        )
        cbar.ax.set_xlabel(cbar_label)
        # cbar.ax.set_ylabel_position("top")

        return cmp


def label_objects(
        tbl: table.Table,
        img,
        output,
        figsize=None,
        corners=None,
        corner_1=None,
        corner_2=None,
        factor=1.,
        normalize_kwargs=None,
        imshow_kwargs=None,
        frb_kwargs=None,
        ellipse_kwargs=None,
        padding=10 * units.arcsec,
        show_frb: bool = True,
        frb_cross: bool = False,
        do_text=True,
        short_labels=False,
        height_factor=1,
        do_ellipses: bool = True,
        ellipse_colour="red",
        text_colour="white",
        do_cut: bool = None,
        do_colorbar: bool = False,
        save=True,
):
    if corners is None:
        corners = construct_corners(tbl, padding=padding, img=img)
    if corner_1 is None:
        corner_1 = corners["corner_1"]
    if corner_2 is None:
        corner_2 = corners["corner_2"]
    if figsize is None:
        figsize = (figwidth, figwidth * 2 / 3)
    if do_cut is None:
        if "uvis" in img.name.lower():
            do_cut = True
        else:
            do_cut = False

    fig = plt.figure(figsize=figsize)
    if normalize_kwargs is None:
        normalize_kwargs = {}
    if ellipse_kwargs is None:
        ellipse_kwargs = {}
    if frb_kwargs is None:
        frb_kwargs = {"colour": "violet"}

    print(corner_1, corner_2)

    if "vmax" not in normalize_kwargs:
        hg = fld.frb.host_galaxy
        fld.frb.host_galaxy.load_output_file()
        print(img.instrument, img.filter, img.name, type(img))
        phot_dict = hg.photometry[img.instrument.name][img.filter.name]
        phot_dict = phot_dict[list(phot_dict.keys())[0]]
        peak = phot_dict["flux_max"].value + phot_dict["background_se"].value
        # x, y = img.world_to_pixel(hg_coord)
        normalize_kwargs["vmax"] = peak
        # np.max(img.data[0][int(y) - 10:int(y) + 10, int(x) - 10:int(x) + 10]).value

    if do_cut and "vmin" not in normalize_kwargs:
        if "hst" not in img.instrument.name.lower():
            cut = np.nanstd(sigma_clip(img.data[0].value, sigma=1.))
        else:
            cut = 0.
        print(f"CUT {img.name}: {np.nanmean(img.data[0]).value}, {cut}, {np.nanmean(img.data[0]).value + cut}")
        normalize_kwargs["vmin"] = np.nanmean(img.data[0]).value + 9 * cut

    fig, ax, other_args = img.plot_subimage(
        fig=fig,
        corners=(corner_1, corner_2),
        imshow_kwargs=imshow_kwargs,
        normalize_kwargs=normalize_kwargs,
    )
    ax.tick_params(axis="both", labelsize=12)

    if do_colorbar:
        fig.colorbar(
            other_args["mapping"], location="top", label="$\mathrm{DM_{halos}}$ (pc\,cm$^{-3}$)", shrink=0.8
        )

    ra_ax, dec_ax = ax.coords

    # plt.scatter(tbl["ra"], tbl["dec"])
    # plt.scatter(gaia["ra"], gaia["dec"])
    tbl["x"], tbl["y"] = img.wcs[0].all_world2pix(tbl["ra"], tbl["dec"], 0)

    tbl = tbl[tbl["x"] > corners["left"]]
    tbl = tbl[tbl["x"] < corners["right"]]
    tbl = tbl[tbl["y"] > corners["bottom"]]
    tbl = tbl[tbl["y"] < corners["top"]]

    # ax.scatter(frb_x, frb_y, marker="x", c="red")
    # ax.scatter(tbl["x"], tbl["y"], marker="X", c="white")
    cmp = None
    if do_ellipses:
        cmp = add_ellipses(img, ax, tbl=tbl, colour=ellipse_colour, **ellipse_kwargs)
    if do_text:
        add_labels(
            ax=ax, tbl=tbl,
            img=img,
            short_labels=short_labels,
            factor=factor,
            height_factor=height_factor,
            text_colour=text_colour
        )

    #     plt.tight_layout(w_pad=10.)
    ra_ax.set_axislabel('Right Ascension (J2000)', fontsize=size_font)
    dec_ax.set_axislabel("Declination (J2000)", fontsize=14, rotation=-90)
    # dec_ax.set
    #     position="right",
    #     # labelpad=30.,
    #     ,
    #     rotation=-90
    # )
    dec_ax.set_axislabel_position("r")

    if show_frb:
        if "colour" in frb_kwargs:
            c = frb_kwargs.pop("colour")
        else:
            c = "violet"
        if frb_cross:
            frb_x, frb_y = fld.frb.position.to_pixel(img.wcs[0])
            ax.scatter(frb_x, frb_y, color=c, marker="x", **frb_kwargs)
        else:

            fld.frb_ellipse_to_plot(ax, img, include_img_err=False, colour=c, frb_kwargs=frb_kwargs)

    if save:
        savefig(fig, tight=True, subdir="imaging", filename=output)
    # fig.savefig(output.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    return cmp, fig, ax


def construct_corners(
        tbl: table.Table,
        img,
        padding=10 * units.arcsec,
):
    # img_fors2_g = load_image(band_name="vlt-fors2_g")

    tbl = tbl[img.wcs[0].footprint_contains(tbl["coord"])]

    # ra_max = tbl["ra"].max() + padding
    # ra_min = tbl["ra"].min() - padding
    # dec_max = tbl["dec"].max() + padding
    # dec_min = tbl["dec"].min() - padding

    padding_pix = np.max((img.pixel(padding).value, img.pixel(np.max(tbl["a"] * np.max(tbl["kron_radius"]))).value))

    x, y = img.world_to_pixel(tbl["coord"])
    corner_x_1 = np.max(x) + padding_pix
    corner_y_1 = np.max(y) + padding_pix
    corner_x_2 = np.min(x) - padding_pix
    corner_y_2 = np.min(y) - padding_pix

    if corner_x_2 < 0:
        corner_x_2 = 0
    if corner_y_2 < 0:
        corner_y_2 = 0

    corner_1 = img.pixel_to_world(corner_x_1, corner_y_1)
    corner_2 = img.pixel_to_world(corner_x_2, corner_y_2)

    # corner_1 = SkyCoord(ra_max, dec_max)
    # corner_2 = SkyCoord(ra_min, dec_min)

    # corner_x_1, corner_y_1 = corner_1.to_pixel(img.wcs[0])
    # corner_x_2, corner_y_2 = corner_2.to_pixel(img.wcs[0])

    left = int(np.min([corner_x_1, corner_x_2]))
    right = int(np.max([corner_x_1, corner_x_2]))
    bottom = int(np.min([corner_y_1, corner_y_2]))
    top = int(np.max([corner_y_1, corner_y_2]))

    print("Corners: ", corner_1, corner_2)
    print(left, right, bottom, top)

    return dict(
        corners=(corner_1, corner_2),
        corner_1=corner_1,
        corner_2=corner_2,
        corner_x_1=corner_x_1,
        corner_x_2=corner_x_2,
        corner_y_1=corner_y_1,
        corner_y_2=corner_y_2,
        # ra_max=ra_max,
        # ra_min=ra_min,
        # dec_max=dec_max,
        # dec_min=dec_min,
        left=left,
        right=right,
        bottom=bottom,
        top=top
    )


def get_by_id(tbl: table.QTable, ids: List[str], id_type: str = "id_short"):
    tbl = tbl[[n in ids for n in tbl[id_type]]]
    return tbl


def peculiar_velocity(z_obs, z_cos):
    return constants.c.to("km/s") * (z_obs - z_cos) / (1 + z_cos)
