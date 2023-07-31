import os
import copy
from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt

import astropy.table as table
import astropy.units as units
import astropy.cosmology as cosmology
import astropy.coordinates as coordinates
from astropy.modeling import models, fitting

import craftutils.utils as u
import craftutils.params as p
import craftutils.photometry as ph
from craftutils.plotting import latex_setup, textwidths
import craftutils.observation.field as field
import craftutils.observation.sed as sed
import craftutils.observation.instrument as inst
import craftutils.observation.filters as fil
import craftutils.observation.image as image
import craftutils.observation.objects as objects

from matplotlib.lines import Line2D
from matplotlib.pyplot import cm

script_dir = os.path.dirname(__file__)

params = p.load_params(os.path.join(script_dir, "params.yaml"))
_, paper_name = os.path.split(script_dir)

default_data_dir = os.path.join(
    os.path.expanduser("~"),
    "Data",
    "publications",
    paper_name
)

default_output_path = os.path.join(
    default_data_dir, "output"
)
output_path = default_output_path
u.mkdir_check_nested(default_output_path, False)


def set_output_path(path):
    global output_path
    output_path = path
    u.mkdir_check_nested(output_path, False)


default_input_path = os.path.join(
    default_data_dir, "input"
)
input_path = default_input_path
repo_data = os.path.join(script_dir, "data")


def set_input_path(path):
    global input_path
    input_path = path


objects.set_cosmology("Planck18")

afg_units = units.pc ** (-2 / 3) * units.km ** (-1 / 3)

model_dir = os.path.join(
    input_path,
    "a.c.gordon-prospector-models"
)
model_files = os.listdir(model_dir)
events = list(set(map(lambda f: f[:6], model_files)))
events.sort()
# Dictionary in which models and other information will be stored.
model_dict = {
    "FRB20180301A": {"z": 0.3305},
    "FRB20180916B": {"z": 0.0327},
    "FRB20190520B": {"z": 0.2417},
    "FRB20201124A": {"z": 0.0980},
    "FRB20210410D": {"z": 0.1415}
}
do_not_load = [
]

# fld = field.Field.from_params("FRB20210912")
fld = field.Field.from_file(os.path.join(script_dir, "param", "FRB20210912", "FRB20210912.yaml"))

frb_list = field.list_fields()
phot_tbl = table.QTable.read(
    os.path.join(
        p.config["table_dir"],
        "master_select_objects_table.ecsv"
    )
)

faintest = {
    "vlt-fors2_g-HIGH": 27.3,
    "vlt-fors2_R-SPECIAL": 27.0,
    "vlt-fors2_I-BESS": 25.5,
    "vlt-hawki_Ks": 22.3
}

# Load instruments and bandpasses

fors2 = inst.Instrument.from_params("vlt-fors2")
u_fors2 = fors2.filters["u_HIGH"]
b_fors2 = fors2.filters["b_HIGH"]
g_fors2 = fors2.filters["g_HIGH"]
v_fors2 = fors2.filters["v_HIGH"]
R_fors2 = fors2.filters["R_SPECIAL"]
I_fors2 = fors2.filters["I_BESS"]

wise = inst.Instrument.from_params("wise")
w1 = wise.filters["W1"]
w2 = wise.filters["W2"]
w3 = wise.filters["W3"]
w4 = wise.filters["W4"]

hawki = inst.Instrument.from_params("vlt-hawki")
K_hawki = hawki.filters["Ks"]

bands_default = (
    g_fors2,
    R_fors2,
    I_fors2,
    K_hawki
)

bands_wise = (
    w1,
    w2,
    w3,
    w4
)

# def quickpath(
#         p_u: float,
#         img: image.ImagingImage,
#         frb_object: objects.FRB,
#         radius: float = 10
# ):
#     """
#     Performs a customised PATH run on an image.
#
#     :param p_u: The prior for the probability of the host being unseen in the image.
#     :param img: The image on which to run PATH.
#     :param frb_object: the FRB in question.
#     :param radius: Maximum distance in arcseconds for an object to be considered as a candidate.
#     :return:
#     """
#     astm_rms = img.extract_astrometry_err()
#     a, b = frb_object.position_err.uncertainty_quadrature()
#     a = np.sqrt(a ** 2 + astm_rms ** 2)
#     b = np.sqrt(b ** 2 + astm_rms ** 2)
#     x_frb = frb.FRB(
#         frb_name=frb_object.name,
#         coord=frb_object.position,
#         DM=frb_object.dm,
#     )
#     x_frb.set_ee(
#         a=a.value,
#         b=b.value,
#         theta=0.,
#         cl=0.68,
#     )
#     #     img.load_output_file()
#     img.extract_pixel_scale()
#     filname = f'VLT_FORS2_{img.filter.band_name}'
#     config = dict(
#         max_radius=radius,
#         skip_bayesian=False,
#         npixels=9,
#         image_file=img.path,
#         cut_size=30.,
#         filter=filname,
#         ZP=img.zeropoint_best["zeropoint_img"].value,
#         deblend=True,
#         cand_bright=17.,
#         cand_separation=radius * units.arcsec,
#         plate_scale=(1 * units.pix).to(units.arcsec, img.pixel_scale_y),
#     )
#     print("P(U) ==", p_u)
#     priors = path.priors.load_std_priors()["adopted"]
#     priors["U"] = p_u
#     ass = associate.run_individual(
#         config=config,
#         #         show=True,
#         #         verbose=True,
#         FRB=x_frb,
#         prior=priors
#         #     skip_bayesian=True
#     )
#
#     p_ux = ass.P_Ux
#     print("P(U|x) ==", p_ux)
#     cand_tbl = table.QTable.from_pandas(ass.candidates)
#     p_ox = cand_tbl[0]["P_Ox"]
#     print("Max P(O|x_i) ==", p_ox)
#     print("\n\n")
#     cand_tbl["ra"] *= units.deg
#     cand_tbl["dec"] *= units.deg
#     cand_tbl["separation"] *= units.arcsec
#     cand_tbl[filname] *= units.mag
#     return cand_tbl, p_ox, p_ux

g_img = None
R_img = None
I_img = None
K_img = None

W1_img = None
W2_img = None
W3_img = None
W4_img = None

g_img_subbed = None
R_img_subbed = None
I_img_subbed = None

g_img_trimmed = None
R_img_trimmed = None
I_img_trimmed = None

textwidth = textwidths["MNRAS"]
cmaps = {
    "g_HIGH": "viridis",
    "R_SPECIAL": "plasma",  # "binary_r",
    "I_BESS": "cividis",
    "Ks": "hot",
    "W1": "viridis",
    "W2": "plasma",
    "W3": "cividis",
    "W4": "hot"
}
tick_fontsize = 12
axis_fontsize = 14
lineweight = 1.5

markers_best = [
    "^",
    "o",
    "s",
    "p",
    "h",
    "v",
    "X",
    "D",
    ">",
    "d",
    "<",
    "*",
    "P",
    "+",
    "x"
]

colours = [
    "magenta",
    "green",
    "red",
    "blue",
    "cyan",
    "purple",
    "violet",
    "darkorange",
    "gray",
    "lightblue",
    "lime",
    "gold",
    "brown",
    "maroon",
    "pink",
]

p_z_dm = None
z_p_z_dm = None
p_z_dm_best = None

l_star_table = None

band_z_mag_tables = {}
band_z_flux_tables = {}


def load_image(name: str):
    base = image.FORS2CoaddedImage(
        os.path.join(
            input_path,
            "imaging",
            f"{name}.fits"
        )
    )
    base.load_output_file()
    subbed = image.FORS2CoaddedImage(
        os.path.join(
            input_path,
            "imaging",
            f"{name}_subbed.fits"
        )
    )
    subbed.load_output_file()
    subbed.headers = copy.deepcopy(base.headers)
    subbed.zeropoint_best = base.zeropoint_best
    trimmed_path = os.path.join(
        input_path,
        f"{name}_subbed_trimmed.fits"
    )
    if os.path.exists(trimmed_path):
        trimmed = image.FORS2CoaddedImage(trimmed_path)
    else:
        frb210912 = field.Field.from_params("FRB20210912")
        left, right, bottom, top = subbed.frame_from_coord(
            centre=frb210912.frb.position,
            frame=14 * units.arcsec
        )
        trimmed = subbed.trim(left, right, bottom, top)
    trimmed.load_output_file()
    trimmed.load_data()
    # A reminder to myself: we do this instead of copying the entire header because the trimmed file necessarily has
    #   different astrometry.
    trimmed.headers[0]["PSF_FWHM"] = base.headers[0]["PSF_FWHM"]
    trimmed.headers[0]["ASTM_RMS"] = base.headers[0]["ASTM_RMS"]

    return base, subbed, trimmed


def load_images():
    global g_img, g_img_subbed, g_img_trimmed
    g_img, g_img_subbed, g_img_trimmed = load_image("FRB20210912_VLT-FORS2_g-HIGH_2021-10-04")

    global I_img, I_img_subbed, I_img_trimmed
    I_img, I_img_subbed, I_img_trimmed = load_image("FRB20210912_VLT-FORS2_I-BESS_2021-10-04")

    global R_img, R_img_subbed, R_img_trimmed
    R_img, R_img_subbed, R_img_trimmed = load_image("FRB20210912_VLT-FORS2_R-SPECIAL_2021-10-09")

    global K_img
    K_img = image.HAWKICoaddedImage(
        os.path.join(
            input_path,
            "imaging",
            "FRB20210912_VLT-HAWKI_Ks_coadded_mean.fits"
        )
    )

    global W1_img
    W1_img = image.WISECutout(
        os.path.join(
            input_path,
            "imaging",
            "3513m303_ac51-w1-int-3_ra350.7934333333333_dec-30.405430555555554_asec600.000.fits"
        )
    )
    W1_img.zeropoint()

    global W2_img
    W2_img = image.WISECutout(
        os.path.join(
            input_path,
            "imaging",
            "3513m303_ac51-w2-int-3_ra350.7934333333333_dec-30.405430555555554_asec600.000.fits"
        )
    )
    W2_img.zeropoint()

    global W3_img
    W3_img = image.WISECutout(
        os.path.join(
            input_path,
            "imaging",
            "3513m303_ac51-w3-int-3_ra350.7934333333333_dec-30.405430555555554_asec600.000.fits"
        )
    )
    W3_img.zeropoint()

    global W4_img
    W4_img = image.WISECutout(
        os.path.join(
            input_path,
            "imaging",
            "3513m303_ac51-w4-int-3_ra350.7934333333333_dec-30.405430555555554_asec600.000.fits"
        )
    )
    W4_img.zeropoint()


def load_path_results(
        path_slug: str = "trimmed_exp",
):
    path_path = os.path.join(output_path, "PATH")
    run_path_path = os.path.join(path_path, path_slug)
    bands = list(
        filter(
            lambda d: os.path.isdir(
                os.path.join(run_path_path, d)
            ),
            os.listdir(run_path_path)
        )
    )

    path_dict = {}
    for band in bands:
        band_dict = {
            "P_tbl": table.QTable.read(os.path.join(run_path_path, f"{band}_P_tbl.ecsv")),
            "candidate_tbls": {}
        }
        for i, p_u in enumerate(band_dict["P_tbl"]["P_U"]):
            band_dict["candidate_tbls"][p_u] = table.QTable.read(band_dict["P_tbl"]["cand_tbls"][i],
                                                                 format="ascii.ecsv")
        path_dict[band] = band_dict

    consolidated = {}
    c_tbl_files = list(
        filter(
            lambda f: os.path.isfile(os.path.join(run_path_path, f)) and f.startswith("candidate_table_consolidated"),
            os.listdir(run_path_path)
        )
    )
    for file in c_tbl_files:
        p_u = np.round(float(file[32:file.find(".ecsv")]), 2)
        consolidated[p_u] = table.QTable.read(os.path.join(run_path_path, file))

    return path_dict, consolidated


limits_5 = {}


def load_limits(
        path_slug: str = "trimmed",
):
    lim_path = os.path.join(output_path, "limits")
    run_lim_path = os.path.join(lim_path, path_slug)
    lim_tables = {}
    for file in os.listdir(run_lim_path):
        if file.endswith(".ecsv"):
            tbl = table.QTable.read(os.path.join(run_lim_path, file))
            fil_name = file[8 + len(path_slug):file.find(".ecsv")]
            lim_tables[fil_name] = tbl
            if path_slug == "trimmed":
                limits_5[fil_name] = tbl[4]["mag"]
    return lim_tables


def load_prospector_files():
    """
    Loads all of the Prospector models into model_dict as `craftutils.observation.sed.GordonProspectorModel` instances.

    :return: `model_dict`, structured with FRB names as keys and nested dicts as values; these dicts are structured:
        `model`: the `GordonProspectorModel` instance.
        `z`: the (real) redshift of the host.
    """
    for event in events:
        frb_name = f"FRB20{event}"
        # print(frb_name)
        if frb_name in do_not_load:
            continue
        z = None
        found = False
        for frb_this in frb_list:
            if frb_this.startswith(frb_name):
                frb_name = frb_this
                fld_this = field.Field.from_params(frb_name)
                if fld_this.frb.tns_name is not None:
                    frb_name = fld_this.frb.tns_name
                z = fld_this.frb.host_galaxy.z
                found = True
        if not found:
            for frb_this in model_dict:
                if frb_this.startswith(frb_name):
                    frb_name = frb_this
                    z = model_dict[frb_name]["z"]
                    found = True
        if not found:
            raise ValueError(f"A redshift could not be found for {frb_name}.")
        if frb_name not in model_dict:
            model_dict[frb_name] = {}

        # Check for an associated row in the photometry table
        thisrow = None
        frbdigits = frb_name[3:]
        if len(frbdigits) > 8:
            frbdigits = frbdigits[:8]
        for row in phot_tbl:
            if row["field_name"].startswith(frb_name) and f"HG{frbdigits}" in row["object_name"]:
                thisrow = row
                break
        model_dict[frb_name]["phot_tbl_row"] = thisrow

        model_flux_path = os.path.join(model_dir, f"{event}_model_spectrum_FM07.txt")
        if not os.path.isfile(model_flux_path):
            model_flux_path = os.path.join(model_dir, f"{event}_model_spectrum.txt")

        model_wavelength_path = os.path.join(model_dir, f"{event}_model_wavelengths_FM07.txt")
        if not os.path.isfile(model_wavelength_path):
            model_wavelength_path = os.path.join(model_dir, f"{event}_model_wavelengths.txt")

        observed_flux_path = os.path.join(model_dir, f"{event}_observed_spec_FM07.txt")
        if not os.path.isfile(observed_flux_path):
            observed_flux_path = os.path.join(model_dir, f"{event}_observed_spectrum_FM07.txt")

        print(frb_name, z)
        model = sed.GordonProspectorModel(
            model_flux_path=model_flux_path,
            model_wavelength_path=model_wavelength_path,
            observed_flux_path=observed_flux_path,
            observed_wavelength_path=os.path.join(model_dir, f"{event}_observed_wave_FM07.txt"),
            observed_flux_err=os.path.join(model_dir, f"{event}_observed_err_FM07.txt"),
            z=z
        )
        model_dict[frb_name]["model"] = model
        model_dict[frb_name]["z"] = z

    return model_dict


def load_magnitude_tables():
    """
    Loads the tables of magnitudes located in the output directory.
    :param output_dir:
    :return:
    """
    for frb_name in model_dict:
        tbl_path = mag_table_path(frb_name=frb_name)
        if os.path.isfile(tbl_path):
            model_dict[frb_name]["mag_table"] = table.QTable.read(tbl_path)

        tbl_path = flux_table_path(frb_name=frb_name)
        if os.path.isfile(tbl_path):
            model_dict[frb_name]["flux_table"] = table.QTable.read(tbl_path)
    return model_dict


def load_band_z_tables():
    path_tbls = os.path.join(output_path, "band_z_mag_tables", objects.cosmology.name)
    for tbl in os.listdir(path_tbls):
        path = os.path.join(path_tbls, tbl)
        if os.path.isfile(path) and path.endswith(".ecsv"):
            band_name = tbl[29:-5]
            band_z_mag_tables[band_name] = table.QTable.read(path)

    path_tbls = os.path.join(output_path, "band_z_flux_tables", objects.cosmology.name)
    for tbl in os.listdir(path_tbls):
        path = os.path.join(path_tbls, tbl)
        if os.path.isfile(path) and path.endswith(".ecsv"):
            band_name = tbl[25:-5]
            band_z_flux_tables[band_name] = table.QTable.read(path)


def load_p_z_dm():
    global p_z_dm
    dir_this = os.path.join(repo_data, "james.c.w-p-z-dm")
    if p_z_dm is None:
        p_z_dm = np.load(os.path.join(dir_this, "all_pzgdm.npy"))
    global z_p_z_dm
    if z_p_z_dm is None:
        z_p_z_dm = np.load(os.path.join(dir_this, "zvals.npy"))
    global p_z_dm_best
    if p_z_dm_best is None:
        p_z_dm_best = p_z_dm[0]


def load_l_star_table():
    l_star_path = os.path.join(
        input_path,
        "w.fong-luminosity-distributions",
        "galLF_vs_z.txt"
    )
    global l_star_table
    if os.path.isfile(l_star_path):
        l_star_table = table.QTable.read(l_star_path, format="ascii")
    else:
        raise FileNotFoundError(
            "The file `w.fong-luminosity-distributions/galLF_vs_z.txt` is not present in the input directory; please see "
            "the README for instructions on obtaining it."
        )


def mag_table_path(
        frb_name: str,
):
    """
    Simply builds a path for a particular table of magnitudes for an FRB host.

    :param frb_name: Name of the FRB.
    :return:p(z|DM)
    """
    path = os.path.join(
        output_path,
        "magnitude_tables",
        objects.cosmology.name,
        f"{frb_name}_redshifted_magnitudes.ecsv"
    )
    u.mkdir_check_nested(path, True)
    return path


def flux_table_path(
        frb_name: str,
):
    """
    Simply builds a path for a particular table of magnitudes for an FRB host.

    :param frb_name: Name of the FRB.
    :return:p(z|DM)
    """
    path = os.path.join(
        output_path,
        "flux_tables",
        objects.cosmology.name,
        f"{frb_name}_redshifted_fluxes.ecsv"
    )
    u.mkdir_check_nested(path, True)
    return path


def band_z_mag_table_path(
        band: fil.Filter,
):
    """
    Simply builds a path for a particular table of magnitudes and other properties (as a function of z) for a band.

    :param band: band object for which to load.
    :return:p(z|DM)
    """
    path = os.path.join(
        output_path,
        "band_z_mag_tables",
        objects.cosmology.name,
        f"magnitudes_and_probabilities_{band.machine_name()}.ecsv"
    )

    u.mkdir_check_nested(path, True)
    return path


def band_z_flux_table_path(
        band: fil.Filter,
):
    """
    Simply builds a path for a particular table of magnitudes and other properties (as a function of z) for a band.

    :param band: band object for which to load.
    :return:p(z|DM)
    """
    path = os.path.join(
        output_path,
        "band_z_flux_tables",
        objects.cosmology.name,
        f"fluxes_and_probabilities_{band.machine_name()}.ecsv"
    )

    u.mkdir_check_nested(path, True)
    return path


def multipath(
        imgs: list,
        path_radius: int,
        frb_object: objects.FRB,
        path_slug: str,
        offset_prior: str = "exp",
        d_p_u: float = 0.01,
        p_u_adopted: float = 0.2
):
    fildict = {
        "P_Ox_max": [],
        "P_Ux": [],
        "cand_tbls": []
    }

    path_dict = {
    }

    path_cat_dict = {}

    p_us = np.arange(0., 1. + d_p_u, 0.01)
    p_us = p_us.round(2)
    path_path = os.path.join(output_path, "PATH")
    u.mkdir_check(path_path)
    run_path_path = os.path.join(path_path, path_slug)
    u.mkdir_check(run_path_path)

    p_us_done = []

    for p_u in p_us:
        num_pu_1 = 0
        for img in imgs:
            if img.filter_name not in path_dict:
                path_dict[img.filter_name] = copy.deepcopy(fildict)
            img_path = os.path.join(run_path_path, img.filter_name)
            u.mkdir_check(img_path)
            cand_tbl, p_ox, p_ux = frb_object.probabilistic_association(
                prior_set={"U": p_u},
                offset_priors={"method": offset_prior, "scale": 0.5},
                img=img,
                config={"cand_separation": path_radius * units.arcsec}
            )
            tbl_path = os.path.join(img_path, f"PATH_{img.filter_name}_PU-{p_u}.ecsv")
            cand_tbl.write(tbl_path, overwrite=True)
            path_cat_dict[img.filter_name] = {
                "cand_tbl": cand_tbl,
                "img": img,
                "img_path": img_path
            }

            path_dict[img.filter_name]["P_Ox_max"].append(p_ox)
            path_dict[img.filter_name]["P_Ux"].append(p_ux)
            path_dict[img.filter_name]["cand_tbls"].append(tbl_path)

            if p_ux == 1.:
                num_pu_1 += 1

        path_cat = frb_object.consolidate_candidate_tables(sort_by="separation", reverse_sort=False)

        for band, this_dict in path_cat_dict.items():
            img_path = this_dict["img_path"]
            cand_tbl = this_dict["cand_tbl"]
            img = this_dict["img"]
            tbl_path = os.path.join(img_path, f"PATH_{band}_PU-{p_u}.tex")
            cand_dict = {
                "ID": [],
                r"$\theta$": [],
                "$m$": [],
                "$P^c$": [],
                "$P(O)$": [],
                "$p(x|O_i)$": [],
                "$P(O_i|x)$": [],
            }
            for row in cand_tbl:
                label = row["label"]
                i, _ = u.find_nearest(path_cat[f"label_{img.name}"], label)
                idx = path_cat[i]["id"]
                cand_dict["ID"].append(idx)
                cand_dict[r"$\theta$"].append(row["ang_size"].round(2))
                cand_dict["$m$"].append(row["mag"].round(1))
                cand_dict["$P^c$"].append((row["P_c"] * units.Unit("")).to_string(format="latex", precision=2))
                cand_dict["$P(O)$"].append((row["P_O"] * units.Unit("")).to_string(format="latex", precision=2))
                cand_dict["$p(x|O_i)$"].append((row["p_xO"] * units.Unit("")).to_string(format="latex", precision=2))
                cand_dict["$P(O_i|x)$"].append((row["P_Ox"] * units.Unit("")).to_string(format="latex", precision=2))

            cand_tbl_tex = table.QTable(cand_dict)
            cand_tbl_tex.sort("ID")
            cand_tbl_tex.write(tbl_path, overwrite=True, format="latex")

        # Find the maximum posteriors for each object and include them in the table.

        max_ps = []
        max_p_strs = []
        max_bands = []
        colnames = list(filter(lambda n: n.startswith("P_Ox"), path_cat.colnames))
        for row in path_cat:
            p_list = list(row[colnames])
            max_i = np.argmax(p_list)
            max_p = p_list[max_i]
            max_ps.append(max_p)
            max_colname = colnames[max_i]
            max_band = max_colname[max_colname.find("-") + 7:]
            max_band = f"${max_band[0]}$"
            max_p_str = units.Quantity(max_p).to_string(precision=1, format="latex")
            max_p_strs.append(max_p_str)
            max_bands.append(max_band)
        path_cat["max_pox"] = max_ps
        path_cat["max_pox_str"] = max_p_strs
        path_cat["max_log_pox"] = np.log10(path_cat["max_pox"])
        path_cat["max_band"] = max_bands

        path_cat.write(
            os.path.join(output_path, "PATH", path_slug, f"candidate_table_consolidated_PU-{p_u}.ecsv"),
            overwrite=True
        )

        path_cat["coord"] = coordinates.SkyCoord(path_cat["ra"], path_cat["dec"])

        # Generate Latex table
        path_cat_tex = table.QTable()
        path_cat_tex["ID"] = path_cat["id"]
        # path_cat["id", "ra", "dec", "separation", "P_Ox_R", "P_Ox_g", "P_Ox_I", "P_Ox_K"]
        path_cat_tex["$\\alpha$"] = path_cat["coord"].ra.to(units.hourangle).to_string(precision=2, format='latex')
        path_cat_tex["$\\delta$"] = path_cat["coord"].dec.to_string(precision=2, format='latex')
        path_cat_tex["$R_\\perp$"] = path_cat["separation"].round(1)
        path_cat_tex["Maximum $P(O_i|x)$"] = path_cat["max_pox_str"]
        path_cat_tex["Band of max $P(O_i|x)$"] = path_cat["max_band"]
        # for f in ["R", "g", "I", "K"]:
        #     path_cat_tex[f"$P_{f}(O_i|x)$"] = np.zeros(len(path_cat_tex), dtype='S50')
        #     for i, row in enumerate(path_cat):
        #         if row[f"P_Ox_{f}"] == 0.0:
        #             path_cat_tex[i][f"$P_{f}(O_i|x)$"] = "0.0"
        #         elif row[f"P_Ox_{f}"] > -998.:
        #             path_cat_tex[i][f"$P_{f}(O_i|x)$"] = units.Quantity(row[f"P_Ox_{f}"]).to_string(precision=1, format="latex")#"$" + '{:0.3e}'.format(row[f"P_Ox_{f}"]).replace("e", "\\times 10^{").replace("+", "") + "}$"
        #         else:
        #             path_cat_tex[i][f"$P_{f}(O_i|x)$"] = "--"
        path_cat_tex.sort("ID")
        path_cat_tex.write(os.path.join(output_path, "PATH", path_slug, f"path_tbl_PU-{p_u}.tex"), format="ascii.latex",
                           overwrite=True)

        p_us_done.append(p_u)

        print("n_pu_1", num_pu_1)
        if num_pu_1 == len(imgs) and p_u >= p_u_adopted:
            break

    for img in imgs:
        path_tbl = table.QTable(path_dict[img.filter_name])
        path_tbl["P_U"] = p_us_done
        path_tbl.write(os.path.join(run_path_path, f"{img.filter_name}_P_tbl.ecsv"), overwrite=True)

    # for img in imgs:
    #
    #
    #     for p_u in p_us:
    #         cand_tbl, p_ox, p_ux = frb_object.probabilistic_association(
    #             p_u=p_u,
    #             img=img,
    #             # frb_object=frb_object,
    #             radius=path_radius
    #         )

    return path_dict


def label_candidates(
        ax: plt.Axes,
        f: str,
        offset: int,
        path_cat: table.QTable
):
    """
    Draws the ID label for each candidate near to it in the image plot.

    :param ax: Axes object on which to draw.
    :param f: One-letter suffix associated with the pixel coordinates, probably a band name.
    :param offset: x pixel offset from the candidate centre to draw the letters.
    :param path_cat: astropy QTable containing the PATH candidates
    """
    # print(path_cat.colnames)
    for row in path_cat:
        offset_y = 0.
        if f"P_Ox_{f}" in row.colnames and row[f"P_Ox_{f}"] > -998.:
            if row["id"] == "G":
                offset_y += 6
            ax.text(row[f"x_{f}"] - offset, row[f"y_{f}"] + offset_y, row["id"], c="white")


def candidate_image(
        images,
        path_cat: table.QTable,
        plot_width: float = textwidth / 2,
        plot_height: float = textwidth * 1.5,
        stretch: str = "sqrt",
        n_x: int = 1,
        n_y: int = 4,
        label: bool = True,
        wspace: float = None,
        ylabelpad: float = 0.,
        offset=1.5 * units.arcsec,
        suffix: str = None,
        band_titles: bool = False
):
    """

    :param path_cat:
    :param plot_width:
    :param plot_height:
    :param stretch:
    :param n_x:
    :param n_y:
    :param label:
    :param wspace:
    :param ylabelpad:
    :param offset:
    :param output_path:
    :param images:
    :param suffix:
    :return:
    """
    latex_setup()
    frame = path_cat["separation"].max() + 1 * units.arcsec
    fig = plt.figure(figsize=(plot_width, plot_height))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_frame_on(False)
    ax.tick_params(left=False, right=False, top=False, bottom=False)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    # ax.set_ticks([])
    # ax.set_aspect("equal")
    ax.set_xlabel(
        "Right Ascension (J2000)",
        labelpad=30,
        fontsize=axis_fontsize
    )
    ax.set_ylabel(
        "Declination (J2000)",
        labelpad=ylabelpad,
        fontsize=axis_fontsize,
        rotation=-90
    )
    ax.yaxis.set_label_position("right")

    for i, img in enumerate(images):
        n = i + 1
        f = img.name
        ax, fig, _ = fld.plot_host(
            img=img,
            fig=fig,
            frame=frame,
            show_frb=True,
            n_x=n_x,
            n_y=n_y,
            n=n,
            imshow_kwargs=dict(
                cmap=cmaps[img.filter_name]
            ),
            normalize_kwargs=dict(
                stretch=stretch
            ),
            frb_kwargs=dict(
                edgecolor="black",
                lw=lineweight
            )
        )
        #     ax.text(0.5, 0.5, n, transform=ax.transAxes, colour="black")
        ra, dec = ax.coords
        ra.set_axislabel(" ")
        ra.set_ticks(spacing=0.5 * units.hourangle / 3600)
        dec.set_axislabel(" ")
        dec.set_ticklabel(fontsize=tick_fontsize)

        if n <= n_x * (n_y - 1):
            #         ra.set_ticks_visible(False)
            ra.set_ticklabel_visible(False)
        else:
            ra.set_ticklabel(fontsize=tick_fontsize)

        if n % n_y or n_x == 1:
            #         ra.set_ticks_visible(False)
            dec.set_ticklabel(fontsize=tick_fontsize)
        else:
            dec.set_ticklabel_visible(False)

        ax.set_aspect("equal")

        path_cat[f"x_{f}"], path_cat[f"y_{f}"] = img.world_to_pixel(path_cat["coord"])
        #     plt.scatter(path_cat[f"x_{f}"], path_cat[f"y_{f}"], marker="x", c="white")
        if label:
            label_candidates(ax=ax, f=f, offset=img.pixel(offset).value, path_cat=path_cat)
        if band_titles:
            ax.set_title(f"{img.instrument.formatted_name}, {img.filter.formatted_name}")
    hspace = 0.1
    if band_titles:
        hspace = 0.5
    fig.subplots_adjust(hspace=0.1, wspace=wspace)
    filename = f"imaging_{n_x}x{n_y}_{stretch}"
    if suffix:
        filename += f"_{suffix}"
    if label:
        filename += "_labelled"
    if band_titles:
        filename += "_bands"

    output_this = os.path.join(output_path, "imaging")
    u.mkdir_check(output_this)
    fig.savefig(os.path.join(output_this, filename + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_this, filename + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return ax, fig


def mag_from_model_flux(
        model: sed.SEDModel,
        z_shift: float = 1.0,
        bands: List[fil.Filter] = bands_default,
):
    """
    Uses a set of bandpasses to derive observed magnitudes for an FRB host placed at a counterfactual redshift.

    :param model: The `SEDModel` object to shift and measure.
    :param z_shift: Redshift at which to place the model.
    :param bands: List or other iterable of `craftutils.observation.filters.Filter` objects through which to observe
        the redshifted galaxy.
    :return: dict of AB magnitudes in the given bands, with keys being the band names and values being the magnitudes.
    """
    mags = {}
    shifted_model = model.move_to_redshift(z_new=z_shift)

    fluxes = {}

    for band in bands:
        band_name = band.machine_name()
        m_ab_shift = shifted_model.magnitude_AB(band=band)
        mags[band_name] = m_ab_shift

        fluxes[band_name] = shifted_model.flux_in_band(band=band)

    return mags, fluxes


def get_mags_shifted(
        model: sed.SEDModel,
        z_min: float = 0.01,
        z_max: float = 2.0,
        n: int = 10,
        bands: List[fil.Filter] = bands_default
):
    """
    Applies mag_from_model_flux() to a model over a range of redshifts.

    :param model: The `SEDModel` object to shift and measure.
    :param z_min: Minimum redshift to take measurements at.
    :param z_max: Maximum redshift to take measurements at.
    :param n: number of redshifts to take measurements at.
    :param bands: List or other iterable of `craftutils.observation.filters.Filter` objects through which to observe
        the redshifted galaxy.
    :param bands: List or other iterable of `craftutils.observation.filters.Filter` objects through which to observe
        the redshifted galaxy.
    :return: `astropy.table.QTable` of the derived magnitudes, with columns corresponding to bands
        and redshift.
    """
    mags = {}
    fluxes = {}
    z_host = model.z
    for band in bands:
        band_name = band.machine_name()
        mags[band_name] = []
        fluxes[band_name] = []
    # Construct array of redshifts.
    zs = list(np.linspace(z_min, z_max, n))
    zs.append(z_host)
    zs.sort()
    for z in zs:
        # Get magnitudes at this redshift.
        mags_this, flux_this = mag_from_model_flux(
            model=model,
            bands=bands,
            z_shift=z,
        )
        for band in bands:
            band_name = band.machine_name()
            mags[band_name].append(
                mags_this[band_name]
            )
            fluxes[band_name].append(
                flux_this[band_name]
            )
    mags["z"] = zs
    fluxes["z"] = zs
    return table.QTable(mags), table.QTable(fluxes)


def set_plot_properties(frbs: list = None):
    if frbs is None:
        frbs = []
    else:
        frbs = list(set(frbs))
        frbs.sort()

    for i, frb in enumerate(frbs):
        model_dict[frb]["marker"] = markers_best[i]
        model_dict[frb]["colour"] = colours[i]

    frbs_other = list(set(model_dict.keys()) - set(frbs))
    n = len(frbs_other)  # - len(colours)
    colours_rb = list(cm.rainbow(np.linspace(0, 1, n)))
    print()
    markers_all = list(Line2D.filled_markers)
    for i, frb in enumerate(frbs_other):
        model_dict[frb]["marker"] = markers_all[i]
        model_dict[frb]["colour"] = colours_rb[i]


def pdf_shading(
        ax,
        cmap: str = "binary",
        alpha: float = 1.,
        pdf: np.ndarray = None,
        z: np.ndarray = None
):
    if pdf is None:
        pdf = p_z_dm_best
    if z is None:
        z = z_p_z_dm
    c = ax.pcolor(
        z,
        ax.get_ylim(),
        pdf[np.newaxis] * np.ones((2, len(pdf))),
        cmap=cmap,
        alpha=alpha,
        zorder=-np.inf
    )
    return ax, c


def get_band_mags(band: fil.Filter):
    """
    Transforms the model table information to put the magnitudes of each FRB host, in a particular band, as columns.
    z is also provided as a column.

    :param band: A Filter object corresponding to the particular band.
    :return: Table
    """
    mag_dict = {}
    flux_dict = {}
    bn = band.machine_name()
    for frb_name in model_dict:
        mag_tbl = model_dict[frb_name]["mag_table"]
        mag_dict[frb_name] = mag_tbl[bn]

        flux_tbl = model_dict[frb_name]["flux_table"]
        flux_dict[frb_name] = flux_tbl[bn]
    mag_dict["z"] = mag_tbl["z"]
    flux_dict["z"] = flux_tbl["z"]
    return table.QTable(mag_dict), table.QTable(flux_dict)


def band_mag_table(band: fil.Filter):
    """
    Generates a table of magnitudes, for each FRB host in the given band, as a function of redshift and performs some
    other calculations.

    :param band: A Filter object corresponding to the particular band.
    :return:
    """
    latex_setup()
    mag_tbl, flux_tbl = get_band_mags(band)
    band_name = band.name

    values = {}

    # Load limits and get the value for 5-sigma
    limits = load_limits()
    if band_name in limits:
        limit = limits_5[band_name].value
    else:
        limits = load_limits("WISE")
        limit = limits[band_name][4]["mag"].value

    # Set up some columns
    mag_tbl.add_column(np.zeros(len(mag_tbl), dtype=int), name="n>lim")
    mag_tbl.add_column(np.zeros(len(mag_tbl), dtype=int), name="n<lim")
    # Interpolate p(z|DM) to match the table z.
    mag_tbl["p(z|DM)"] = np.interp(
        x=mag_tbl["z"],
        xp=z_p_z_dm,
        fp=p_z_dm_best
    )

    # Gather FRB names
    columns = tuple(filter(lambda c: c.startswith("FRB"), mag_tbl.colnames))
    n_frbs = len(columns)
    # Sum the number of FRB hosts above and below the limit,
    # and write to the model_dict the redshift at which each host becomes fainter than the limit.

    avg_dict = {
        "mean": {},
        "median": {},
        "mean_flux": {}
    }

    for colname in columns:
        mag_tbl["n>lim"] += mag_tbl[colname] > limit
        mag_tbl["n<lim"] += mag_tbl[colname] < limit
        model_dict[colname][f"z_lost_{band.band_name}"] = np.min(mag_tbl["z"][mag_tbl[colname] > limit])

    values["n_visible"] = {}

    print(f"\n In {band_name}:")
    for key_z in (0.5, 1., 1.5, 2., 2.5, 3.):
        i_z, _ = u.find_nearest(mag_tbl["z"], key_z)
        i_z += 1
        values["n_visible"][key_z] = mag_tbl["n<lim"][i_z]
        print(f"At z={key_z}, {values['n_visible'][key_z]} / {n_frbs} visible")

    # Record the mean and median magnitudes at each z.
    mag_tbl["mean"] = np.zeros(len(mag_tbl))
    mag_tbl["median"] = np.zeros(len(mag_tbl))
    for row in mag_tbl:
        row["mean"] = np.mean(list(row[columns]))
        row["median"] = np.median(list(row[columns]))
    avg_dict["mean"][f"z_lost_{band.band_name}"] = np.min(mag_tbl["z"][mag_tbl["mean"] > limit])
    avg_dict["median"][f"z_lost_{band.band_name}"] = np.min(mag_tbl["z"][mag_tbl["median"] > limit])

    flux_tbl["mean"] = np.zeros(len(flux_tbl))
    flux_tbl["median"] = np.zeros(len(flux_tbl))
    for row in flux_tbl:
        row["mean"] = np.mean(list(map(lambda v: v.value, row[columns])))  # * row[columns[0]].unit
        row["median"] = np.median(list(map(lambda v: v.value, row[columns])))  # * row[columns[0]].unit
    flux_tbl["mean"] *= flux_tbl[columns[0]].unit
    flux_tbl["median"] *= flux_tbl[columns[0]].unit
    ab_flux_band = band.ab_flux()
    flux_tbl["mean_mag"] = -2.5 * np.log10(flux_tbl["mean"] / ab_flux_band)
    avg_dict["mean_flux"][f"z_lost_{band.band_name}"] = np.min(mag_tbl["z"][flux_tbl["mean_mag"] > limit])

    # Calculate P(U|z) as the fraction of hosts that are unseen at a given redshift.
    mag_tbl["P(U|z)"] = mag_tbl["n>lim"] / n_frbs
    # Some extra values
    mag_tbl["d_L"] = cosmology.WMAP9.luminosity_distance(mag_tbl["z"])
    mag_tbl["mu"] = ph.distance_modulus(mag_tbl["d_L"])

    # Calculate P(U) and, at the same time, get p(z|U,DM)
    curve = mag_tbl["P(U|z)"] * mag_tbl["p(z|DM)"]
    p_u = np.trapz(
        y=curve,
        x=mag_tbl["z"]
    )
    mag_tbl["p(z|U,DM)"] = curve / p_u
    mag_tbl["P(U|z) * p(z|DM)"] = curve

    print("Peak p(z|U,DM) at z =", mag_tbl["z"][np.argmax(mag_tbl["p(z|U,DM)"])])
    print("P(U) =", p_u)

    dirpath = os.path.join(output_path, "distributions", objects.cosmology.name)
    u.mkdir_check_nested(dirpath, False)

    leg_x = 1.13
    fig, ax = plt.subplots(figsize=(textwidth, textwidth / 4))

    ax_pdf = ax.twinx()
    ax_pdf.set_ylabel("Host fraction", rotation=-90, labelpad=35, fontsize=axis_fontsize)
    ax_pdf.tick_params(right=False,labelright=False)
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis="y", labelright=True, labelsize=tick_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)

    # Do some plotting
    ax.plot(
        mag_tbl["z"],
        mag_tbl["P(U|z)"],
        label="$P(U|z) = N_\mathrm{unseen}(z)/N_\mathrm{hosts}$",  # = \dfrac{N_\mathrm{unseen}(z)}{N_\mathrm{hosts}}$"
        lw=2,
        c="cyan"
    )

    ax.set_xlabel("$z$")
    ax.set_xlim(0., 5)
    fig.savefig(
        os.path.join(dirpath, f"probability_{objects.cosmology.name}_{band_name}_steps_only.pdf"),
        bbox_inches="tight"
    )
    fig.savefig(
        os.path.join(dirpath, f"probability_{objects.cosmology.name}_{band_name}_steps_only.png"),
        bbox_inches="tight",
        dpi=200
    )

    ax.set_ylabel("Probability density", fontsize=axis_fontsize)

    ax.plot(
        mag_tbl["z"],
        mag_tbl["p(z|DM)"],
        label="$p(z|\mathrm{DM})$",
        lw=2,
        c="purple"
    )
    ax.legend(
        loc=(leg_x, 0),
    )
    fig.savefig(
        os.path.join(dirpath, f"probability_{objects.cosmology.name}_{band_name}_combined.pdf"),
        bbox_inches="tight"
    )
    fig.savefig(
        os.path.join(dirpath, f"probability_{objects.cosmology.name}_{band_name}_combined.png"),
        bbox_inches="tight",
        dpi=200
    )

    ax.plot(
        mag_tbl["z"],
        mag_tbl["p(z|U,DM)"],
        label="$p(z|U,\mathrm{DM})$",  # = \dfrac{P(U|z)p(z|\mathrm{DM,etc.})}{P(U)}$"
        lw=2,
        c="green"
    )
    ax.legend(
        loc=(leg_x, 0)
    )
    fig.savefig(
        os.path.join(dirpath, f"probability_{objects.cosmology.name}_{band_name}.pdf"),
        bbox_inches="tight"
    )
    fig.savefig(
        os.path.join(dirpath, f"probability_{objects.cosmology.name}_{band_name}.png"),
        bbox_inches="tight",
        dpi=200
    )

    # Add normal approximation to plot
    gauss_init = models.Gaussian1D(mean=1.2, amplitude=1.75, stddev=1.)
    fitter = fitting.LevMarLSQFitter()
    gauss_fit = fitter(gauss_init, x=mag_tbl["z"], y=mag_tbl["P(U|z) * p(z|DM)"])
    p_u_gauss = np.trapz(
        y=gauss_fit(mag_tbl["z"]),
        x=mag_tbl["z"]
    )
    mag_tbl["p(z|U,DM) gauss"] = gauss_fit(mag_tbl["z"]) / p_u_gauss

    print()
    print("Normal approximation")
    print("Peak p(z|U,DM) at z =", gauss_fit.mean.value)
    print("P(U) =", p_u_gauss)

    ax.plot(
        mag_tbl["z"],
        mag_tbl["p(z|U,DM) gauss"],
        label="$p(z|U,\mathrm{DM})$, Gaussian fit",  # = \dfrac{P(U|z)p(z|\mathrm{DM,etc.})}{P(U)}$"
        lw=2,
        c="darkorange",
        ls=":"
    )
    ax.legend(
        loc=(leg_x, 0),
        fontsize=tick_fontsize
    )
    fig.savefig(
        os.path.join(dirpath, f"probability_{objects.cosmology.name}_{band_name}_gaussian.pdf"),
        bbox_inches="tight"
    )
    fig.savefig(
        os.path.join(dirpath, f"probability_{objects.cosmology.name}_{band_name}_gaussian.png"),
        bbox_inches="tight", dpi=200
    )

    plt.close(fig)

    values["P(U)"] = p_u
    values["P(U)_gaussian"] = p_u_gauss
    values["p(z|U,DM)_gaussian"] = {
        "mu": gauss_fit.mean.value,
        "sigma": gauss_fit.stddev.value,
        "A": gauss_fit.amplitude.value / p_u_gauss
    }

    p.save_params(os.path.join(dirpath, f"values_{band_name}.yaml"), values)

    mag_tbl.write(band_z_mag_table_path(band), overwrite=True)
    flux_tbl.write(band_z_flux_table_path(band), overwrite=True)

    global band_z_mag_tables
    band_z_mag_tables[band.machine_name()] = mag_tbl

    output_dicts = os.path.join(output_path, "frb_properties")
    u.mkdir_check(output_dicts)
    for frb in model_dict:
        model_dict_write = model_dict[frb].copy()
        keys_pop = []
        for key in model_dict_write:
            if isinstance(model_dict_write[key], (table.Table, sed.SEDModel, table.Row)):
                keys_pop.append(key)
        for key in keys_pop:
            model_dict_write.pop(key)
        p.save_params(os.path.join(output_dicts, f"{frb}_properties.yaml"), model_dict_write)

    p.save_params(os.path.join(output_dicts, f"{band.name}_avg_properties.yaml"), avg_dict)

    return mag_tbl


def magnitude_redshift_plot(
        band,
        frbs=None,
        draw_lstar=False,
        suffix="",
        draw_observed_phot=False,
        kwargs_lim={},
        path_lim: bool = False,
        kwargs_path_lim={},
        textwidth_factor=1.,
        do_pdf_panel: bool = False,
        do_pdf_shading: bool = False,
        do_legend: bool = True,
        legend_frbs: list = None,
        path_slug: str = "trimmed",
        do_mean=False,
        do_median=False,
        do_other_photometry=False,
        grey_lines=False,
        n_panels: int = 1,
):
    ident = f"{suffix}_{textwidth_factor}tw"
    if do_legend:
        ident += "_legend"
    if draw_lstar:
        ident += "_lstar"
    if do_pdf_shading:
        ident += "_pdfshading"
    if do_pdf_panel:
        ident += "_pdfpanel"
    if draw_observed_phot:
        ident += "_obsphot"
    if path_lim:
        ident += "_pathlim"
    if do_mean:
        ident += "_mean"
    if do_median:
        ident += "_median"
    if do_other_photometry:
        ident += "_all-photom"
    if grey_lines:
        ident += "_grey-lines"
    print("Plotting", ident, "...")

    if frbs is None:
        frbs = list(model_dict.keys())

    if do_pdf_panel:
        n_panels_ = n_panels + 1
        heights = [1] + [3] * n_panels
    else:
        n_panels_ = n_panels
        heights = [3] * n_panels

    if isinstance(grey_lines, bool):
        grey_lines = [grey_lines] * n_panels
    if isinstance(do_median, bool):
        do_median = [do_median] * n_panels
    if not isinstance(frbs[0], list):
        frbs = [frbs] * n_panels
    if not isinstance(band, list):
        band = [band] * n_panels

    fig = plt.figure(figsize=(textwidth * textwidth_factor, textwidth_factor * textwidth * 0.4 * n_panels))
    gs = fig.add_gridspec(nrows=n_panels_, ncols=1, height_ratios=heights)

    load_band_z_tables()

    # Plot the distributions panel in the figure
    if do_pdf_panel:
        band_this = band[0]

        band_name = band_this.machine_name()
        band_z_tbl = band_z_mag_tables[band_name]

        ax_pdf = fig.add_subplot(gs[0, 0])

        ax_pdf.plot(
            band_z_tbl["z"],
            band_z_tbl["p(z|DM)"],
            lw=2,
            c="purple"
        )
        ax_pdf.plot(
            band_z_tbl["z"],
            band_z_tbl["p(z|U,DM) gauss"],
            lw=2,
            c="darkorange",
            ls=":"
        )
        ax_pdf.set_ylabel("Probability\ndensity", fontsize=axis_fontsize)
        ax_pdf.set_xlim(0.01, 2.)
        ax_pdf.tick_params(bottom=False, labelsize=tick_fontsize)
        ax_pdf.xaxis.set_ticks([])

    if do_pdf_panel:
        n = 1
    else:
        n = 0

    for panel in range(n, n_panels_):

        if do_pdf_panel:
            index = panel - 1
        else:
            index = panel

        band_this = band[index]

        band_name = band_this.machine_name()
        limits = load_limits(path_slug)
        limit = limits[band_this.name]["mag"][4].value
        band_z_tbl = band_z_mag_tables[band_name]

        grey = grey_lines[index]
        median = do_median[index]

        ax = fig.add_subplot(gs[panel, 0])
        fig.subplots_adjust(hspace=0.)

        # Limit line
        kwargs_lim_def = dict(c="black", lw=2, ls=":")
        kwargs_lim_def.update(kwargs_lim)
        ax.plot(
            (0.0, 4.),
            (limit, limit),
            **kwargs_lim_def
        )
        if path_lim:
            path_limit = faintest[band_name]
            kwargs_path_def = dict(c="black", lw=2, ls="--")
            kwargs_path_def.update(kwargs_path_lim)
            ax.plot((0.0, 4.), (path_limit, path_limit), label="$m_\mathrm{faintest}$", **kwargs_path_def)

        frbs_ = frbs[index]

        frbs_.sort()

        for n, frb in enumerate(frbs_):

            model_dict_frb = model_dict[frb]
            i, _ = u.find_nearest(band_z_tbl["z"], model_dict_frb["z"])
            # Draw the line
            if grey:
                colour = "black"
                alpha = 0.1
                lw = 5
            else:
                colour = model_dict_frb["colour"]
                alpha = 1.
                lw = 1.5
            ax.plot(
                band_z_tbl["z"],
                band_z_tbl[frb],
                color=colour,  # colour[n],
                alpha=alpha,
                zorder=-1,
                lw=lw
            )
            # Draw the photometry data point
            if not grey:
                ax.scatter(
                    model_dict_frb["z"],
                    band_z_tbl[frb][i],
                    color=model_dict_frb["colour"],
                    alpha=1.,
                    marker=model_dict_frb["marker"],
                    edgecolors="black",
                    label=frb.replace("FRB", "FRB\,"),
                    zorder=1
                )
            # Pulls photometry from the host table and overplots it, more of a debug thing
            if draw_observed_phot:
                thisrow = None
                frbdigits = frb[3:]
                if len(frbdigits) > 8:
                    frbdigits = frbdigits[:8]
                for row in phot_tbl:
                    #                 print(frbdigits, row["field_name"], row["object_name"])
                    if row["field_name"].startswith(frb) and f"HG{frbdigits}" in row["object_name"]:
                        thisrow = row
                        break
                #             print(thisrow)
                if thisrow is not None:
                    colstring = f"mag_best_{band_this.instrument.name}_{band_this.name.replace('_', '-')}"
                    colstring_err = colstring + "_err"
                    if thisrow[colstring] > -999. * units.mag and thisrow[colstring_err] > -999. * units.mag:
                        ax.scatter(model_dict_frb["z"], thisrow[colstring].value, marker="x", color="black")

        ax.invert_yaxis()
        ax.set_xlabel("$z$", fontsize=axis_fontsize)
        ax.set_ylabel(f"$m_\mathrm{{{band_this.band_name}}}$", fontsize=axis_fontsize)

        if do_mean:
            band_z_flux_tbl = band_z_flux_tables[band_name]
            if grey:
                colour = "red"
                lw = 2
            else:
                colour = "grey"
                lw = 3
            ax.plot(
                band_z_flux_tbl["z"],
                band_z_flux_tbl["mean_mag"],
                color=colour,
                zorder=1,
                lw=lw,
                ls="--"
            )
        if median:
            if grey:
                colour = "red"
                lw = 2
            else:
                colour = "grey"
                lw = 3
            ax.plot(
                band_z_tbl["z"],
                band_z_tbl["median"],
                color=colour,
                zorder=1,
                lw=lw,
                ls=":"
            )
        if do_other_photometry:
            other_frbs = list(set(model_dict.keys()) - set(frbs_))
            for n, frb in enumerate(other_frbs):
                model_dict_frb = model_dict[frb]
                i, _ = u.find_nearest(band_z_tbl["z"], model_dict_frb["z"])
                # Draw the photometry data point
                ax.scatter(
                    model_dict_frb["z"],
                    band_z_tbl[frb][i],
                    color="black",
                    alpha=1.,
                    marker="x",
                    label=frb.replace("FRB", "FRB\,"),
                    zorder=0
                )

        # Draw L* fraction lines
        if draw_lstar:
            if l_star_table is None:
                load_l_star_table()
            ax.plot(l_star_table["z"], l_star_table["m_r(L*)"], c="grey", ls=":", label="L*")
            ax.plot(l_star_table["z"], l_star_table["m_r(0.1L*)"], c="grey", ls="--", label="0.1 L*")
            ax.plot(l_star_table["z"], l_star_table["m_r(0.01L*)"], c="grey", ls="-.", label="0.01 L*")

        if do_pdf_shading:
            ax, c = pdf_shading(ax, alpha=1.)  # , pdf=band_table["p(z|DM)"], z=band_table["z"])

        ax.set_xlim(min(band_z_tbl["z"]), 2.)
        ax.set_ylim(32, 10)
        if textwidth_factor < 1.:
            yticks = np.arange(12, 32, 4)
        else:
            yticks = np.arange(12, 32, 2)
        ax.set_yticks(yticks)
        if panel == n_panels_ - 1:
            ax.set_xticks(np.arange(0.25, 2.25, 0.25))
        else:
            ax.tick_params(bottom=False, labelsize=tick_fontsize)
            ax.xaxis.set_ticks([])

        ax.tick_params(labelsize=tick_fontsize)

        if do_pdf_shading:
            fig.colorbar(c, ax=ax, location="bottom")

        #     plt.tight_layout()

    if do_legend:
        if legend_frbs is None:
            ax.legend(
                loc=(1.03, 0)
            )
        else:
            legend_elements = []
            legend_frbs = list(set(legend_frbs))
            legend_frbs.sort()
            for frb in legend_frbs:
                legend_elements.append(
                    Line2D(
                        [0], [0],
                        marker=model_dict[frb]["marker"],
                        markeredgecolor="black",
                        color=model_dict[frb]["colour"],
                        label=frb,
                        markerfacecolor=model_dict[frb]["colour"],
                        # markersize=10
                    )
                )
            ax.legend(
                handles=legend_elements,
                loc=(1.03, 0)
            )

    subdir = os.path.join(output_path, "z_magnitude_diagrams")
    u.mkdir_check(subdir)
    subdir = os.path.join(subdir, ident)
    u.mkdir_check(subdir)
    fig.savefig(
        os.path.join(
            subdir,
            f"m{band[0].band_name}_z_{ident}.pdf"),
        bbox_inches='tight'
    )
    fig.savefig(
        os.path.join(
            subdir,
            f"m{band[0].band_name}_z_{ident}.png"),
        bbox_inches='tight', dpi=200
    )
    plt.close(fig)

    return fig
