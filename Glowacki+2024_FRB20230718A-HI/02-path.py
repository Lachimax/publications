#!/usr/bin/env python
# Code by Lachlan Marnoch, 2023-2024

import os
import copy

import numpy as np
import matplotlib.pyplot as plt

from astropy import units, table
from astropy.coordinates import SkyCoord
from astropy.io import fits

import craftutils.utils as u
import craftutils.params as p
from craftutils.observation import objects, field, instrument, image, sed
import craftutils.astrometry as astm
import craftutils.plotting as pl

import lib

description = """
Performs PATH analysis.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    output_dict = {}

    fld = lib.fld

    pl.latex_setup()

    decam = lib.decam
    r_decam = lib.r_decam

    g_img = image.SurveyCutout(lib.cutout_path("g"))
    g_img.filter = lib.g_decam
    r_img = image.SurveyCutout(lib.cutout_path("r"))
    r_img.filter = lib.r_decam
    z_img = image.SurveyCutout(lib.cutout_path("z"))
    z_img.filter = lib.z_decam

    # Try to estimate the zeropoint of the r-band image.

    # Start by pulling the catalogue

    # Set the initial zeropoint to zero.
    zp_this = 0.  # zp_per_sec

    r_img.headers[0]["EXPTIME"] = 1.0
    r_img.headers[0]["ZP"] = zp_this

    r_img.instrument = decam
    r_img.add_zeropoint(
        catalogue="calib",
        zeropoint=zp_this,
        zeropoint_err=0.,
        extinction=0.,
        extinction_err=0.,
        airmass=0.,
        airmass_err=0.
    )
    r_img.zeropoint_best = r_img.zeropoints["calib"]["self"]
    r_img.write_fits_file()

    # Increase npixels (argument passed to image segmentation) to avoid breaking up the host.
    npixels = 36

    print()
    print("Doing initial calibration run.")
    path_tbl_calib, p_ox, p_ux, config, priors = fld.frb.probabilistic_association(
        img=r_img,
        priors={"U": 0.1},
        max_radius=1 * units.arcmin,
        config={
            "cand_bright": -20,
            "npixels": npixels
        },
        associate_kwargs={"extinction_correct": False}
    )

    # Match derived catalogue to DECaPS catalogue

    decaps_cat = table.QTable.read(lib.cat_path())
    decaps_cat = decaps_cat[np.isfinite(decaps_cat["mean_mag_r"])]

    plt.scatter(path_tbl_calib["ra"], path_tbl_calib["dec"], marker="x", label="Image")
    plt.scatter(decaps_cat["ra"], decaps_cat["dec"], marker="x", label="Catalogue")
    plt.legend()
    plt.savefig(os.path.join(lib.output_path, "catalogue_all.pdf"))
    plt.close()

    path_matched_all, decaps_matched_all, distance_all = astm.match_catalogs(
        path_tbl_calib, decaps_cat,
        "ra", "dec",
        tolerance=0.5 * units.arcsec
    )

    plt.scatter(path_matched_all["ra"], path_matched_all["dec"], marker="x", label="Image")
    plt.scatter(decaps_matched_all["ra"], decaps_matched_all["dec"], marker="x", label="Catalogue")
    plt.legend()
    plt.savefig(os.path.join(lib.output_path, "catalogue_matches.pdf"))
    plt.close()

    plt.scatter(decaps_matched_all["mean_mag_r"], path_matched_all["mag"], marker="x")
    plt.savefig(os.path.join(lib.output_path, "zeropoint.pdf"))
    plt.close()

    zp_derived = np.nanmean(decaps_matched_all["mean_mag_r"].value - path_matched_all["mag"].value)
    zp_derived_err = np.nanstd(decaps_matched_all["mean_mag_r"].value - path_matched_all["mag"].value) / np.sqrt(
        len(decaps_matched_all))

    print("Zeropoint:", zp_derived, "+/-", zp_derived_err)

    # Set the true zeropoint.
    r_img.headers[0]["ZP"] = zp_derived
    output_dict["zeropoint"] = zp_derived
    output_dict["zeropoint_err"] = zp_derived_err

    r_img.filter = r_decam
    r_img.instrument = decam
    r_img.add_zeropoint(
        catalogue="calib",
        zeropoint=zp_derived,
        zeropoint_err=zp_derived_err,
        extinction=0.,
        extinction_err=0.,
        airmass=0.,
        airmass_err=0.
    )
    r_img.zeropoint_best = r_img.zeropoints["calib"]["self"]
    r_img.write_fits_file()

    print("Estimating P(U) using the Gordon+2023 Prospector sample")
    properties = p.load_params(os.path.join(output_path, "frb_properties.yaml"))
    ext = properties["extinction_fm07"]["r"]

    test_coord = SkyCoord("8h32m38.46s -40d27m21.59")
    limits = r_img.test_limit_location(
        test_coord,
        ap_radius=4 * units.arcsec,
    )
    limits["mag_ext"] = limits["mag"] - ext
    limit = limits["mag_ext"][4]
    print("5-sigma limit:", limit)
    g23 = sed.SEDSample.from_file(os.path.join(lib.param_path, "Gordon+2023", "Gordon+2023.yaml"))
    g23.z_displace_sample(
        bands=[r_decam],
        z_max=0.1,
        n_z=250,
        save_memory=True
    )
    fld.frb.read_p_z_dm(os.path.join(lib.data_path, "230718_pzgdm.npz"))
    pu_vals, pu_tbl, z_lost = g23.probability_unseen(
        limit=limit,
        obj=fld.frb,
        output=output_path,
        show=True,
        band=r_decam,
        exclude=[],
        plot=True
    )
    pu_tbl.write(
        os.path.join(lib.output_path, "PU.ecsv"),
        overwrite=True
    )

    p_u_est = pu_vals["P(U)"]["step"]
    p_u_adopt = 0.1
    pu_vals.pop('p(z|DM,U)')
    pu_vals["z_lost"] = z_lost

    p.save_params(
        os.path.join(output_path, f"PU_results.yaml"),
        output_dict,
    )

    for p_u in (p_u_est, p_u_adopt):
        print(f"Doing 'final' PATH run with P(U) == {p_u}.")
        path_tbl_ext, p_ox, p_ux, config, priors = fld.frb.probabilistic_association(
            img=r_img,
            priors={"U": p_u},
            max_radius=1 * units.arcmin,
            config={"cand_bright": 20.3, "npixels": npixels},
            associate_kwargs={"extinction_correct": True},
            show=False
        )

        print(f"Doing final PATH run with P(U) == {p_u}.")
        fld.frb.probabilistic_association(
            img=r_img,
            priors={"U": p_u},
            max_radius=1 * units.arcmin,
            config={"cand_bright": 20.3, "npixels": npixels},
            associate_kwargs={"extinction_correct": True},
            show=False
        )

        output_dict["config"] = config
        output_dict["priors"] = priors
        output_dict["P(U|x)"] = p_ux
        output_dict["max_P(O|x)"] = p_ox

        p_u_rounded = np.round(p_u, 3)

        path_tbl_ext.write(
            os.path.join(lib.output_path, f"PATH_candidates_pu_{p_u_rounded}.ecsv"),
            overwrite=True
        )
        path_tbl_ext.write(
            os.path.join(lib.output_path, f"PATH_candidates_pu_{p_u_rounded}.csv"),
            overwrite=True
        )

        p.save_params(
            os.path.join(output_path, f"path_results_pu_{p_u_rounded}.yaml"),
            output_dict,
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
