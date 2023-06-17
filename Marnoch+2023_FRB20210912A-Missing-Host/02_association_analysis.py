#!/usr/bin/env python
# Code by Lachlan Marnoch, 2023

import os

import numpy as np
import matplotlib.pyplot as plt

from astropy import coordinates, table, units

import craftutils.observation.objects as objects
import craftutils.params as p
import craftutils.utils as u
from craftutils.plotting import textwidths

import lib

description = """
Derives results related to the PATH (Probabilistic Association of Transients; Aggarwal et al 2021) runs. 
Generates Figure 2.
Must be run AFTER `01_association.py`.
"""


def main(
        output_dir: str,
        input_dir: str,
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)
    # Set up field object
    fld = lib.fld
    frb210912 = fld.frb

    values = {}

    # Exponential
    path_dir = os.path.join(lib.output_path, "PATH")
    for run in filter(
            lambda d: os.path.isdir(os.path.join(path_dir, d)),
            os.listdir(path_dir)
    ):
        run_dict = {}
        path_dict, consolidated = lib.load_path_results(run)

        for band in lib.bands_default:
            band_dict = path_dict[band.name]
            p_tbl = band_dict["P_tbl"]
            band_dict_val = {
                "P(U|x)": {},
                "P(O_i|x)_max": {},
                "P(O_i|x)_sum": {}
            }
            p_us = (0.01, 0.2)
            for p_u in p_us:
                i = int(p_u * 100)
                row = p_tbl[i]
                band_dict_val["P(U|x)"][p_u] = row["P_Ux"]
                band_dict_val["P(O_i|x)_max"][p_u] = row["P_Ox_max"]
                cand_tbl = band_dict["candidate_tbls"][p_u]

                band_dict_val["P(O_i|x)_sum"][p_u] = np.nansum(cand_tbl["P_Ox"])

            run_dict[band.band_name] = band_dict_val

        values[run] = run_dict

    p.save_params(os.path.join(path_dir, "summary.yaml"), values)

    path_dict, consolidated = lib.load_path_results("trimmed_exp")
    path_cat_02 = consolidated[0.2]
    path_cat_02["coord"] = coordinates.SkyCoord(path_cat_02["ra"], path_cat_02["dec"])

    # Generate object yaml file (for craft-optical-pipeline use)
    yaml_dict = {}

    for i, o in enumerate(path_cat_02):
        obj_name = f"HC{o['id']}-20210912A"
        obj_dict = objects.Galaxy.default_params()
        obj_dict["name"] = obj_name
        obj_dict["position"]["alpha"]["decimal"] = o["ra"].value
        obj_dict["position"]["delta"]["decimal"] = o["dec"].value
        yaml_dict[obj_name] = obj_dict

    p.save_params(os.path.join(output_path, "PATH", "host_candidates.yaml"), yaml_dict)

    # Set up images
    lib.load_images()

    # Square imaging plot
    for label in (True, False):
        lib.candidate_image(
            images=(
                lib.g_img_trimmed,
                lib.R_img_trimmed,
                lib.I_img_trimmed,
                lib.K_img
            ),
            path_cat=path_cat_02,
            plot_width=textwidths["mqthesis"],
            plot_height=textwidths["mqthesis"],
            n_x=2,
            n_y=2,
            ylabelpad=30,
            label=label
        )
        lib.candidate_image(
            images=(
                lib.g_img_trimmed,
                lib.R_img_trimmed,
                lib.I_img_trimmed,
                lib.K_img
            ),
            path_cat=path_cat_02,
            plot_width=textwidths["mqthesis"],
            plot_height=textwidths["mqthesis"],
            n_x=2,
            n_y=2,
            ylabelpad=30,
            label=label,
            band_titles=True
        )
        lib.candidate_image(
            images=(
                lib.g_img_trimmed,
                lib.R_img_trimmed,
                lib.I_img_trimmed,
                lib.K_img
            ),
            path_cat=path_cat_02,
            plot_width=textwidths["mqthesis"],
            plot_height=textwidths["mqthesis"],
            n_x=2,
            n_y=2,
            ylabelpad=30,
            label=label
        )
        lib.candidate_image(
            images=(
                lib.g_img,
                lib.R_img,
                lib.I_img,
                lib.K_img
            ),
            path_cat=path_cat_02,
            plot_width=textwidths["mqthesis"],
            plot_height=textwidths["mqthesis"],
            n_x=2,
            n_y=2,
            ylabelpad=30,
            suffix="unsubtracted",
            label=label
        )
        # Columnar imaging plot
        lib.candidate_image(
            images=(
                lib.g_img_trimmed,
                lib.R_img_trimmed,
                lib.I_img_trimmed,
                lib.K_img
            ),
            path_cat=path_cat_02,
            label=label
        )

        # WISE
        lib.candidate_image(
            path_cat=path_cat_02,
            plot_width=textwidths["mqthesis"],
            plot_height=textwidths["mqthesis"],
            n_x=2,
            n_y=2,
            ylabelpad=30,
            images=(lib.W1_img, lib.W2_img, lib.W3_img, lib.W4_img),
            suffix="wise",
            label=label
        )

    # A stray neutron star?
    rogue_dir = os.path.join(output_path, "rogue_progenitor")
    u.mkdir_check(rogue_dir)
    # Set velocity to a reasonable ejection velocity for a nuclear encounter
    v = 1000 * units.km / units.second
    # Derive a redshift assuming no host contribution to the DM, and average IGM contribution
    z_dm_nohost = frb210912.z_from_dm()
    # Save some quantities
    values = {
        "dm_nohost": {
            "z": z_dm_nohost
        }
    }
    # Get the least-separated host candidate
    nearest_row = path_cat_02[path_cat_02["separation"].argmin()]
    nearest_dist = nearest_row["separation"]
    nearest_position = coordinates.SkyCoord(nearest_row["ra"], nearest_row["dec"])
    nearest_gal = objects.Galaxy(z=1, position=nearest_position)
    nearest_gal.set_z(z_dm_nohost)
    dist_drift_centre = nearest_gal.projected_size(nearest_dist)

    values["dm_nohost"]["r_perp"] = nearest_dist
    values["dm_nohost"]["t_drift"] = (dist_drift_centre / v).to(units.Myr)

    zs = np.arange(0, 3.01, 0.01)
    dists = []
    dists_err = []
    ts = []
    ts_err = []
    for z in zs:
        nearest_gal.set_z(z)
        dist = nearest_gal.projected_size(nearest_dist)
        dist_err = nearest_gal.projected_size(0.7 * units.arcsec)
        dists_err.append(dist_err)
        dists.append(dist)
        t = (dist / v).to(units.Myr)
        ts.append(t)
        t_err = ((dist + dist_err) / v).to(units.Myr)
        ts_err.append(t_err)
    ejected_tbl = table.QTable({
        "z": zs,
        "r_perp": dists,
        "r_perp_err": dists_err,
        "t_drift": ts,
        "t_drift_err": ts_err
    })
    print(ejected_tbl)
    fig, ax = plt.subplots()
    ax.plot(ejected_tbl["z"], ejected_tbl["r_perp"])
    ax.plot(ejected_tbl["z"], ejected_tbl["r_perp"] + ejected_tbl["r_perp_err"])
    ax.plot(ejected_tbl["z"], ejected_tbl["r_perp"] - ejected_tbl["r_perp_err"])
    ax.set_xlabel("z")
    ax.set_ylabel("$R_\perp$")
    fig.savefig(os.path.join(rogue_dir, "angular_distance_from_A.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(ejected_tbl["z"], ejected_tbl["t_drift"])
    ax.plot(ejected_tbl["z"], ejected_tbl["t_drift"] + ejected_tbl["t_drift_err"])
    ax.plot(ejected_tbl["z"], ejected_tbl["t_drift"] - ejected_tbl["t_drift_err"])
    ax.set_xlabel("z")
    ax.set_ylabel("$T_\mathrm{drift}$")
    fig.savefig(os.path.join(rogue_dir, "travel_time_from_A.pdf"))
    plt.close(fig)

    values["ang_size_turnover"] = {
        "z": zs[np.argmax(ejected_tbl["r_perp"])],
        "r_perp": np.max(ejected_tbl["r_perp"]),
        "t_drift": np.max(ejected_tbl["t_drift"])
    }

    ejected_tbl.write(os.path.join(rogue_dir, "rogue_progenitor.ecsv"), overwrite=True)
    p.save_params(os.path.join(rogue_dir, "rogue_progenitor.yaml"), values)


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
        input_dir=input_path,
    )
