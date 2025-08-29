#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as units
import astropy.table as table
import astropy.coordinates as coordinates

import craftutils.utils as u
import craftutils.params as p
import craftutils.plotting as pl
import craftutils.observation.image as image
import craftutils.observation.instrument as instrument
import craftutils.observation.sed as sed
import craftutils.astrometry as astm

import lib

description = """
Runs PATH (Probabilistic Association of Transients; Aggarwal et al 2021) in various configurations on the HST imaging 
data covering the field of FRB 20220610A, and generates some figures.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb220610 = lib.fld

    max_radius = 10.

    pu_trials = {
        "exp": [0., 0.1, 0.2, 0.5, 0.6],
        "uniform": [],
        "core": [],
    }

    lib.load_images()

    # Load HST candidate table
    hst_cand_tbl = table.QTable.read(os.path.join(lib.output_path, "candidates.ecsv"))

    # Load HST image from disk
    lib.load_images()

    centre_ra = np.mean(hst_cand_tbl["ra"])
    centre_dec = np.mean(hst_cand_tbl["dec"])
    centre = coordinates.SkyCoord(centre_ra, centre_dec)

    config_diff = {
        "hst-wfc3_uvis2": {"deblend": False, "npixels": 18},
        "hst-wfc3_ir": {"deblend": True, "npixels": 9}
    }

    for i, img in enumerate(lib.imgs):
        for offset_prior in ("exp", "uniform", "core"):
            name = img.instrument_name
            output_this = os.path.join(lib.output_path, name)
            output_p = os.path.join(output_this, "PATH")
            u.mkdir_check(output_p)
            results = p.load_params(os.path.join(output_this, f"{name}_results.yaml"))

            print(f"\n Processing {name}, {img.filter.name}, {img.filter.cmap} ({img.name})")

            p_u_calc = results["P(U)"]["step"]

            pu_trials_this = pu_trials[offset_prior] + [p_u_calc]

            config = {
                "max_radius": int(max_radius),
                "cut_size": max_radius * 2,
            }
            config.update(config_diff[name])

            for p_u in pu_trials_this:
                run_name = f"offset_{offset_prior}_pu_{np.round(p_u, 3)}"
                output_trial = os.path.join(output_p, run_name)
                u.mkdir_check(output_trial)
                path_tbl, p_ox, p_ux, prior_set, config_n = frb220610.frb.probabilistic_association(
                    img=img,
                    priors={"U": p_u},
                    offset_priors={'method': offset_prior, 'max': 6.0, 'scale': 0.5},
                    config=config,
                    include_img_err=True,
                    associate_kwargs={"show": False},
                    do_plot=True,
                    output_dir=output_trial
                )
                pathdict = {
                    # "table": path_tbl,
                    "p_ox": p_ox,
                    "p_ux": p_ux,
                    "priors": prior_set,
                    "config": config_n
                }
                # Plot 1
                # frame = max_radius * units.arcsec
                # fig = plt.figure(figsize=(12, 12))
                # fig, ax, other_args = frb220610.plot_host(
                #     hst_ir,
                #     frame=frame,
                #     fig=fig,
                #     include_img_err=True,
                # )
                # c = ax.scatter(path_tbl_ir["x"], path_tbl_ir["y"], marker="x", cmap="bwr", c=path_tbl_ir["P_Ox"])
                # cbar = plt.colorbar(c)
                # cbar.set_label('$P(O|x)$', rotation=270)
                # fig.savefig(os.path.join(lib.output_path, f"path_candidates_c_{p_u}.pdf"))
                # plt.show()
                # plt.close(fig)

                # path_tbl_ir.sort("P_Ox", reverse=True)
                # path_tbl_ir.write(os.path.join(lib.output_path, f"path_candidates_{p_u}.csv"), overwrite=True)
                # path_tbl_ir.write(os.path.join(lib.output_path, f"path_candidates_{p_u}.ecsv"), overwrite=True)

                matches_path, matches_cand, distance = astm.match_catalogs(
                    cat_1=path_tbl, ra_col_1="ra", dec_col_1="dec",
                    cat_2=hst_cand_tbl
                )

                matches_stack = table.hstack([matches_cand, matches_path])
                matches_stack["separation"] = distance.to("arcsec")
                matches_stack["position_path"] = coordinates.SkyCoord(
                    matches_stack["ra_2"], matches_stack["dec_2"],
                    unit="deg"
                )

                fig, ax, _ = frb220610.plot_host(
                    img,
                    frame=3 * units.arcsec,
                    imshow_kwargs={"cmap": "viridis"},
                    include_img_err=True,
                    centre=centre
                )
                matches_stack["x_g"], matches_stack["y_g"] = img.world_to_pixel(matches_stack["position"])
                matches_stack["x_p"], matches_stack["y_p"] = img.world_to_pixel(matches_stack["position_path"])

                c = ax.scatter(matches_stack["x_p"], matches_stack["y_p"], marker="x", c=matches_stack["P_Ox"],
                               cmap="bwr")
                for row in matches_stack:
                    ax.text(row["x_g"], row["y_g"], row["id"])
                plt.colorbar(c)
                plt.savefig(os.path.join(output_trial, f"PATH_matches_{run_name}.pdf"))
                plt.close(fig)

                matches_export = matches_stack["id", "P_Ox"]
                matches_export["P(O|x)"] = matches_export["P_Ox"].round(4)
                matches_export["P(O|x) %"] = matches_export["P(O|x)"] * 100
                for fmt in ("csv", "ecsv"):
                    matches_export.write(
                        os.path.join(output_trial, f"PATH_matches_{run_name}.{fmt}"),
                        overwrite=True,
                        format=f"ascii.{fmt}"
                    )
                matches_export.write(os.path.join(lib.output_path, f"PATH_matches_{run_name}.csv"), overwrite=True)
                # pathdict["matches"] = matches_export

                p.save_params(os.path.join(output_trial, f"PATH_results_{run_name}.yaml"), pathdict)

        # p.save_params(os.path.join(output_this, f"{name}_results.yaml"), results)


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
