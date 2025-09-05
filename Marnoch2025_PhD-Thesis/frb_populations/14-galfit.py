#!/usr/bin/env python
# Code by Lachlan Marnoch, 2024

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import astropy.units as units
import astropy.uncertainty as unc
from astropy.coordinates import SkyCoord
import astropy.table as table

import craftutils.utils as u
import craftutils.plotting as pl
from craftutils.observation import field
from craftutils.observation import image

import lib

description = """
Does some analysis with the GALFIT results and writes some tables.
"""


def main(
        output_dir: str,
        input_dir: str,
        skip_plots: bool
):
    np.random.seed(1994)
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)
    plot_dir = os.path.join(output_dir, "galaxy_schematics")
    os.makedirs(plot_dir, exist_ok=True)

    n_samples = 1000
    pox_min = 0.75
    mag_max = 21. * units.mag
    n_max = 2.
    this_script = u.latex_sanitise(os.path.basename(__file__))

    if isinstance(lib.dropbox_path, str):
        db_path = os.path.join(lib.dropbox_path, "tables")
    else:
        db_path = None

    # Sanitise the big FRB table a little
    craft_hosts = lib.load_frb_table(hosts_only=True, craft_only=True)
    craft_hosts["hg_name"] = ["HG" + row["name"] for row in craft_hosts]
    craft_hosts["frb_ra"] = craft_hosts["ra"]
    craft_hosts["frb_ra_err"] = craft_hosts["ra_err"]
    craft_hosts["frb_dec"] = craft_hosts["dec"]
    craft_hosts["frb_dec_err"] = craft_hosts["dec_err"]
    craft_hosts["frb_a"] = craft_hosts["a"]
    craft_hosts["frb_a_proj"] = craft_hosts["a_proj"]
    craft_hosts["frb_b"] = craft_hosts["b"]
    craft_hosts["frb_b_proj"] = craft_hosts["b_proj"]
    craft_hosts["frb_theta"] = craft_hosts["theta"]
    craft_hosts["frb_coord"] = craft_hosts["coord"]
    craft_hosts.remove_columns(["ra", "ra_err", "dec", "dec_err", "coord", "a", "b", "a_proj", "b_proj", "theta"])

    def print_delta(tbl_new, tbl_old, col="object_name"):
        delta = list(set(tbl_old[col]) - set(tbl_new[col]))
        print("\nMinus:", sorted(delta))

    print(len(craft_hosts), 'craft_hosts')
    craft_hosts_imaged = craft_hosts[craft_hosts["vlt_imaged"]]
    print_delta(tbl_new=craft_hosts_imaged, tbl_old=craft_hosts, col="name")
    print(len(craft_hosts_imaged), 'craft_hosts_imaged, remove vlt_imaged == False')

    craft_photometry = lib.load_photometry_table(True)
    print(len(craft_photometry), 'craft_photometry')
    craft_photometry_hosts = craft_photometry[[n.startswith("HG") for n in craft_photometry["object_name"]]]
    print_delta(tbl_new=craft_photometry_hosts, tbl_old=craft_photometry)
    print(len(craft_photometry_hosts), 'craft_photometry_hosts, remove objects without "HG"')

    craft_photometry_hosts["host_ra"] = craft_photometry_hosts["ra"]
    craft_photometry_hosts["host_ra_err"] = craft_photometry_hosts["ra_err"].to("arcsec")
    craft_photometry_hosts["host_dec"] = craft_photometry_hosts["dec"]
    craft_photometry_hosts["host_dec_err"] = craft_photometry_hosts["dec_err"]
    craft_photometry_hosts.remove_columns(["ra", "ra_err", "dec", "dec_err", "z"])
    print(craft_photometry_hosts["object_name", "host_dec"][craft_photometry_hosts["host_ra"] < -90 * units.deg])
    craft_photometry_hosts["host_coord"] = SkyCoord(
        craft_photometry_hosts["host_ra"],
        craft_photometry_hosts["host_dec"]
    )

    craft_joined = table.join(
        left=craft_hosts_imaged,
        right=craft_photometry_hosts,
        keys_left=["hg_name"],
        keys_right=["object_name"],
        join_type="inner"
    )
    print_delta(tbl_new=craft_joined, tbl_old=craft_photometry_hosts)
    print(len(craft_joined), 'craft_joined')

    craft_joined.remove_columns(["frb_object"])
    craft_joined.write(
        os.path.join(lib.table_path, "craft_photometry_extended.ecsv"),
        overwrite=True
    )

    craft_galfit_1 = craft_joined.copy()
    craft_galfit_r = craft_galfit_1[craft_galfit_1["galfit_r_eff"] > 0. * units.arcsec]
    print_delta(tbl_new=craft_galfit_r, tbl_old=craft_galfit_1)
    print(len(craft_galfit_r), 'craft_galfit, remove r_eff < 0 arcsec (ie GALFIT info missing)')

    craft_galfit_n = craft_galfit_r[craft_galfit_r["galfit_n_err"] < 100.]
    print_delta(tbl_new=craft_galfit_n, tbl_old=craft_galfit_r)
    print(len(craft_galfit_n), 'craft_galfit, remove n_err >= 100')

    craft_galfit_ba = craft_galfit_n[craft_galfit_n["galfit_axis_ratio_err"] < craft_galfit_n["galfit_axis_ratio"]]
    print_delta(tbl_new=craft_galfit_ba, tbl_old=craft_galfit_n)
    print(len(craft_galfit_ba), 'craft_galfit, remove b/a err >= b/a')
    craft_galfit = craft_galfit_ba

    # craft_galfit = craft_galfit_n
    craft_galfit["galfit_ra_err"] = craft_galfit["galfit_ra_err"].to("arcsec")
    craft_galfit["galfit_dec_err"] = craft_galfit["galfit_dec_err"].to("arcsec")

    craft_galfit = craft_galfit[craft_galfit["name"] != "20220610A"]
    craft_galfit = craft_galfit[craft_galfit["name"] != "20230526A"]
    print(len(craft_galfit), 'craft_galfit; remove 220610A and 230526A')

    # GALFIT OUTPUTS, PART ONE
    # ================

    flds = field.list_fields()



    # q_0 = 0.2

    legend = []
    for row in craft_galfit:
        row["galfit_band"] = {
            "g_HIGH": r"$g$",
            "I_BESS": r"$I$",
            "R_SPECIAL": r"$R$",
            "Ks": r"\Ks"
        }[row["galfit_band"]]

        sup = "${"
        if row["path_pox"] >= pox_min:
            # sup += r"\dagger"
            # if row["z"] > 0:
            #     sup += "\square"
            if row["galfit_mag"] <= mag_max:
                sup += r"\star"
            if row["galfit_n"] < n_max:
                sup += r"\diamond"
        sup += "}$"
        legend.append(sup)
    craft_galfit["legend"] = legend

    host_galfit_1 = craft_galfit.copy()[
        "name", "legend", "galfit_band",
        "galfit_axis_ratio", "galfit_axis_ratio_err",
        "galfit_theta", "galfit_theta_err",
        "galfit_r_eff", "galfit_r_eff_err",
        "galfit_mag", "galfit_mag_err",
        "galfit_n", "galfit_n_err"
    ]
    tbl_path = os.path.join(lib.tex_path, "craft_galfit.tex")
    print("Writing table to", tbl_path)
    u.latexise_table(
        # tbl=host_galfit_1,
        tbl=u.add_stats(
            host_galfit_1,
            name_col="name",
            cols_exclude=["name", "legend", "galfit_band"],
            round_n=3
        ),
        column_dict=lib.nice_var_dict,
        output_path=tbl_path,
        caption=r"Single-Sérsic morphological properties of CRAFT host galaxies, as derived using \galfit{} with VLT imaging. "
                r"Centroid positions are reported in \autoref{tab:craft_galfit_derived}. Uncertainties are the statistical uncertainties provided by \galfit{}. "
                r"Symbols next to the names indicate sample membership: "
                # r"$\dagger:$Secure PATH association ($P(O_i|x) > 0.75$); "
                # rf"$\square:$Secure hosts with measured redshifts; "
                rf"$\star:$ magnitude-limited sample ($\mgalfit\leq{mag_max.value}$); " 
                rf"$\diamond:\ n$-limited sample ($n<{n_max}$). "
                # r"\textbf{Add zeropoint uncertainty to mag error?}"
                r" \tabscript{" + u.latex_sanitise(os.path.basename(__file__)) + "}",
        short_caption="\galfit{} properties of CRAFT hosts",
        label="tab:craft_galfit",
        sub_colnames={
            "galfit_theta": r"($\degrees$)",
            "galfit_r_eff": r"($\arcsec$)",
        },
        # second_path=os.path.join(db_path, "craft_galfit.tex"),
        coltypes="llcccccc"
    )

    craft_galfit["galfit_coord"] = SkyCoord(craft_galfit["galfit_ra"], craft_galfit["galfit_dec"])
    craft_galfit["galfit_offset"] = units.Quantity(
        craft_galfit["frb_coord"].separation(craft_galfit["galfit_coord"]).to("arcsec")
    )

    # frb_ra_d = unc.normal(
    #     center=craft_galfit["frb_coord"].ra,
    #     std=craft_galfit["frb_ra_err"],
    #     n_samples=n_samples
    # )
    # frb_dec_d = unc.normal(
    #     center=craft_galfit["frb_coord"].dec,
    #     std=craft_galfit["frb_dec_err"],
    #     n_samples=n_samples
    # )

    # galfit_ra_d = unc.normal(
    #     center=craft_galfit["galfit_ra"],
    #     std=craft_galfit["galfit_ra_err"],
    #     n_samples=n_samples
    # )
    # galfit_dec_d = unc.normal(
    #     center=craft_galfit["galfit_dec"],
    #     std=craft_galfit["galfit_dec_err"],
    #     n_samples=n_samples
    # )

    craft_galfit["galfit_a"] = craft_galfit["galfit_r_eff"]
    craft_galfit["galfit_a_err"] = craft_galfit["galfit_r_eff_err"]
    craft_galfit["galfit_b"] = craft_galfit["galfit_a"] * craft_galfit["galfit_axis_ratio"]
    craft_galfit["galfit_b_err"] = u.uncertainty_product(
        craft_galfit["galfit_b"],
        (craft_galfit["galfit_r_eff"], craft_galfit["galfit_r_eff_err"]),
        (craft_galfit["galfit_axis_ratio"], craft_galfit["galfit_axis_ratio_err"])
    )

    craft_galfit["galfit_offset_norm"] = craft_galfit["galfit_offset"] / craft_galfit["galfit_r_eff"]

    # lib.add_q_0(craft_galfit, "galfit_axis_ratio")

    # craft_galfit["galfit_inclination"][craft_galfit["galfit_axis_ratio"] > q_0] = u.inclination(
    #     craft_galfit["galfit_axis_ratio"],
    #     q_0=q_0,  # np.min(craft_galfit["galfit_axis_ratio"])
    # )

    skip_plots_ = skip_plots

    for q_0 in (0.13, 0.2):

        if q_0 == 0.13:
            skip_plots = True
        else:
            skip_plots = skip_plots_

        craft_galfit = u.inclination_table(
            craft_galfit,
            axis_ratio_column="galfit_axis_ratio",
            cos_column="galfit_cos_inclination",
            inclination_column="galfit_inclination",
            q_0=q_0,  # np.min(craft_galfit["galfit_axis_ratio"]),
        )

        craft_galfit["galfit_1-cos_inclination"] = 1 / craft_galfit["galfit_cos_inclination"]

        craft_galfit["galfit_offset_deproj"] = u.deprojected_offset(
            object_coord=craft_galfit["frb_coord"],
            galaxy_coord=craft_galfit["galfit_coord"],
            position_angle=craft_galfit["galfit_theta"],
            inc=craft_galfit["galfit_inclination"]
        )

        craft_galfit["galfit_offset_norm_deproj"] = craft_galfit["galfit_offset_deproj"] / craft_galfit["galfit_r_eff"]

        craft_galfit["galfit_offset_err"] = -999 * units.arcsec
        craft_galfit["galfit_offset_proj"] = -999 * units.kpc
        craft_galfit["galfit_offset_proj_err"] = -999 * units.kpc
        craft_galfit["galfit_offset_disk"] = -999 * units.kpc
        craft_galfit["galfit_offset_disk_err"] = -999 * units.kpc
        craft_galfit["galfit_a_proj"] = -999 * units.kpc
        craft_galfit["galfit_a_proj_err"] = -999 * units.kpc
        craft_galfit["galfit_b_proj"] = -999 * units.kpc
        craft_galfit["galfit_b_proj_err"] = -999 * units.kpc
        craft_galfit["galfit_frb_theta"] = craft_galfit["galfit_coord"].position_angle(craft_galfit["frb_coord"]).to("deg")

        frb_coords_d = []
        gal_coords_d = []
        position_angle_d = []
        deprojected_d = []
        deprojected_norm_errs = []
        deprojected_errs = []
        x_ds = []
        y_ds = []
        inc_ds = []
        inc_errs = []
        offset_norm_errs = []
        offset_disk_errs = []
        cos_i_errs = []
        cos1_i_errs = []

        for i, row in enumerate(craft_galfit):

            dec_gal = row["galfit_coord"].dec
            cos_dec = np.cos(dec_gal)

            gal_ra_d = unc.normal(
                center=row["galfit_coord"].ra,
                std=row["galfit_ra_err"] / cos_dec,
                n_samples=n_samples
            )
            gal_dec_d = unc.normal(
                center=row["galfit_coord"].dec,
                std=row["galfit_dec_err"],
                n_samples=n_samples
            )
            gal_theta_d = unc.normal(
                center=row["galfit_theta"],
                std=row["galfit_theta_err"],
                n_samples=n_samples
            )
            # Clean out unphysical values produced by large error bars
            axis_ratio_d = unc.normal(
                center=row["galfit_axis_ratio"],
                std=row["galfit_axis_ratio_err"],
                n_samples=n_samples
            )
            # axis_ratio_d.distribution[axis_ratio_d.distribution < row["q_0"]] = row["q_0"]
            axis_ratio_d.distribution[axis_ratio_d.distribution > 1] = 1.
            inclination_d = u.inclination_array(
                axis_ratio_d.distribution,
                q_0=q_0
                # row["q_0"]
            )
            inc_ds.append(inclination_d)
            inc_errs.append(np.nanstd(inclination_d))

            cos_inc_d = u.inclination_array(
                axis_ratio_d.distribution,
                q_0=q_0,
                # row["q_0"],
                uncos=False
            )
            cos_i_errs.append(np.nanstd(cos_inc_d))

            cos1_inc_d = 1 / cos_inc_d
            cos1_i_errs.append(np.nanstd(cos1_inc_d))

            # The following generates a sample of FRB SkyCoords following a distribution defined by the uncertainty ellipse.
            a_frb = row["frb_a"]
            b_frb = row["frb_b"]
            if a_frb == 0:
                if row["frb_ra_err"] > row["frb_dec_err"]:
                    a_frb = row["frb_ra_err"]
                    b_frb = row["frb_dec_err"]
                    theta_frb = 90 * units.deg
                else:
                    a_frb = row["frb_dec_err"]
                    b_frb = row["frb_ra_err"]
                    theta_frb = 0 * units.deg
            else:
                theta_frb = row["frb_theta"]

            delta_a_d = unc.normal(
                center=0,
                std=a_frb,
                n_samples=n_samples
            )

            delta_b_d = unc.normal(
                center=0,
                std=b_frb,
                n_samples=n_samples
            )

            sint = np.cos(theta_frb)
            cost = np.sin(theta_frb)

            delta_x_d = delta_a_d * cost - delta_b_d * sint
            delta_alpha_d = delta_x_d / cos_dec
            delta_delta_d = delta_b_d * cost + delta_a_d * sint

            frb_ra_d = (delta_alpha_d + row["frb_coord"].ra).to(units.deg)
            frb_dec_d = (delta_delta_d + row["frb_coord"].dec).to(units.deg)

            # Use to get guesstimate uncertainty
            offset_d = u.great_circle_dist(
                ra_1=gal_ra_d,
                ra_2=frb_ra_d,
                dec_1=gal_dec_d,
                dec_2=frb_dec_d
            )
            row["galfit_offset_err"] = np.nanstd(offset_d.distribution)

            r_eff_d = unc.normal(
                center=row["galfit_r_eff"],
                std=row["galfit_r_eff_err"],
                n_samples=n_samples
            )

            offset_norm_d = offset_d / r_eff_d
            offset_norm_errs.append(np.nanstd(offset_norm_d.distribution))

            frb_c = SkyCoord(frb_ra_d.distribution, frb_dec_d.distribution)
            frb_coords_d.append(frb_c)
            gal_c = SkyCoord(gal_ra_d.distribution, gal_dec_d.distribution)
            gal_coords_d.append(gal_c)

            deproj_d = u.deprojected_offset(
                object_coord=frb_c,
                galaxy_coord=gal_c,
                position_angle=gal_theta_d,
                inc=inclination_d
            )
            deprojected_d.append(deproj_d)
            deproj_err = np.nanstd(deproj_d.distribution)
            deprojected_errs.append(deproj_err)

            deproj_norm_d = deproj_d / r_eff_d
            deprojected_norm_errs.append(np.nanstd(deproj_norm_d.distribution))

            if not skip_plots:
                fig = plt.figure(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] * 2 / 3))
                if row["z"] > 0.:
                    n_col = 2
                else:
                    n_col = 1
                ax = fig.add_subplot(1, n_col, 1)
                x_frb = (row["galfit_offset"] * np.sin(-row["galfit_frb_theta"])).to("arcsec")
                y_frb = row["galfit_offset"].to("arcsec") * np.cos(-row["galfit_frb_theta"])
                x_gal, y_gal = 0, 0
                a_e, b_e = row["galfit_a"].to("arcsec").value, row["galfit_b"].to("arcsec").value
                # print(a_e, b_e)
                theta_gal = row["galfit_theta"].value
                e = Ellipse(
                    xy=(x_gal, y_gal),
                    width=a_e * 2,
                    height=b_e * 2,
                    angle=theta_gal + 90,
                    edgecolor="black",
                    facecolor="none",
                    label="Host GALFIT shape"
                )
                ax.scatter(
                    (gal_ra_d.distribution - frb_ra_d.distribution).to("arcsec") * cos_dec,
                    (frb_dec_d.distribution - gal_dec_d.distribution).to("arcsec"),
                    label="FRB uncertainty distribution",
                    marker="."
                )
                ax.scatter(x_frb, y_frb, label="FRB", marker="x")
                ax.scatter(x_gal, y_gal, label="Host centroid", marker="x")
                ax.add_artist(e)
                ax.axis('equal')
                # print(a, x_frb.value, y_frb.value)
                lim = (x_frb.value + y_frb.value + a_e)
                ax.plot((-lim, lim), (0, 0), c="black")
                ax.plot((0, 0), (-lim, lim), c="black")

                a_e, b_e = a_frb.to("arcsec").value, b_frb.to("arcsec").value
                theta_pl = theta_frb.value
                e_frb = Ellipse(
                    xy=(x_frb, y_frb),
                    width=a_e * 2,
                    height=b_e * 2,
                    angle=theta_pl + 90,
                    edgecolor="red",
                    facecolor="none",
                    label="FRB uncertainty ellipse"
                )
                ax.add_artist(e_frb)

                # plt.savefig(os.path.join(plot_dir, f"ang_{row['name']}.png"))
                # plt.close(fig)

            if row["z"] > 0.:
                x_d = -((frb_ra_d.distribution - gal_ra_d.distribution) * cos_dec).to("rad").value * row["d_A"].to("kpc")
                y_d = (frb_dec_d.distribution - gal_dec_d.distribution).to("rad").value * row["d_A"].to("kpc")

                row["galfit_offset_proj"] = (row["galfit_offset"].to("rad").value * row["d_A"]).to("kpc")
                row["galfit_offset_proj_err"] = (row["galfit_offset_err"].to("rad").value * row["d_A"]).to("kpc")
                row["galfit_offset_disk"] = row["galfit_offset_deproj"].to("rad").value * row["d_A"].to("kpc")
                row["galfit_a_proj"] = row["galfit_r_eff_proj"]
                row["galfit_a_proj_err"] = row["galfit_r_eff_proj_err"]
                row["galfit_b_proj"] = row["galfit_a_proj"] * row["galfit_axis_ratio"]
                row["galfit_b_proj_err"] = u.uncertainty_product(
                    row["galfit_b_proj"],
                    (row["galfit_a_proj"], row["galfit_a_proj_err"]),
                    (row["galfit_axis_ratio"], row["galfit_axis_ratio_err"])
                )

                offset_disk_errs.append(deproj_err.to("rad").value * row["d_A"].to("kpc"))

                # fig = plt.figure()

                if not skip_plots:
                    ax = fig.add_subplot(1, n_col, 2)
                    x_frb = row["galfit_offset_proj"] * np.sin(-row["galfit_frb_theta"])
                    y_frb = row["galfit_offset_proj"] * np.cos(-row["galfit_frb_theta"])
                    x_gal, y_gal = 0, 0
                    a_e, b_e = row["galfit_a_proj"].value, row["galfit_b_proj"].value
                    theta_gal = row["galfit_theta"].value
                    e = Ellipse(
                        xy=(x_gal, y_gal),
                        width=a_e * 2,
                        height=b_e * 2,
                        angle=theta_gal + 90,
                        edgecolor="black",
                        facecolor="none",
                        label="Host GALFIT shape"
                    )
                    ax.scatter(x_d, y_d, label="FRB uncertainty distribution", marker=".")
                    ax.scatter(x_frb, y_frb, label="FRB", marker="x")
                    ax.scatter(x_gal, y_gal, label="Host centroid", marker="x")
                    ax.add_artist(e)
                    ax.axis('equal')
                    # print(a, x_frb.value, y_frb.value)
                    lim = (x_frb.value + y_frb.value + a_e)
                    ax.plot((-lim, lim), (0, 0), c="black")
                    ax.plot((0, 0), (-lim, lim), c="black")

                    a_e = (a_frb.to("rad").value * row["d_A"]).to("kpc")
                    b_e = (b_frb.to("rad").value * row["d_A"]).to("kpc")
                    theta_pl = theta_frb.value
                    e_frb = Ellipse(
                        xy=(x_frb, y_frb),
                        width=a_e * 2,
                        height=b_e * 2,
                        angle=theta_pl + 90,
                        edgecolor="red",
                        facecolor="none",
                        label="FRB uncertainty ellipse"
                    )
                    ax.add_artist(e_frb)
            else:
                offset_disk_errs.append(-999 * units.kpc)

            if not skip_plots:
                fig.suptitle(row["name"])
                fig.tight_layout()
                lib.savefig(fig, f"both_{row['name']}", subdir="galaxy_schematics", tight=True)
                plt.close(fig)

                fld_name = row["field_name"]
                if fld_name not in flds:
                    fld_name = fld_name[:-1]
                if fld_name in flds:
                    fld = field.Field.from_params(name=fld_name)
                    img_dict = fld.imaging[row["galfit_img"]]
                    img_path = img_dict["path"]
                    img = image.ImagingImage.from_fits(img_path)
                    fig = plt.figure(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] * 2 / 3))
                    fig, ax, _ = fld.plot_host(
                        img=img,
                        fig=fig,
                        frame=np.max(offset_d.distribution) * 2,
                        imshow_kwargs=dict(
                            cmap="cmr.bubblegum"
                        )
                    )
                    x_img_frb, y_img_frb = img.world_to_pixel(
                        coord=frb_c
                    )
                    x_img_gal, y_img_gal = img.world_to_pixel(
                        coord=gal_c
                    )
                    ax.scatter(x_img_frb, y_img_frb, marker=".", c="lime", zorder=0)
                    ax.scatter(x_img_gal, y_img_gal, marker=".", c="black", zorder=0)
                    fig.suptitle(row["name"])
                    fig.tight_layout()
                    lib.savefig(
                        fig,
                        filename=f"mc_img_{row['name']}",
                        subdir="mc_imgs",
                        tight=True
                    )
                    plt.close(fig)

        craft_galfit["galfit_inclination_err"] = inc_errs
        craft_galfit["galfit_offset_deproj_err"] = deprojected_errs
        craft_galfit["galfit_offset_norm_deproj_err"] = deprojected_norm_errs
        craft_galfit["galfit_offset_norm_err"] = offset_norm_errs
        craft_galfit["galfit_offset_disk_err"] = offset_disk_errs
        craft_galfit["galfit_cos_inclination_err"] = cos_i_errs
        craft_galfit["galfit_1-cos_inclination_err"] = cos1_i_errs

        craft_galfit_p = craft_galfit.copy()
        craft_galfit_p = craft_galfit_p[craft_galfit_p["path_pox"] >= pox_min]
        print_delta(tbl_new=craft_galfit_p, tbl_old=craft_galfit)
        print(len(craft_galfit_p), f"remove path_pox < {pox_min}")

        path = lib.craft_galfit_path(q_0)

        if "frb_object" in craft_galfit_p.colnames:
            craft_galfit_p.remove_column("frb_object")
        craft_galfit_p.write(path, overwrite=True)
        craft_galfit_p.write(path.replace(".ecsv", ".csv"), overwrite=True)

    craft_galfit_2 = craft_galfit[
        "name", "legend", "galfit_ra", "galfit_ra_err",
        "galfit_dec", "galfit_dec_err",
        "galfit_r_eff_proj", "galfit_r_eff_proj_err",
        "galfit_offset", "galfit_offset_err",
        "galfit_offset_proj", "galfit_offset_proj_err",
        "galfit_offset_norm", "galfit_offset_norm_err",
    ]
    craft_galfit_2["galfit_ra_err"] = craft_galfit_2["galfit_ra_err"].to("deg")
    craft_galfit_2["galfit_dec_err"] = craft_galfit_2["galfit_dec_err"].to("deg")

    craft_galfit_2 = u.latexise_table(
        tbl=u.add_stats(
            craft_galfit_2,
            name_col="name",
            cols_exclude=["name", "legend", "galfit_ra", "galfit_dec"],
            round_n=3
        ),
        column_dict=lib.nice_var_dict,
        sub_colnames={
            "galfit_ra": "(J2000)",
            "galfit_dec": "(J2000)",
            "galfit_offset": r"($\arcsec$)",
            "galfit_offset_proj": r"(kpc)",
            "galfit_r_eff_proj": r"(kpc)",
        },
        output_path=str(os.path.join(lib.tex_path, "craft_galfit_2.tex")),
        # second_path=os.path.join(db_path, "craft_galfit_2.tex"),
        short_caption="\galfit{}-derived properties of CRAFT hosts",
        caption=r"Properties derived from single-Sérsic morphological parameters of CRAFT host galaxies. "
                r"The sky coordinates here are the host centroid as found by \galfit{}, with FRB separations derived from this position. "
                r"Morphological parameters are given in \autoref{tab:craft_galfit}. "
                # r"$\alpha_\mathrm{host}$ and"
                # r" $\delta_\mathrm{host}$ give the J2000 sky position of the \galfit{} centroid{}, with uncertainties"
                # r" combining the image astrometric uncertainty with the \galfit{} centroid uncertainty."
                r" \tabscript{" + this_script + "}",
        label="tab:craft_galfit_derived",
        coord_kwargs={
            "ra_err_seconds": False
        },
        coltypes="llcccccc"
    )

    craft_galfit_3 = craft_galfit[
        "name", "legend",
        "galfit_inclination",
            # "galfit_inclination_err",
        "galfit_offset_deproj",
            # "galfit_offset_deproj_err",
        "galfit_offset_norm_deproj",
            # "galfit_offset_norm_deproj_err",
        "galfit_offset_disk",
        # "galfit_offset_disk_err",
        # "galfit_offset_norm_disk",
        # "galfit_offset_norm_disk_err",
    ]

    # craft_galfit_3["galfit_inclination"] = [int(row["galfit_inclination"].round(0).value) for row in craft_galfit_2]
    for row in craft_galfit_3:
        if row["galfit_offset_deproj"] > 100 * units.arcsec:
            row["galfit_offset_deproj"] = -999 * units.arcsec
            row["galfit_offset_norm_deproj"] = -999
            row["galfit_offset_disk"] = -999 * units.kpc

    craft_galfit_3 = u.latexise_table(
        tbl=u.add_stats(
            craft_galfit_3,
            name_col="name",
            cols_exclude=["name", "legend",],  # , "galfit_ra", "galfit_dec"],
            round_n=3
        ),
        column_dict=lib.nice_var_dict,
        sub_colnames={
            "galfit_offset_deproj": r"($\arcsec$)",
            "galfit_offset_disk": r"(kpc)",
            "galfit_inclination": r"($\degrees$)",
        },
        output_path=str(os.path.join(lib.tex_path, "craft_galfit_3.tex")),
        # second_path=os.path.join(db_path, "craft_galfit_3.tex"),
        short_caption="Inclination-dependent properties of CRAFT hosts",
        caption=r"Properties derived from the inclination formula (\autoref{equ:inclination}) and \galfit{} morphological parameters (\autoref{tab:craft_galfit} and \autoref{tab:craft_galfit_derived}). Although uncertainties are estimated for each quantity, we report only the fiducial values."
                r" \tabscript{" + this_script + "}",
        label="tab:craft_galfit_inclination",
        round_cols=[
            "galfit_inclination",
            "galfit_offset_deproj",
            "galfit_offset_norm_deproj",
            "galfit_offset_disk",
            # "galfit_offset_norm_disk"
        ],
        round_digits=1,
        coltypes="llcccc"
    )

    craft_galfit_z = craft_galfit_p.copy()
    craft_galfit_z = craft_galfit_z[craft_galfit_z["z"] > 0]
    print_delta(craft_galfit_z, craft_galfit_p)
    print(len(craft_galfit_z), 'craft_galfit_z')

    commands_file = os.path.join(lib.tex_path, "commands_galfit_generated.tex")
    stats_lines = u.latex_command_file(
        {
            "nGALFIT": len(craft_galfit_p),
            "nGALFITz": len(craft_galfit_z),
            "nGALFITmag": len(craft_galfit[craft_galfit["galfit_mag"] >= mag_max]),
            "nGALFITn": len(craft_galfit[craft_galfit["galfit_n"] < n_max]),
        },
        output_path=commands_file
    )

    if isinstance(lib.dropbox_path, str):
        db_path = os.path.join(lib.dropbox_path, "commands")
        if os.path.isdir(lib.dropbox_path):
            shutil.copy(commands_file, db_path)

    galfit_colnames = [
        "field_name", "z", "frb_ra", "frb_ra_err", "frb_dec", "frb_dec_err", "frb_a", "frb_b", "frb_theta"
    ] + list(filter(lambda n: n.startswith("galfit"), craft_galfit.colnames))
    galfit_tbl = craft_galfit[galfit_colnames]
    galfit_tbl.write(os.path.join(output_dir, "galfit_table.ecsv"), overwrite=True)
    galfit_tbl.write(os.path.join(output_dir, "galfit_table.csv"), overwrite=True)




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
        "--do_plots",
        help="Skip plotting",
        action="store_true"
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        skip_plots=not args.do_plots
    )
