#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np

import astropy.units as units
import astropy.table as table

import craftutils.utils as u

import lib

description = """
Translates tables into Latex for thesis insertion.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    this_script = u.latex_sanitise(os.path.basename(__file__))

    halos = lib.read_master_table()

    halos["r_perp_eff"] = (halos["offset_angle_mc"] / halos["galfit_r_eff"]).decompose()
    halos["r_perp_eff_err"] = u.uncertainty_product(
        halos["r_perp_eff"],
        (halos["offset_angle_mc"], halos["offset_angle_mc_err"]),
         (halos["galfit_r_eff"], halos["galfit_r_eff_err"])
    ).decompose()
    # halos.sort("id_short")
    halos_list = list([dict(row) for row in halos])
    host_row = halos_list.pop(-1)
    halos_list.insert(0, host_row)
    halos = table.QTable(halos_list)

    bands = (
        "hst-wfc3_uvis2_F300X",
        "vlt-fors2_g-HIGH",
        "panstarrs1_r",
        "vlt-fors2_I-BESS",
        "panstarrs1_z",
        "panstarrs1_y",
        "hst-wfc3_ir_F160W",
    )

    mag_cols = []
    col_dict = {
        "id_short": "ID",
        "id_cat": r"ID\phantom{.}",
        "mag_corr_hst-wfc3_uvis2_F300X": "F300X",
        "mag_corr_vlt-fors2_g-HIGH": "$g_\mathrm{HIGH}$",
        "mag_corr_panstarrs1_r": "$r$",
        "mag_corr_vlt-fors2_I-BESS": "$I_\mathrm{BESS}$",
        "mag_corr_panstarrs1_z": "$z$",
        "mag_corr_panstarrs1_y": "$y$",
        "mag_corr_hst-wfc3_ir_F160W": "F160W",
        "offset_angle": r"$\phi$",
        "offset_angle_mc": r"$\phi$",
        "r_perp": r"$R_\perp$",
        "r_perp_mc_K18": r"$R_\perp$",
        "r_perp_eff": r"$R_\perp / \Reff$",
        "z": "$z$",
        "ra": r"$\alpha$",
        "dec": r"$\delta$",
        "sample": "Sample",
        "log_mass_stellar": r"$\log_{10}{(\massstar/\si{\solarmass})}$",
        "d_A": r"$D_A$",
        "galfit_axis_ratio": r"$b/a$",
        "galfit_mag": r"$m_\mathrm{galfit}$",
        "galfit_n": "$n$",
        "galfit_r_eff": r"$R_\mathrm{eff}$",
        "galfit_r_eff_proj": r"$R_\mathrm{proj}$",
        "galfit_theta": r"$\theta_\mathrm{PA}$",
    }

    sub_colnames = {
        "id_cat": "(PS1)",
        "offset_angle": r"$\prime\prime$",
        "offset_angle_mc": r"$\prime\prime$",
        "r_perp": r"kpc",
        "r_perp_mc_K18": r"kpc",
        # "ra": r"$\circ$",
        # "dec": r"$\circ$",
        "d_A": r"Mpc",
        "galfit_mag": r"AB mag",
        "galfit_r_eff": r"$\prime\prime$",
        "galfit_r_eff_proj": r"kpc",
        "galfit_theta": r"$\circ$",
    }

    round_dict = {
        "z": 3,
        "d_A": 0,
        "offset_angle": 1,
        "r_perp": 1,
        # "r_perp_eff": 1,
    }

    for band in bands:
        mag_key = f"mag_corr_{band}"
        err_key = f"mag_corr_{band}_err"
        halos[mag_key] = halos[f"mag_best_{band}"] - halos[f"ext_gal_{band}"]
        halos[err_key] = halos[f"mag_best_{band}_err"]

        halos[mag_key][halos[mag_key] < 0 * units.mag] = -999 * units.mag

        mag_cols.append(mag_key)
        mag_cols.append(err_key)

    # print(halos.colnames)

    photm_tbl = halos[["id_short"] + mag_cols]
    for col in photm_tbl.columns:
        print(col, photm_tbl[1][col])

    tbl_photom = u.latexise_table(
        tbl=photm_tbl,
        column_dict=col_dict,
        output_path=os.path.join(lib.latex_table_path, "fg_photometry.tex"),
        second_path=os.path.join(lib.latex_table_path_db, "fg_photometry.tex"),
        label=f"tab:fg_photometry",
        short_caption="Photometry of FRB 20190714A host and foreground galaxies",
        caption="Table of photometric (AB) magnitudes of host and foreground galaxies for \FRB{20190714A}, corrected for Galactic extinction." + r"\tabscript{" + this_script + "}",
    )

    for rel in "K18", "M13":
        halos[f"r_200_angle_{rel}"] = (halos[f"r_200_{rel}"].to("kpc") * units.rad / halos["d_A"].to("kpc")).to(
            "arcsec")
        halos[f"r_perp_200_{rel}"] = (halos[f"offset_angle_mc"] / halos[f"r_200_angle_{rel}"]).decompose()
        halos[f"r_perp_200_{rel}_err_plus"] = u.uncertainty_product(
            halos[f"r_perp_200_{rel}"],
            (halos[f"offset_angle_mc"], halos[f"offset_angle_mean_K18_err_plus"]),
            (halos[f"r_200_mean_{rel}"], halos[f"r_200_mean_{rel}_err_plus"]),
        ).decompose()
        halos[f"r_perp_200_{rel}_err_minus"] = u.uncertainty_product(
            halos[f"r_perp_200_{rel}"],
            (halos[f"offset_angle_mc"], halos[f"offset_angle_mean_K18_err_minus"]),
            (halos[f"r_200_mean_{rel}"], halos[f"r_200_mean_{rel}_err_minus"]),
        ).decompose()

        rel_ = rel.replace(
            "1",
            "+1")

        halos[f"dm_halo_mc_mean_{rel}"] = halos[f"dm_halo_mc_{rel}"]
        halos[f"path_length_mean_{rel}"] = halos[f"path_length_mc_{rel}"]
        # halos[f"log_mass_halo_mean_{rel}"] = halos[f"log_mass_halo_mc_{rel}"]
        halos[f"r_200_mean_{rel}"] = halos[f"r_200_{rel}"]
        # halos[f"r_perp_200_mean_{rel}"] = halos[f"r_perp_200_{rel}"]

        col_dict.update({
            # f"log_mass_halo_{rel}": r"$\log{\Mhalo/\si{\solarmass}}$\phantom{" + rel[0].lower() + "}",
            f"log_mass_halo_mc_{rel}": r"$\log_{10}({\Mhalo/\si{\solarmass})}$",
            # f"r_200_{rel}": r"$R_{200}$\phantom{" + rel[0].lower() + "}",
            f"r_200_mean_{rel}": r"$R_{200}$",
            # f"dm_halo_{rel}": r"\DMHalo f \phantom{" + rel[0].lower() + "}",
            f"dm_halo_{rel}": "Fiducial",
            f"dm_halo_mc_mean_{rel}": "Mean",  # r"\DMHalo",
            f"dm_halo_mc_med_{rel}": r"Median",
            f"r_perp_200_{rel}": r"$R_\perp / \Rvir$",
            f"tau_halo_{rel}": r"$\tauHalo <$",
            f"ne_avg_{rel}": r"$\tavg{n_e}$",
            f"path_length_mean_{rel}": r"$L_\mathrm{FRB}$",
            f"g_scatt_{rel}": "$G_\mathrm{scatt}$",
            # fr"\Ftilde\times\DM{{}}^2"
        })
        sub_colnames.update({
            f"r_200_{rel}": r"kpc",
            f"r_200_mean_{rel}": r"kpc",
            f"dm_halo_{rel}": "pc\,cm$^{-3}$",
            f"dm_halo_mc_mean_{rel}": r"pc\,cm$^{-3}$",
            f"dm_halo_mc_med_{rel}": r"pc\,cm$^{-3}$",
            f"tau_halo_{rel}": r"ms",
            f"path_length_mean_{rel}": r"kpc",
            f"ne_avg_{rel}": "cm$^{-3}$",
        })
        tbl_halo = u.latexise_table(
            tbl=halos[
                "id_short",
                f"log_mass_halo_mc_{rel}",
                f"log_mass_halo_mc_{rel}_err",
                f"r_200_mean_{rel}",
                f"r_200_mean_{rel}_err_plus",
                f"r_200_mean_{rel}_err_minus",
                f"r_perp_200_{rel}",
                f"r_perp_200_{rel}_err_plus",
                f"r_perp_200_{rel}_err_minus",
                    # f"r_perp_200_mean_{rel}",
                    # f"r_perp_200_mean_{rel}_err_plus",
                    # f"r_perp_200_mean_{rel}_err_minus",
                f"dm_halo_{rel}",
                f"dm_halo_mc_mean_{rel}",
                f"dm_halo_mc_mean_{rel}_err_plus",
                f"dm_halo_mc_mean_{rel}_err_minus",
                    # "c",
                f"dm_halo_mc_med_{rel}",
                f"dm_halo_mc_med_{rel}_err_plus",
                f"dm_halo_mc_med_{rel}_err_minus",
            ],
            column_dict=col_dict,
            uncertainty_kwargs={"brackets": False, "n_digits_err": 2},
            sub_colnames=sub_colnames,
            round_dict={
                "c": 2,
                f"dm_halo_{rel}": 1
                # f"log_mass_halo_mean_{rel}": 2,
                # f"r_200_{rel}": 0,
                # f"dm_halo_{rel}": 1,
                # f"dm_halo_mc_med_{rel}": 1,
            },
            output_path=os.path.join(lib.latex_table_path, f"fg_{rel}.tex"),
            second_path=os.path.join(lib.latex_table_path_db, f"fg_{rel}.tex"),
            label=f"tab:fg_{rel}",
            short_caption="Halo properties of host and foreground galaxies for FRB 20190714A (" + rel_ + ")",
            caption="Halo properties of host and foreground galaxies for \FRB{20190714A}, using the " + rel_ + " stellar-to-halo-mass relationship. Uncertainties are the 68.27\% confidence intervals taken from the Monte Carlo analysis."
                        r" \tabscript{" + this_script + "}",
            multicolumn=((4, "c", ""), (3, "|c", "\DMHalo")),
            coltypes="cccc|ccc",
            positioning="t!"
        )

        tbl_scatt = u.latexise_table(
            tbl=halos[
                "id_short",
                f"tau_halo_{rel}",
                # f"ne_avg_{rel}",
                f"path_length_mean_{rel}",
                f"path_length_mean_{rel}_err_plus",
                f"path_length_mean_{rel}_err_minus",
                f"g_scatt_{rel}",
                f"g_scatt_{rel}_err_plus",
                f"g_scatt_{rel}_err_minus",
                # f"tau_halo_{rel}_err"
            ],
            column_dict=col_dict,
            uncertainty_kwargs={"brackets": False, "n_digits_err": 2},
            sub_colnames=sub_colnames,
            round_dict={
                f"tau_halo_{rel}": 6,
                f"ne_avg_{rel}": 6,
            },
            output_path=os.path.join(lib.latex_table_path, f"fg_scatter_{rel}.tex"),
            second_path=os.path.join(lib.latex_table_path_db, f"fg_scatter_{rel}.tex"),
            label=f"tab:fg_scattering_{rel}",
            short_caption="Scattering-related properties of host and foreground galaxies for FRB 20190714A (" + rel_ + ")",
            caption="Scattering-related properties of host and foreground galaxies for \FRB{20190714A}, using the " + rel.replace(
                "1",
                "+1") + rf" stellar-to-halo-mass relationship. The values in the \tauHalo{{}} column are estimated assuming a fixed $\FA<\SI{{\Flim{rel[0]}}}{{\Funits}}$."
                        r" \tabscript{" + this_script + "}",
        )

    # tbl_both = u.latexise_table(
    #     tbl=halos[
    #         "id_short",
    #         f"log_mass_halo_mc_K18",
    #         f"log_mass_halo_mc_K18_err",
    #         f"r_200_mc_K18",
    #         f"r_200_mc_K18_err",
    #         f"dm_halo_mc_K18",
    #         f"dm_halo_mc_K18_err",
    #         f"log_mass_halo_mc_M13",
    #         f"log_mass_halo_mc_M13_err",
    #         f"r_200_mc_M13",
    #         f"r_200_mc_M13_err",
    #         f"dm_halo_mc_M13",
    #         f"dm_halo_mc_M13_err",
    #     ],
    #     column_dict=col_dict,
    #     uncertainty_kwargs={"brackets": False, "n_digits_err": 2},
    #     sub_colnames=sub_colnames,
    #     round_dict={
    #         f"log_mass_halo_{rel}": 2,
    #         f"r_200_{rel}": 0,
    #         f"dm_halo_{rel}": 0,
    #     },
    #     output_path=os.path.join(lib.latex_table_path, f"fg_both.tex"),
    #     second_path=os.path.join(lib.latex_table_path_db, f"fg_both.tex"),
    #     label=f"tab:fg_{rel}",
    #     caption="Table of halo properties of host and foreground galaxies for \FRB{20190714A}. Uncertainties are the \\xsigma{1} intervals taken from the Monte Carlo analysis.",
    #     multicolumn=((1, "c|", ""), (3, "c", "K+18"), (3, "c|", "M+13")),
    #     coltypes="l|ccc|ccc"
    # )

    tbl_astrom = u.latexise_table(
        tbl=halos[
            "id_short",
                # "offset_angle",
            "offset_angle_mc",
            "offset_angle_mc_err",
            "z",
                # "r_perp",
            "r_perp_mc_K18",
            "r_perp_mc_K18_err",
            "r_perp_eff",
            "r_perp_eff_err",
            "ra",
                # "ra_err",
            "dec",
            # "dec_err",
            # "d_A",
        ],
        ra_col="ra",
        dec_col="dec",
        # ra_err_col="ra_err",
        # dec_err_col="dec_err",
        column_dict=col_dict,
        # round_cols=round_cols,
        round_dict=round_dict,
        round_digits=2,
        sub_colnames=sub_colnames,
        output_path=os.path.join(lib.latex_table_path, "fg_astrometry.tex"),
        second_path=os.path.join(lib.latex_table_path_db, "fg_astrometry.tex"),
        label="tab:fg_astrometry",
        short_caption="Astrometric properties of FRB 20190714A host and foreground galaxies",
        caption="Astrometric properties of host and foreground galaxies for \FRB{20190714A}."
                r" \tabscript{" + this_script + "}",
    )

    gal_tbl = halos[
        "id_short",
        "galfit_n",
        "galfit_n_err",
        "galfit_mag",
        "galfit_mag_err",
        "galfit_r_eff",
        "galfit_r_eff_err",
        "galfit_r_eff_proj",
        "galfit_r_eff_proj_err",
        "galfit_axis_ratio",
        "galfit_axis_ratio_err",
        "galfit_theta",
        "galfit_theta_err",
    ]

    remove = ("e", "i", "j", "p")
    gal_tbl = gal_tbl[[n[-1] not in remove for n in gal_tbl["id_short"]]]

    tbl_galfit = u.latexise_table(
        tbl=gal_tbl,
        # round_cols=round_cols,
        round_dict=round_dict,
        column_dict=col_dict,
        sub_colnames=sub_colnames,
        output_path=os.path.join(lib.latex_table_path, "fg_galfit.tex"),
        second_path=os.path.join(lib.latex_table_path_db, "fg_galfit.tex"),
        label="tab:fg_galfit",
        short_caption="\galfit{}-derived properties of FRB 20190714A host and foreground galaxies",
        caption="Table of \galfit{}-derived properties of host and foreground galaxies for \FRB{20190714A}. Some were not fitted well, and are omitted from this table."
                r" \tabscript{" + this_script + "}",
    )

    sub_colnames.update({
        "id_short": "(this work)",
    })

    tbl_misc = u.latexise_table(
        tbl=halos[
            "id_short",
            "id_cat",
            "sample",
            "log_mass_stellar",
            "log_mass_stellar_err",
        ],
        # round_cols=round_cols,
        round_dict=round_dict,
        column_dict=col_dict,
        sub_colnames=sub_colnames,
        output_path=os.path.join(lib.latex_table_path, "fg_misc.tex"),
        second_path=os.path.join(lib.latex_table_path_db, "fg_misc.tex"),
        label="tab:fg_misc",
        caption="Table of properties of host and foreground galaxies for \FRB{20190714A}."
                r" \tabscript{" + this_script + "}",
    )

    lib.write_master_table(tbl=halos)


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
