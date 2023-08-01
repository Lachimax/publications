#!/usr/bin/env python
# Code by Lachlan Marnoch, 2023

import os

import matplotlib.pyplot as plt

import craftutils.utils as u
from craftutils.observation.objects import set_cosmology
import craftutils.plotting as pl

import lib

description = """
Generates Figures 3+.
`04_redshift_hosts.py` must be run first.
"""


def main(
        output_dir: str,
        input_dir: str,
        cosmo: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)
    set_cosmology(cosmo)

    lib.load_prospector_files()
    lib.load_magnitude_tables()
    lib.load_p_z_dm()

    pl.latex_setup()

    fig, ax = plt.subplots(figsize=(lib.textwidth, lib.textwidth / 4))
    # Do some plotting
    ax.plot(
        lib.z_p_z_dm,
        lib.p_z_dm_best,
        label="$p(z|\mathrm{DM})$",
        lw=2,
        c="purple"
    )
    ax.set_xlabel("$z$")
    ax.set_xlim(0., 5)

    dirpath = os.path.join(output_path, "distributions")
    u.mkdir_check_nested(dirpath, False)
    fig.savefig(os.path.join(dirpath, f"p_z_dm_only.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(dirpath, f"p_z_dm_only.png"), bbox_inches="tight")

    for band in lib.bands_default + lib.bands_wise:
        print("=======================================================")
        print(band.machine_name())
        lib.band_mag_table(band)
        print()

    hosts_R = [
        "FRB20121102A",
        "FRB20210117A",
        "FRB20180924B",
        "FRB20190714A",
        # "FRB20181112",
        "FRB20210807D",
        "FRB20200430A",
        "FRB20190711A",
        "FRB20220105A",
        # "FRB20191001",
        # "FRB20210320",
        "FRB20211127I"
    ]

    hosts_K = [
        "FRB20121102A",
        # "FRB20210117A",
        "FRB20180924B",
        # "FRB20190714A",
        # "FRB20181112",
        # "FRB20210807D",
        # "FRB20200430A",
        "FRB20190711A",
        # "FRB20220105A",
        "FRB20191001A",
        # "FRB20210320",
        # "FRB20211127I",
        "FRB20190520B",
        "FRB20180301A",
        # "FRB20200906A",
    ]

    hosts_minimal = [
        "FRB20121102A",
        # "FRB20220105A",
        "FRB20190711A",
        # "FRB20210807",
        "FRB20211127I",
    ]

    lib.set_plot_properties(frbs=hosts_R + hosts_K)

    # ===============================================================================
    # K-band
    # Default
    lib.magnitude_redshift_plot(
        band=lib.K_hawki,
        frbs=hosts_K,
        draw_lstar=False,
        draw_observed_phot=False,
        suffix="selection_K",
        path_lim=False,
        do_legend=False,
        textwidth_factor=0.5,
    )
    # With all photometric points drawn on
    lib.magnitude_redshift_plot(
        band=lib.K_hawki,
        frbs=hosts_K,
        draw_lstar=False,
        draw_observed_phot=False,
        suffix="selection_K",
        path_lim=False,
        do_legend=False,
        textwidth_factor=0.5,
        do_other_photometry=True
    )

    # ===============================================================================
    # R-band
    # With L* lines
    lib.magnitude_redshift_plot(
        band=lib.R_fors2,
        frbs=hosts_minimal,
        draw_lstar=True,
        draw_observed_phot=False,
        suffix="minimal",
        textwidth_factor=0.5,
        do_legend=False,
    )
    # With L* lines and all photometry crosses
    lib.magnitude_redshift_plot(
        band=lib.R_fors2,
        frbs=hosts_minimal,
        draw_lstar=True,
        draw_observed_phot=False,
        suffix="minimal",
        textwidth_factor=0.5,
        do_legend=False,
        do_other_photometry=True
    )

    lib.magnitude_redshift_plot(
        band=lib.K_hawki,
        frbs=hosts_R,
        draw_lstar=False,
        draw_observed_phot=False,
        suffix="selection",
        textwidth_factor=0.5,
        path_lim=False,
        do_pdf_panel=False,
        do_legend=False
        # do_pdf_shading=True,
        # legend_frbs=hosts_R + hosts_K
        # color=colors
    )

    lib.magnitude_redshift_plot(
        band=lib.R_fors2,
        frbs=hosts_R,
        draw_lstar=False,
        draw_observed_phot=False,
        suffix="selection",
        path_lim=False,
        do_pdf_panel=True,
        # do_pdf_shading=True,
        legend_frbs=hosts_R + hosts_K
        # color=colors
    )

    lib.magnitude_redshift_plot(
        band=lib.R_fors2,
        frbs=hosts_R,
        draw_lstar=False,
        draw_observed_phot=False,
        suffix="selection_pdf_RKhosts",
        path_lim=False,
        do_pdf_panel=True,
        # do_pdf_shading=True,
        legend_frbs=hosts_R + hosts_K,
        # do_median=True,
        # do_mean=True,
        do_other_photometry=True
        # color=colors
    )

    lib.magnitude_redshift_plot(
        band=lib.R_fors2,
        frbs=hosts_R,
        draw_lstar=False,
        draw_observed_phot=False,
        suffix="selection_pdf",
        path_lim=False,
        do_pdf_panel=True,
        # do_pdf_shading=True,
        legend_frbs=hosts_R,  # + hosts_K,
        # do_median=True,
        # do_mean=True,
        do_other_photometry=True
        # color=colors
    )

    lib.magnitude_redshift_plot(
        band=lib.R_fors2,
        frbs=hosts_R,
        draw_lstar=False,
        draw_observed_phot=False,
        suffix="selection_pdf",
        path_lim=False,
        do_pdf_panel=True,
        # do_pdf_shading=True,
        legend_frbs=hosts_R,  # + hosts_K,
        # do_median=True,
        # do_mean=True,
        do_other_photometry=False
        # color=colors
    )

    lib.magnitude_redshift_plot(
        band=[lib.R_fors2, lib.K_hawki],
        suffix="all_twin",
        grey_lines=True,
        do_median=True,
        # do_mean=True,
        do_legend=False,
        n_panels=2,
        textwidth_factor=0.5
    )

    lib.magnitude_redshift_plot(
        band=[lib.R_fors2, lib.K_hawki],
        frbs=hosts_R,
        draw_lstar=False,
        draw_observed_phot=False,
        suffix="selection_twin",
        path_lim=False,
        do_pdf_panel=True,
        # do_pdf_shading=True,
        legend_frbs=hosts_R,  # + hosts_K,
        do_median=False,
        grey_lines=False,
        n_panels=2,
        # do_mean=True,
        do_other_photometry=False
        # color=colors
    )

    for band in lib.bands_default:
        lib.textwidth = pl.textheights["mqthesis"]
        lib.magnitude_redshift_plot(
            band=band,
            suffix="all",
            textwidth_factor=1.,
            height=pl.textwidths["mqthesis"]
        )
        lib.textwidth = pl.textwidths["MNRAS"]
        lib.magnitude_redshift_plot(
            band=band,
            suffix="all",
            grey_lines=True,
            do_median=True,
            # do_mean=True,
            do_legend=False
        )
        lib.magnitude_redshift_plot(
            band=band,
            suffix="all",
            grey_lines=True,
            do_median=True,
            # do_mean=True,
            do_legend=False,
            textwidth_factor=0.5
        )
        lib.magnitude_redshift_plot(
            band=band,
            frbs=hosts_R,
            draw_lstar=False,
            draw_observed_phot=False,
            suffix="selection",
            path_lim=False,
            # color=colors
        )

    for band in lib.bands_wise:
        lib.magnitude_redshift_plot(
            band=band,
            suffix="all",
            path_slug="WISE"
        )
        lib.magnitude_redshift_plot(
            band=band,
            frbs=hosts_R,
            draw_lstar=False,
            draw_observed_phot=False,
            suffix="selection",
            path_lim=False,
            path_slug="WISE"
            # color=colors
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
    parser.add_argument(
        "-c",
        help="Preferred cosmology. Should be listed in astropy.cosmology.available.",
        type=str,
        default="Planck18"
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        cosmo=args.c
    )
