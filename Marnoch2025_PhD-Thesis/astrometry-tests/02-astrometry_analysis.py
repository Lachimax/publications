#!/usr/bin/env python
# Code by Lachlan Marnoch, 2023-2025
import matplotlib.pyplot as plt
import os

import numpy as np

from astropy import table, units

import craftutils.utils as u
import craftutils.params as p
import craftutils.observation.epoch as epoch
import craftutils.plotting as pl

import lib

description = """
Performs analysis of the tests, generating tables and figures.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    pl.latex_setup()

    results_path = os.path.join(output_dir, "astrometry_tests_results.yaml")
    status_path = os.path.join(output_dir, "astrometry_tests_status.yaml")
    output_path_this = os.path.join(output_dir, "output")

    results = p.load_params(results_path)
    status = p.load_params(status_path)

    tbl_dicts = []

    exp_names = list(results[list(results.keys())[0]].keys())

    directory = epoch.load_epoch_directory()

    for epoch_name, epoch_dict in results.items():
        print(epoch_name)
        if "1-no_correction" not in epoch_dict:
            continue
        fils = epoch_dict["1-no_correction"]["astrometry"].keys()
        for fil_name in fils:
            successes = 0
            tbl_dict = {}
            tbl_dicts.append(tbl_dict)
            tbl_dict["epoch_number"] = epoch_name[-1]
            tbl_dict["field"] = directory[epoch_name]["field_name"]
            tbl_dict["filter"] = fil_name
            tbl_dict["n_stars_gaia"] = epoch_dict["1-no_correction"]["astrometry"][fil_name]["n_matches"]
            tbl_dict["n_stars_psf"] = epoch_dict["1-no_correction"]["psf"][fil_name]["n_stars"]
            for i, (exp_name, exp_dict) in enumerate(epoch_dict.items()):
                success = exp_name in status[epoch_name] and status[epoch_name][exp_name]["processing"] == "fine" and \
                          status[epoch_name][exp_name]["analysis"] == "fine"
                n = i + 1
                astrom_dict = exp_dict['astrometry']
                psf_dict = exp_dict['psf']
                fil_dict = astrom_dict[fil_name]
                offset = fil_dict["rms_offset"]
                psf = psf_dict[fil_name]["gauss"]["fwhm_median"]
                if exp_name == "5-astrometry_upload":
                    if tbl_dict[f"2_gaia_rms"] == offset:
                        success = False
                if success:
                    tbl_dict[f"{n}_gaia_rms"] = offset
                    tbl_dict[f"{n}_psf_fwhm"] = psf
                    successes += 1
                else:
                    tbl_dict[f"{n}_gaia_rms"] = np.nan * units.arcsec
                    tbl_dict[f"{n}_psf_fwhm"] = np.nan * units.arcsec
            tbl_dict["successes"] = successes

    astrom_tbl = table.QTable(tbl_dicts)

    filters = list(set(astrom_tbl["filter"]))
    filters.sort(key=lambda f: len(astrom_tbl[astrom_tbl["filter"] == f]), reverse=True)
    filters += ["combined"]
    fields = sorted(set(astrom_tbl["field"]))

    astrom_tbl_full = u.add_stats(tbl=astrom_tbl, name_col="field", cols_exclude=["epoch_number", "filter", "field"])
    astrom_tbl_full.write(os.path.join(output_path_this, "astrometry_tests_results_full.csv"), overwrite=True)
    astrom_tbl_full.write(os.path.join(output_path_this, "astrometry_tests_results_full.ecsv"), overwrite=True)

    figs_big = {}
    figs_med_ast = {}
    figs_med_psf = {}
    for fil_name in filters:
        figs_big[fil_name] = plt.figure(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] * 1.2))
        figs_med_ast[fil_name] = plt.figure(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] * 2 / 3))
        figs_med_psf[fil_name] = plt.figure(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] * 2 / 3))

    count_max_ast = {}
    count_max_psf = {}
    x_max_ast = {}
    x_max_psf = {}

    n_width = 3
    n_height = 4

    cs = {
        "u_HIGH": "violet",
        "g_HIGH": "green",
        "R_SPECIAL": "darkred",
        "I_BESS": "gray",
        "z_GUNN": "black",
        "combined": "purple"
    }

    for i, exp_name in enumerate(exp_names):
        fig_psf, ax_psf = plt.subplots(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"]))
        fig_ast, ax_ast = plt.subplots(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"]))
        n = i + 1
        c = chr(ord("A") + i)
        bins_psf = "auto"
        bins_ast = "auto"

        for fil_name in filters:
            print(fil_name)
            if fil_name == "combined":
                fil_tbl = astrom_tbl_full.copy()
            else:
                fil_tbl = astrom_tbl_full[astrom_tbl_full["filter"] == fil_name]
            fil_tbl = u.add_stats(tbl=fil_tbl, name_col="field", cols_exclude=["epoch_number", "filter", "field"])
            fil_tbl.write(os.path.join(output_path_this, f"astrometry_tests_results_{fil_name}.ecsv"), overwrite=True)
            fil_tbl.write(os.path.join(output_path_this, f"astrometry_tests_results_{fil_name}.csv"), overwrite=True)
            fil_tbl_select = fil_tbl[:1].copy()
            fil_tbl_select.remove_row(0)
            for fld in fields:
                field_tbl = fil_tbl[fil_tbl["field"] == fld]
                field_tbl.sort("n_stars_gaia", reverse=False)
                field_tbl.sort("successes", reverse=True)
                if len(field_tbl) > 0:
                    best_row = field_tbl[0]
                    fil_tbl_select.add_row(best_row)
            fil_tbl_select.write(
                os.path.join(output_path_this, f"astrometry_tests_results_{fil_name}_select.ecsv"),
                overwrite=True)
            fil_tbl_select.write(
                os.path.join(output_path_this, f"astrometry_tests_results_{fil_name}_select.csv"),
                overwrite=True)

            fig_big = figs_big[fil_name]

            # Big combined figure with individual histograms
            if n <= 3:
                n_fig = n
            else:
                n_fig = n + 3
            ax = fig_big.add_subplot(n_height, n_width, n_fig)
            ax.hist(
                fil_tbl_select[f"{n}_gaia_rms"][np.isfinite(fil_tbl_select[f"{n}_gaia_rms"])],
                bins="auto", density=False, label=f"${fil_name[0]}$",
                edgecolor="black",
                lw=2,
                color=cs[fil_name]
            )
            ax.text(0.85, 0.85, f"{c}.", transform=ax.transAxes)
            ax.set_xlabel("$\sigma_\mathrm{astrom}$ ($\prime\prime$)")

            ax = fig_big.add_subplot(n_height, n_width, n_fig + 3)
            ax.hist(
                fil_tbl_select[f"{n}_psf_fwhm"][np.isfinite(fil_tbl_select[f"{n}_psf_fwhm"])],
                bins="auto", density=False, label=f"${fil_name[0]}$",
                edgecolor="black",
                lw=2,
                color=cs[fil_name]
            )
            ax.set_xlim(left=0.)
            ax.text(0.85, 0.85, f"{c}.", transform=ax.transAxes)
            ax.set_xlabel("FWHM ($\prime\prime$)")

            # Separate psf and astrom figures with individual histograms
            n_x = n % n_width
            is_left = (n_x == 1 or n_width == 1)
            is_bottom = n > n_width

            fig_med_ast = figs_med_ast[fil_name]
            ax = fig_med_ast.add_subplot(2, n_width, n)
            counts, bins, _ = ax.hist(
                fil_tbl_select[f"{n}_gaia_rms"][np.isfinite(fil_tbl_select[f"{n}_gaia_rms"])],
                bins="auto", density=False, label=f"${fil_name[0]}$",
                edgecolor=cs[fil_name],
                # lw=1,
                color=cs[fil_name]
            )
            ax.text(0.85, 0.85, f"{c}.", transform=ax.transAxes)
            ax.set_xlabel(" ")
            max_counts = np.max(counts)
            if fil_name not in count_max_ast or max_counts > count_max_ast[fil_name]:
                count_max_ast[fil_name] = max_counts
            max_x = np.max(bins)
            if fil_name not in x_max_ast or max_x > x_max_ast[fil_name]:
                x_max_ast[fil_name] = max_x
            if not is_bottom:
                ax.set_xticks([], labels=[], visible=False)
            if not is_left:
                ax.set_yticks([], labels=[], visible=False)

            fig_med_psf = figs_med_psf[fil_name]
            ax = fig_med_psf.add_subplot(2, n_width, n)
            counts, bins, _ = ax.hist(
                fil_tbl_select[f"{n}_psf_fwhm"][np.isfinite(fil_tbl_select[f"{n}_psf_fwhm"])],
                bins="auto", density=False, label=f"${fil_name[0]}$",
                edgecolor=cs[fil_name],
                # lw=1,
                color=cs[fil_name]
            )
            ax.text(0.85, 0.85, f"{c}.", transform=ax.transAxes)
            ax.set_xlabel(" ")
            max_counts = np.max(counts)
            if fil_name not in count_max_psf or max_counts > count_max_psf[fil_name]:
                count_max_psf[fil_name] = max_counts
            max_x = np.max(bins)
            if fil_name not in x_max_psf or max_x > x_max_psf[fil_name]:
                x_max_psf[fil_name] = max_x
            if not is_bottom:
                ax.set_xticks([], labels=[], visible=False)
            if not is_left:
                ax.set_yticks([], labels=[], visible=False)

            # All hists on the same axes

            _, bins_ast, _ = ax_ast.hist(
                fil_tbl_select[f"{n}_gaia_rms"][np.isfinite(fil_tbl_select[f"{n}_gaia_rms"])],
                bins=bins_ast, density=False, label=f"${fil_name[0]}$", alpha=0.5,
                edgecolor="black",
                lw=2,
                color=cs[fil_name]
            )
            ax_ast.set_xlim(left=0.)

            _, bins_psf, _ = ax_psf.hist(
                fil_tbl_select[f"{n}_psf_fwhm"][np.isfinite(fil_tbl_select[f"{n}_psf_fwhm"])],
                bins=bins_psf, density=False, label=f"${fil_name[0]}$", alpha=0.5,
                edgecolor="black",
                lw=2,
                color=cs[fil_name]
            )
            ax_psf.set_xlim(left=0.)

        ax_ast.set_xlabel("$\sigma_\mathrm{astrom}$ ($\prime\prime$)")
        fig_ast.suptitle(f"{exp_name}")
        ax_ast.legend(loc=(1.1, 0))
        fig_ast.savefig(os.path.join(output_path_this, f"{exp_name}_astrom.pdf"), bbox_inches="tight")
        fig_ast.savefig(os.path.join(output_path_this, f"{exp_name}_astrom.png"), bbox_inches="tight")
        plt.close(fig_ast)

        ax_psf.set_xlabel("PSF FWHM ($\prime\prime$)")
        fig_psf.suptitle(f"{exp_name}")
        ax_psf.legend(loc=(1.1, 0))
        fig_psf.savefig(os.path.join(output_path_this, f"{exp_name}_psf.pdf"), bbox_inches="tight")
        fig_psf.savefig(os.path.join(output_path_this, f"{exp_name}_psf.png"), bbox_inches="tight")
        plt.close(fig_psf)

    for fil_name, fig in figs_big.items():
        fig.tight_layout()
        fig.savefig(os.path.join(output_path_this, f"{fil_name}_all.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(lib.dropbox_path, "figures", f"{fil_name}_astrometry.pdf"), bbox_inches="tight")
        plt.close(fig)


    for fil_name, fig in figs_med_ast.items():
        # fig.tight_layout()
        for ax in fig.get_axes():
            ax.set_xlim(left=0., right=x_max_ast[fil_name] + 0.01)
            ax.set_ylim(bottom=0, top=count_max_ast[fil_name] + 0.1)
        ax_big = fig.add_subplot(1, 1, 1)
        ax_big.set_xlabel("$\sigma_\mathrm{astrom}$ (arcseconds)", labelpad=30.)
        ax_big.set_frame_on(False)
        ax_big.tick_params(left=False, right=False, top=False, bottom=False)
        ax_big.yaxis.set_ticks([])
        ax_big.xaxis.set_ticks([])
        fig.savefig(os.path.join(output_path_this, f"{fil_name}_astrom.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(lib.dropbox_path, "figures", f"{fil_name}_astrometry_sigast.pdf"), bbox_inches="tight")
        plt.close(fig)

    for fil_name, fig in figs_med_psf.items():
        # fig.tight_layout()
        for ax in fig.get_axes():
            ax.set_xlim(left=0., right=x_max_psf[fil_name] + 0.01)
            ax.set_ylim(bottom=0, top=count_max_psf[fil_name] + 0.1)
        ax_big = fig.add_subplot(1, 1, 1)
        ax_big.set_xlabel("PSF FWHM (arcseconds)", labelpad=30.)
        ax_big.set_frame_on(False)
        ax_big.tick_params(left=False, right=False, top=False, bottom=False)
        ax_big.yaxis.set_ticks([])
        ax_big.xaxis.set_ticks([])
        fig.savefig(os.path.join(output_path_this, f"{fil_name}_psf.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(lib.dropbox_path, "figures", f"{fil_name}_astrometry_psf.pdf"), bbox_inches="tight")
        plt.close(fig)

    astrom_tbl_select = astrom_tbl[:1].copy()
    astrom_tbl_select.remove_row(0)
    for fld in fields:
        field_tbl = astrom_tbl[astrom_tbl["field"] == fld]
        field_tbl.sort("n_stars_gaia", reverse=False)
        field_tbl.sort("successes", reverse=True)
        print("\t", fld)
        for row in field_tbl:
            print("\t\t", row["epoch_number"], row["successes"], row["n_stars_gaia"])

        if "g_HIGH" in field_tbl["filter"] and "R_SPECIAL" in field_tbl["filter"]:
            best_row_r = field_tbl[field_tbl["filter"] == "R_SPECIAL"][0]
            best_row_g = field_tbl[field_tbl["filter"] == "g_HIGH"][0]
            if best_row_r["successes"] > best_row_g["successes"]:
                best_fil = "R_SPECIAL"
            elif best_row_r["successes"] < best_row_g["successes"]:
                best_fil = "g_HIGH"
            else:
                if best_row_r["n_stars_gaia"] > best_row_g["n_stars_gaia"]:
                    best_fil = "R_SPECIAL"
                else:
                    best_fil = "g_HIGH"
        elif "g_HIGH" in field_tbl["filter"]:
            best_fil = "g_HIGH"
        else:
            best_fil = "R_SPECIAL"
        best_row = field_tbl[field_tbl["filter"] == best_fil][0]
        astrom_tbl_select.add_row(best_row)

    astrom_tbl_select["filter"] = [f"${f[0]}$" for f in astrom_tbl_select["filter"]]

    astrom_tbl_select = u.add_stats(
        table.QTable(astrom_tbl_select), name_col="field", cols_exclude=["epoch_number", "filter", "field"])
    astrom_tbl_select.write(os.path.join(output_path_this, "astrometry_tests_results_select.csv"), overwrite=True)
    astrom_tbl_select.write(os.path.join(output_path_this, "astrometry_tests_results_select.ecsv"), overwrite=True)

    col_dict = {
        "field": "Field",
        "epoch_number": "Epoch",
        "filter": "Filter",
        "n_stars_gaia": "Ast.",
        "n_stars_psf": "PSF",
    }
    sub_cols = {
        # "n_stars_gaia": "$N_\star$",
        # "n_stars_psf": "$N_\star$",
    }
    round_cols = []
    astrom_tbl_select.remove_column("successes")
    for col in astrom_tbl_select.colnames:
        if col not in col_dict:
            n = col[0]
            if col.endswith("_gaia_rms"):
                col_dict[col] = r"\vphantom{" + n + r"}\sigmaast{}"
            elif col.endswith("_psf_fwhm"):
                col_dict[col] = r"\vphantom{" + n + r"}PSF"
                sub_cols[col] = "FWHM"
            round_cols.append(col)

    astrom_tbl_all = astrom_tbl_select[list(col_dict.keys())]

    u.latexise_table(
        tbl=astrom_tbl_all,
        label="tab:pipeline:astrometry-comparison",
        landscape=True,
        longtable=True,
        column_dict=col_dict,
        output_path=os.path.join(output_path_this, "astrometry_tests.tex"),
        second_path=os.path.join(lib.dropbox_path, "tables", "astrometry_tests.tex"),
        multicolumn=[(3, "c|", ""), (2, "c|", "$N_\star$"),
                     (2, "c|", f"1."), (2, "c|", f"2."),
                     (2, "c|", f"3."), (2, "c|", f"4."),
                     (2, "c|", f"5."), (2, "c|", f"6.")],
        coltypes="ccc|cc|cc|cc|cc|cc|cc|cc",
        sub_colnames=sub_cols,
        round_cols=round_cols,
        round_digits=3,
        short_caption="Results of astrometry tests.",
        caption=r"Results of astrometry tests. In this table, one epoch was selected to represent each field, prioritising $g$ or $R$ band."
                r" The numbered columns correspond to the six scenarios enumerated in \autoref{pipeline:astrometry:comparison}. The $N_\star$ columns give the number of stars used in calculations. \tabscript{" + u.latex_sanitise(
            os.path.basename(__file__)) + "}",
    )

    col_dict = {
        "field": "Field",
        "epoch_number": "Epoch",
        "filter": "Filter",
        "n_stars_gaia": "$N_\star$",
    }
    sub_cols = {}
    round_cols = []
    for col in astrom_tbl_select.colnames:
        if col not in col_dict:
            n = col[0]
            if col.endswith("_gaia_rms"):
                col_dict[col] = str(n) + "."
                # elif col.endswith("_psf_fwhm"):
                #     col_dict[col] = r"\vphantom{" + n + r"}PSF"
                #     sub_cols[col] = "FWHM"
                round_cols.append(col)

    astrom_tbl_offsets = astrom_tbl_select[list(col_dict.keys())]

    u.latexise_table(
        tbl=astrom_tbl_offsets,
        label="tab:pipeline:astrometry-offsets",
        # landscape=True,
        # longtable=True,
        column_dict=col_dict,
        output_path=os.path.join(output_dir, "astrometry_tests_offsets.tex"),
        second_path=os.path.join(lib.dropbox_path, "tables", "astrometry_test_offsets.tex"),
        coltypes="ccc|c|ccccccccccccc",
        round_cols=round_cols,
        round_digits=3,
        short_caption=r"\textit{Gaia} offsets from astrometry tests.",
        caption=r"\textit{Gaia} offset RMS from astrometry tests. The numbered columns correspond to the six scenarios enumerated in \autoref{pipeline:astrometry:comparison}. The $N_\star$ column gives the number of stars used in statistical calculations; all other quantities are in arcseconds. \tabscript{" + u.latex_sanitise(
            os.path.basename(__file__)) + "}",
    )

    col_dict = {
        "field": "Field",
        "epoch_number": "Epoch",
        "filter": "Filter",
        "n_stars_psf": "$N_\star$",
    }
    sub_cols = {}
    round_cols = []
    for col in astrom_tbl_select.colnames:
        if col not in col_dict:
            n = col[0]
            if col.endswith("_psf_fwhm"):
                col_dict[col] = str(n) + "."
                round_cols.append(col)

    astrom_tbl_offsets = astrom_tbl_select[list(col_dict.keys())]

    u.latexise_table(
        tbl=astrom_tbl_offsets,
        label="tab:pipeline:astrometry-psf",
        # landscape=True,
        # longtable=True,
        column_dict=col_dict,
        output_path=os.path.join(output_dir, "astrometry_tests_psf.tex"),
        second_path=os.path.join(lib.dropbox_path, "tables", "astrometry_test_psf.tex"),
        coltypes="ccc|c|ccccccccccccc",
        round_cols=round_cols,
        round_digits=3,
        short_caption=r"PSF measurements from astrometry tests.",
        caption=r"PSF FWHM measurements from astrometry tests. The numbered columns correspond to the six scenarios enumerated in \autoref{pipeline:astrometry:comparison}. The $N_\star$ column gives the number of stars used in statistical calculations (that is, the number successfully fitted); all other quantities are in arcseconds. \tabscript{" + u.latex_sanitise(
            os.path.basename(__file__)) + "}",
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
        default=lib.default_data_dir
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
