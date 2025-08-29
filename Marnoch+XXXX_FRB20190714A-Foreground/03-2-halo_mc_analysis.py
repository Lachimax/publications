#!/usr/bin/env python
# Code by Lachlan Marnoch, 2025

import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.table as table
import astropy.units as units

import craftutils.utils as u
from hypothesis.extra.cli import obj_name

import lib

description = """
Analyses the collated results of the MC modelling.
"""


def main(
        output_dir: str,
        input_dir: str,
        skip_plots: bool,
        skip_individual: bool,
        n_real: float
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb190714 = lib.fld

    bin_colour = "violet"

    comm_dict = {}
    tbl_dict = {}

    halo_tbl = lib.read_master_table()
    props_main = lib.read_master_properties()

    for rel in ["K18", "M13"]:

        print()
        print("=" * 50)
        print(rel)

        prop_tbl = lib.read_properties_table_mc(relationship=rel)

        # n_real = len(prop_tbl)

        dm_halos_fiducial = props_main[f"dm_halos_emp_{rel}_fiducial"]

        dm_halos_mean = np.nanmean(prop_tbl["dm_halos_inclusive"])
        dm_halos_max = np.nanmax(prop_tbl["dm_halos_inclusive"])
        dm_halos_min = np.nanmin(prop_tbl["dm_halos_inclusive"])
        dm_halos_std = np.nanstd(prop_tbl["dm_halos_inclusive"])
        dm_halos_median = np.nanmedian(prop_tbl["dm_halos_inclusive"])
        d = 68.27 / 2
        dm_halos_upper = np.nanpercentile(prop_tbl["dm_halos_inclusive"], q=50 + d)
        dm_halos_lower = np.nanpercentile(prop_tbl["dm_halos_inclusive"], q=50 - d)

        dm_exgal = props_main["dm_exgal"]
        dm_igm = props_main["dm_igm_avg"]
        dm_budget = dm_exgal - dm_igm - lib.dm_host_ism_lb
        dm_budget_flimflam = dm_exgal - lib.dm_igm_flimflam - lib.dm_host_ism_lb

        if not skip_plots:
            fig, ax = plt.subplots()
            counts, bins, _ = ax.hist(
                prop_tbl["dm_halos_inclusive"],
                color=bin_colour, bins="auto", lw=1,
                edgecolor=bin_colour,
                # density=True
            )

            ylim = ax.get_ylim()
            ax.plot(
                (dm_halos_mean.value, dm_halos_mean.value),
                ylim,
                c="black",
                label=f"Mean = {int(np.round(dm_halos_mean.value))}\ pc\,cm$^{{-3}}$",
                lw="2",
            )
            ax.plot(
                (dm_halos_median.value, dm_halos_median.value),
                ylim,
                c="blue", ls=":",
                label=f"Median = {int(np.round(dm_halos_median.value))}\ pc\,cm$^{{-3}}$", lw="2"
            )
            ax.plot(
                (dm_halos_max.value, dm_halos_median.value),
                ylim,
                c="none", label=f"Max = {int(np.round(dm_halos_max.value))}\ pc\,cm$^{{-3}}$", lw="2"
            )
            ax.plot(
                (dm_halos_min.value, dm_halos_median.value),
                ylim,
                c="none", label=f"Min = {int(np.round(dm_halos_min.value))}\ pc\,cm$^{{-3}}$", lw="2"
            )
            ax.plot(
                (dm_halos_std.value, dm_halos_std.value),
                ylim,
                c="none",
                label=f"$\sigma$ = {int(np.round(dm_halos_std.value))}\ pc\,cm$^{{-3}}$", lw="2"
            )
            ax.plot(
                (frb190714.frb.dm.value, frb190714.frb.dm.value),
                ylim,
                c="green", ls="-.",
                label="$\mathrm{DM_{FRB}}$", lw="2"
            )
            ax.plot(
                (dm_exgal.value, dm_exgal.value),
                ylim,
                c="red", ls="--",
                label="$\mathrm{DM_{exgal}}$", lw="2"
            )
            ax.plot(
                (dm_budget.value, dm_budget.value),
                ylim,
                c="cyan", ls=":",
                label=r"$\mathrm{DM_{budget}^{halos}}$ with $\langle\mathrm{DM_{cosmic}^{IGM}}\rangle$",
                lw="3"
            )
            ax.plot(
                (dm_budget_flimflam.value, dm_budget_flimflam.value),
                ylim,
                c="orange", ls="-.",
                label=r"$\mathrm{DM_{budget}^{halos}}$ with FLIMFLAM",
                lw="3"
            )
            ax.plot(
                (dm_halos_fiducial.value, dm_halos_fiducial.value),
                ylim,
                c="purple",
                label=f"Fiducial $\mathrm{{DM_{{cosmic}}^{{halos}}}}$ = {int(np.round(dm_halos_fiducial.value))}\ pc\,cm$^{{-3}}$",
                lw=3
            )
            ax.fill_betweenx(
                ylim,
                (dm_halos_upper.value, dm_halos_upper.value),
                (dm_halos_lower.value, dm_halos_lower.value),
                color="black",
                lw=0,
                alpha=0.5,
                zorder=-2,
                label="68.27\% region"
            )

            ax.set_ylim(ylim)
            index_max = np.max(np.argwhere(counts > 3))
            ax.tick_params(axis='both', labelsize=12)
            ax.set_xlim(0, bins[index_max + 1])
            # ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
            ax.legend(loc="upper right", bbox_to_anchor=(1.01, 1), fontsize=12)
            ax.set_ylabel("N")
            ax.set_xlabel("Modelled $\mathrm{DM_{cosmic}^{halos}}$")
            fig.tight_layout()
            lib.savefig(
                fig=fig,
                filename=f"dm_halos_dist_{rel}_{n_real}",
                subdir="mc"
            )

            # All together

            fig, ax = plt.subplots()
            dms_total = prop_tbl["dm_halos_inclusive"] + props_main["dm_mw"] + props_main["dm_igm_avg"] + props_main[
                "dm_host_ism"]
            counts, bins, _ = ax.hist(
                dms_total.value,
                color=bin_colour, bins="auto", lw=1,
                edgecolor=bin_colour,
                # density=True
            )

            ylim = ax.get_ylim()
            ax.plot(
                (np.mean(dms_total).value, np.mean(dms_total).value),
                ylim,
                c="black", label=f"Mean = {int(np.round(np.mean(dms_total).value))}\ pc\ cm$^{{-3}}$", lw="2"
            )
            ax.plot(
                (np.median(dms_total).value, np.median(dms_total).value),
                ylim,
                c="blue", label=f"Median = {int(np.round(np.median(dms_total).value))}\ pc\ cm$^{{-3}}$", lw="2"
            )
            ax.plot(
                (np.max(dms_total).value, np.max(dms_total).value),
                ylim,
                c="none", label=f"Max = {int(np.round(np.max(dms_total).value))}\ pc\ cm$^{{-3}}$", lw="2"
            )
            ax.plot(
                (np.min(dms_total).value, np.min(dms_total).value),
                ylim,
                c="none", label=f"Min = {int(np.round(np.min(dms_total).value))}\ pc\ cm$^{{-3}}$", lw="2"
            )
            ax.plot(
                (np.std(dms_total).value, np.std(dms_total).value),
                ylim,
                c="none", label=f"$\sigma$ = {int(np.round(np.std(dms_total).value))}\ pc\ cm$^{{-3}}$", lw="2"
            )
            ax.plot(
                (frb190714.frb.dm.value, frb190714.frb.dm.value),
                ylim,
                c="limegreen", label="$\mathrm{DM_{FRB}}$", lw="2"
            )
            # ax.plot(
            #     (dm_exgal.value, dm_exgal.value),
            #     ylim,
            #     c="red", label="$\mathrm{DM_{exgal}}$", lw="2"
            # )
            # ax.plot(
            #     (dm_budget.value, dm_budget.value),
            #     ylim,
            #     c="cyan", label=r"$\mathrm{DM_{budget}}$ with $\langle\mathrm{DM_{IGM}^{cosmic}}\rangle$", lw="2"
            # )
            # ax.plot(
            #     (dm_budget_flimflam.value, dm_budget_flimflam.value),
            #     ylim,
            #     c="orange", label=r"$\mathrm{DM_{budget}}$ with FLIMFLAM", lw="2"
            # )
            ax.set_ylim(ylim)
            index_max = np.max(np.argwhere(counts > 3))
            ax.set_xlim(0, bins[index_max + 1])
            ax.legend(loc="upper right")
            ax.set_ylabel("N")
            ax.set_xlabel("$\mathrm{DM^{modelled}_{FRB}}$")
            ax.tick_params(axis='both', labelsize=12)
            lib.savefig(
                fig=fig,
                filename=f"dm_halos_dist_total_{rel}_{n_real}",
                subdir="mc"
            )

            if not skip_individual:
                prop_tbls = {}
                for row in halo_tbl:
                    obj_name = row["id"]
                    prop_tbl = lib.read_halo_individual_table_collated_mc(
                        obj_id=obj_name,
                        relationship=rel
                    )
                    prop_tbls[obj_name] = prop_tbl

                for key in prop_tbl.colnames:

                    for row in halo_tbl:
                        obj_name = row["id"]

                        prop_tbl = prop_tbls[obj_name]

                        q = prop_tbl[key]
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        ax.hist(
                            q,
                            bins="auto",
                            color=bin_colour,
                            lw=1,
                            edgecolor=bin_colour,
                        )
                        ax.set_xlabel(key)
                        ax.set_ylabel("N")
                        ax.tick_params(axis='both', labelsize=12)
                        lib.savefig(fig, filename=f"{obj_name}_{key}_{rel}", subdir="mc/individual")

        rel_ = {"M13": "Moster", "K18": "Kravtsov"}[rel]

        mc_collated = lib.read_halo_table_mc_collated(rel)
        # print("")

        mc_collated.sort("name")
        mc_collated["dm_halo_mc"] = mc_collated["dm_halo"]
        mc_collated["dm_halo_mc_med"] = mc_collated["dm_halo_med"]
        mc_collated["dm_halo_mc_upper"] = mc_collated["dm_halo_upper"]
        mc_collated["dm_halo_mc_lower"] = mc_collated["dm_halo_lower"]
        mc_collated["dm_halo_mc_med_err_minus"] = mc_collated["dm_halo_med_err_minus"]
        mc_collated["dm_halo_mc_med_err_plus"] = mc_collated["dm_halo_med_err_plus"]
        mc_collated["dm_halo_mc_mean_err_minus"] = mc_collated["dm_halo_mean_err_minus"]
        mc_collated["dm_halo_mc_mean_err_plus"] = mc_collated["dm_halo_mean_err_plus"]
        mc_collated["dm_halo_mc_err"] = mc_collated["dm_halo_err"]
        mc_collated["offset_angle_mc"] = mc_collated["offset_angle"]
        mc_collated["offset_angle_mc_err"] = mc_collated["offset_angle_err"]
        mc_collated["r_perp_mc"] = mc_collated["r_perp"]
        mc_collated["r_perp_mc_err"] = mc_collated["r_perp_err"]
        mc_collated["r_200_mc"] = mc_collated["r_200"]
        mc_collated["r_200_mc_err"] = mc_collated["r_200_err"]
        mc_collated["log_mass_halo_mc"] = mc_collated["log_mass_halo"]
        mc_collated["log_mass_halo_mc_err"] = mc_collated["log_mass_halo_err"]
        mc_collated["path_length_mc"] = mc_collated["path_length"]
        mc_collated["path_length_mc_err"] = mc_collated["path_length_err"]
        mc_collated["mass_halo_mc"] = units.solMass * 10 ** mc_collated["log_mass_halo_mc"]

        dm_halo_host = 0. * units.dm
        dm_halo_host_err_plus = 0. * units.dm
        dm_halo_host_err_minus = 0. * units.dm

        for gal in mc_collated:
            gal_id = lib.latex_gal_id(gal['name'])
            comm_dict[f"{gal_id}DMHalo{rel_}"] = int(np.round(gal["dm_halo"].value))
            comm_dict[f"{gal_id}DMHalo{rel_}sig"] = int(np.round(gal["dm_halo_err"].value))
            comm_dict[f"{gal_id}DMHalo{rel_}errplus"] = int(np.round(gal["dm_halo_mc_mean_err_plus"].value))
            comm_dict[f"{gal_id}DMHalo{rel_}errminus"] = int(np.round(gal["dm_halo_mc_mean_err_minus"].value))
            comm_dict[f"{gal_id}LogMassHalo{rel_}"] = np.round(gal["log_mass_halo"].value, 2)
            comm_dict[f"{gal_id}LogMassHalo{rel_}sig"] = np.round(gal["log_mass_halo_err"].value, 2)
            comm_dict[f"{gal_id}Rvir{rel_}"] = int(np.round(gal["r_200"].value))
            comm_dict[f"{gal_id}Rvir{rel_}sig"] = int(np.round(gal["r_200_err"].value))
            if rel == "K18":
                comm_dict[f"{gal_id}Offset"] = np.round(gal["offset_angle"].value, 2)
                comm_dict[f"{gal_id}Offsetsig"] = np.round(gal["offset_angle_err"].value, 2)
                comm_dict[f"{gal_id}Rperp"] = np.round(gal["r_perp"].value, 2)
                comm_dict[f"{gal_id}Rperpsig"] = np.round(gal["r_perp_err"].value, 2)
            if gal_id == "HG":
                dm_halo_host = gal["dm_halo"]
                dm_halo_host_err_plus = gal["dm_halo_mc_mean_err_plus"]
                dm_halo_host_err_minus = gal["dm_halo_mc_mean_err_minus"]

        mc_collated.remove_columns(
            [
                "name",
                "dm_halo", "dm_halo_err", "dm_halo_upper", "dm_halo_lower",
                "dm_halo_med_err_minus", "dm_halo_med_err_plus",
                "dm_halo_mean_err_minus", "dm_halo_mean_err_plus",
                "log_mass_halo", "log_mass_halo_err",
                "r_perp", "r_perp_err",
                "log_mass_stellar", "log_mass_stellar_err",
                "path_length", "path_length_err",
                "offset_angle", "offset_angle_err"
            ]
        )

        props_main.update({
            f"dm_halos_mc_{rel}_min": dm_halos_min,
            f"dm_halos_mc_{rel}_max": dm_halos_max,
            f"dm_halos_mc_{rel}_mean": dm_halos_mean,
            f"dm_halos_mc_{rel}_median": dm_halos_median,
            f"dm_halos_mc_{rel}_std": dm_halos_std,
            f"dm_halos_mc_{rel}_upper": dm_halos_upper,
            f"dm_halos_mc_{rel}_lower": dm_halos_lower,
            f"dm_halos_mc_{rel}_med_err_plus": dm_halos_upper - dm_halos_median,
            f"dm_halos_mc_{rel}_med_err_minus": dm_halos_median - dm_halos_lower,
            f"dm_halos_mc_{rel}_mean_err_plus": dm_halos_upper - dm_halos_mean,
            f"dm_halos_mc_{rel}_mean_err_minus": dm_halos_mean - dm_halos_lower,
            f"dm_host_mc_{rel}": props_main["dm_host_ism"] + dm_halo_host,
            f"dm_host_mc_{rel}_err_plus": np.sqrt(props_main["dm_host_ism_err"] ** 2 + dm_halo_host_err_plus ** 2),
            f"dm_host_mc_{rel}_err_minus": np.sqrt(props_main["dm_host_ism_err"] ** 2 + dm_halo_host_err_minus ** 2),
        })

        props_main.update({
            f"dm_cosmic_mc_{rel}": props_main["dm_igm_avg"] + dm_halos_mean,
            f"dm_cosmic_mc_{rel}_err_plus": np.sqrt(
                props_main[f"dm_halos_mc_{rel}_mean_err_plus"] ** 2 + props_main["dm_igm_avg_err"] ** 2),
            f"dm_cosmic_mc_{rel}_err_minus": np.sqrt(
                props_main[f"dm_halos_mc_{rel}_mean_err_minus"] ** 2 + props_main["dm_igm_avg_err"] ** 2),
            f"dm_cosmic_mc_flimflam_{rel}": lib.dm_igm_flimflam + dm_halos_mean,
            f"dm_cosmic_mc_flimflam_{rel}_err_plus": np.sqrt(
                props_main[f"dm_halos_mc_{rel}_mean_err_plus"] ** 2 + lib.dm_igm_flimflam_err_plus ** 2),
            f"dm_cosmic_mc_flimflam_{rel}_err_minus": np.sqrt(
                props_main[f"dm_halos_mc_{rel}_mean_err_minus"] ** 2 + lib.dm_igm_flimflam_err_minus ** 2),
            f"dm_modelled_mc_{rel}": props_main["dm_mw"] + dm_halos_mean + props_main["dm_igm_avg"] +
                                     props_main[
                                         "dm_host_ism"],
            f"dm_modelled_mc_{rel}_err_plus": np.sqrt(props_main[
                                                          "dm_mw_err"] ** 2 + props_main[
                                                          f"dm_halos_mc_{rel}_mean_err_plus"] ** 2 + props_main[
                                                          "dm_igm_avg_err"] ** 2 + props_main[
                                                          "dm_host_ism_err"] ** 2
                                                      ),
            f"dm_modelled_mc_{rel}_err_minus": np.sqrt(props_main[
                                                           "dm_mw_err"] ** 2 + props_main[
                                                           f"dm_halos_mc_{rel}_mean_err_minus"] ** 2 + props_main[
                                                           "dm_igm_avg_err"] ** 2 + props_main[
                                                           "dm_host_ism_err"] ** 2
                                                       ),
            f"dm_modelled_mc_flimflam_{rel}": props_main[
                                                  "dm_mw"] + dm_halos_mean + lib.dm_igm_flimflam +
                                              props_main[
                                                  "dm_host_ism"],
            f"dm_modelled_mc_flimflam_{rel}_err_plus": np.sqrt(
                props_main[
                    "dm_mw_err"] ** 2 + props_main[
                    f"dm_halos_mc_{rel}_mean_err_plus"] ** 2 + lib.dm_igm_flimflam_err_plus ** 2 + props_main[
                    "dm_host_ism_err"] ** 2
            ),
            f"dm_modelled_mc_flimflam_{rel}_err_minus": np.sqrt(
                props_main[
                    "dm_mw_err"] ** 2 + props_main[
                    f"dm_halos_mc_{rel}_mean_err_minus"] ** 2 + lib.dm_igm_flimflam_err_minus ** 2 + props_main[
                    "dm_host_ism_err"] ** 2),
        })

        comm_dict.update({
            f"FRBDMhalos{rel_}Min": int(np.round(dm_halos_min.value)),
            f"FRBDMhalos{rel_}Max": int(np.round(dm_halos_max.value)),
            f"FRBDMhalos{rel_}Mean": int(np.round(dm_halos_mean.value)),
            f"FRBDMhalos{rel_}Median": int(np.round(dm_halos_median.value)),
            f"FRBDMhalos{rel_}Std": int(np.round(dm_halos_std.value)),
            f"FRBDMhalos{rel_}ErrPlus": int(np.round(props_main[f"dm_halos_mc_{rel}_mean_err_plus"].value)),
            f"FRBDMhalos{rel_}ErrMinus": int(np.round(props_main[f"dm_halos_mc_{rel}_mean_err_minus"].value)),
            f"FRBDMcosmic{rel_}": int(np.round(props_main[f"dm_cosmic_mc_{rel}"].value)),
            f"FRBDMcosmic{rel_}ErrPlus": int(np.round(props_main[f"dm_cosmic_mc_{rel}_err_plus"].value)),
            f"FRBDMcosmic{rel_}ErrMinus": int(np.round(props_main[f"dm_cosmic_mc_{rel}_err_minus"].value)),
            f"FRBDMcosmicFLIMFLAM{rel_}": int(np.round(props_main[f"dm_cosmic_mc_flimflam_{rel}"].value)),
            f"FRBDMcosmicFLIMFLAM{rel_}ErrPlus": int(
                np.round(props_main[f"dm_cosmic_mc_flimflam_{rel}_err_plus"].value)),
            f"FRBDMcosmicFLIMFLAM{rel_}ErrMinus": int(
                np.round(props_main[f"dm_cosmic_mc_flimflam_{rel}_err_minus"].value)),
            f"FRBDMhost{rel_}": int(np.round(props_main[f"dm_host_mc_{rel}"].value)),
            f"FRBDMhost{rel_}ErrPlus": int(np.round(props_main[f"dm_host_mc_{rel}_err_plus"].value)),
            f"FRBDMhost{rel_}ErrMinus": int(np.round(props_main[f"dm_host_mc_{rel}_err_minus"].value)),
            f"FRBDMmodelled{rel_}": int(np.round(props_main[f"dm_modelled_mc_{rel}"].value)),
            f"FRBDMmodelled{rel_}ErrPlus": int(np.round(props_main[f"dm_modelled_mc_{rel}_err_plus"].value)),
            f"FRBDMmodelled{rel_}ErrMinus": int(np.round(props_main[f"dm_modelled_mc_{rel}_err_minus"].value)),
            f"FRBDMmodelledFLIMFLAM{rel_}": int(np.round(props_main[f"dm_modelled_mc_flimflam_{rel}"].value)),
            f"FRBDMmodelledFLIMFLAM{rel_}ErrPlus": int(
                np.round(props_main[f"dm_modelled_mc_flimflam_{rel}_err_plus"].value)),
            f"FRBDMmodelledFLIMFLAM{rel_}ErrMinus": int(
                np.round(props_main[f"dm_modelled_mc_flimflam_{rel}_err_minus"].value)),
        })

        # print(f"DM_halos (MC): {dm_halos_mean} +/- {dm_halos_std}")
        # print(
        #     f"Sum of DM_halo: {np.sum(mc_collated['dm_halo'])} +/- {np.sqrt(np.sum(mc_collated['dm_halo_err'] ** 2))}")
        # print(f"Modelled DM: {props_main[f'dm_modelled_mc_{rel}']} +/- {props_main[f'dm_modelled_mc_{rel}_err']}")
        # print(
        #     f"Modelled DM (FLIMFLAM): {props_main[f'dm_modelled_mc_flimflam_{rel}']} +/- {props_main[f'dm_modelled_mc_flimflam_{rel}_err']}")

        tbl_dict[rel] = mc_collated

    tbl_dict["M13"].remove_columns(["offset_angle_mc", "offset_angle_mc_err"])
    mc = table.hstack((tbl_dict["K18"], tbl_dict["M13"]), table_names=["K18", "M13"])

    for col in mc.colnames:
        if "_err_" in col:
            rel = col[-3:]
            col_new = col.replace(f"_err_{rel}", f"_{rel}_err")
            col_new = col_new.replace(f"_err_plus_{rel}", f"_{rel}_err_plus")
            col_new = col_new.replace(f"_err_minus_{rel}", f"_{rel}_err_minus")
            mc[col_new] = mc[col]
            print(f"Replacing {col} with {col_new}")
            mc.remove_columns(col)

    # mc = mc[sorted(mc.columns)]
    main_table = lib.read_master_table()
    for col in mc.columns:
        if col in main_table.columns:
            main_table.remove_column(col)
    main_table = table.hstack((main_table, mc))

    lib.write_master_table(main_table)
    lib.write_master_properties(props_main)
    lib.add_commands(comm_dict)

    print()


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
        "--skip_plots",
        help="Skip plots.",
        action="store_true",
    )
    parser.add_argument(
        "--do_individual",
        help="Skip individual-halo plots.",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        help="Number of instantiations to read.",
        type=int,
        default=10000
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        skip_plots=args.skip_plots,
        skip_individual=not args.do_individual,
        n_real=args.n
    )
