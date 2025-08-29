#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np
import astropy.table as table
import astropy.units as units
from astropy.coordinates import SkyCoord

import craftutils.utils as u
import craftutils.observation.field as field
import craftutils.params as p

import lib

description = """
Performs fiducial halo calculations.
"""


def main(
        output_dir: str,
        input_dir: str,
        skip_models: bool = False,
        fid_only: bool = False,
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb190714 = lib.fld

    step_size_halo = 1 * units.kpc

    main_properties = lib.read_master_properties()
    main_table_ = lib.read_master_table()

    comm_dict = {}

    if fid_only:
        rmaxes = (1.,)
    elif skip_models:
        rmaxes = ()
    else:
        rmaxes = (1., 2., 3.)

    for rel in ("M13", "K18"):
        rel_ = {"M13": "Moster", "K18": "Kravtsov"}[rel]
        for rmax in rmaxes:
            properties = frb190714.frb.foreground_accounting(
                rmax=rmax,
                step_size_halo=step_size_halo,
                cat_search="panstarrs1",
                skip_other_models=True,
                load_objects=True,
                dm_host_ism=lib.dm_host_ism_lb,
                dm_host_ism_err=lib.dm_host_ism_lb_err,
                smhm_relationship=rel,
                do_incidence=False,
                neval_cosmic=int(1e6),
            )
            halo_tbl = properties.pop("halo_table")
            dm_tbl = properties.pop("dm_cum_table")
            halo_tbl.sort("id")

            halo_dm_cum = properties.pop("halo_dm_cum")
            for galname, tbl in halo_dm_cum.items():
                lib.write_dm_gal_table(galaxy_name=galname, tbl=tbl, rmax=rmax, relationship=rel)

            properties.pop("halo_models")

            # p.save_params(os.path.join(output_path, ""), properties)
            lib.write_dm_table(tbl=dm_tbl, rmax=rmax, relationship=rel)
            lib.write_halo_table(tbl=halo_tbl, rmax=rmax, relationship=rel)
            lib.write_properties(properties, rmax=rmax, relationship=rel)

        props_rel = lib.read_properties(rmax=1., fhot=0.75, relationship=rel)

        main_properties.update(
            {
                f"dm_halos_emp_{rel}_fiducial": props_rel["dm_halos_emp"],
                f"dm_cosmic_emp_{rel}_fiducial": props_rel["dm_cosmic_emp"],
                f"dm_cosmic_emp_flimflam_{rel}_fiducial": props_rel["dm_halos_emp"] + lib.dm_igm_flimflam,
                f"dm_cosmic_avg_{rel}": props_rel["dm_cosmic_avg"],
            }
        )

        # comm_dict.update(
        #     {
        #         f"FRBDMhalosmod{rel_}": int(np.round(props_rel["dm_halos_emp"].value)),
        #         f"FRBDMcosmicmod{rel_}": int(np.round(props_rel["dm_cosmic_emp"].value)),
        #         f"FRBDMcosmicmodFLIMFLAM{rel_}": int(np.round(main_properties[f"dm_cosmic_emp_flimflam_{rel}"].value)),
        #         f"FRBDMcosmicavg{rel_}": int(np.round(props_rel["dm_cosmic_avg"].value)),
        #     }
        # )

    m13 = lib.read_halo_table(relationship="M13", rmax=1.)
    m13.sort("id")
    k18 = lib.read_halo_table(relationship="K18", rmax=1.)
    k18.sort("id")
    k18["log_mass_stellar_err"] = k18["log_mass_stellar_err_minus"]
    k18.remove_column("log_mass_stellar_err_minus")
    k18["mass_stellar_err"] = k18["mass_stellar_err_minus"]
    k18.remove_column("mass_stellar_err_minus")

    m13 = m13[
        "dm_halo", "log_mass_halo", "log_mass_halo_err", "mass_halo", "r_200"
    ]
    # m13.remove_columns([
    #     'id', 'id_short', 'z',
    #     'ra', 'dec', 'offset_angle',
    #     'id_cat', 'ra_cat', "dec_cat", 'offset_cat',
    #     'distance_angular_size', 'distance_luminosity', 'distance_comoving',
    #     'r_perp',
    #     'log_mass_stellar_err_plus', 'log_mass_stellar_err_minus', 'log_mass_stellar',
    #     'mass_stellar_err_plus', 'mass_stellar_err_minus', 'mass_stellar',
    #     'c200', 'h',
    # ])

    main_table = table.hstack((k18, m13), table_names=["K18", "M13"])
    sample = ["MUSE"] * 6 + ["2DF"] * 12 + ["PS1-STRM"] + ["MUSE"]
    print(len(main_table), len(sample))
    main_table["sample"] = sample

    letter = []

    for row in main_table:
        gal_id = row["id_short"]
        if gal_id == "HG":
            letter.append("HG")
        else:
            letter.append(gal_id[2])
        comm_dict.update({
            f"{gal_id}z": row["z"],
            f"{gal_id}name": fr'{gal_id}\,20190714A',
            f"{gal_id}logmassstar": row["log_mass_stellar"].round(1),
            # f"{gal_id}logmasshalo": row["log_mass_halo_K18"].round(1),
            # f"{gal_id}rvir": int(row["r_200_K18"].round().value),
            # f"{gal_id}DMhalo": int(row["dm_halo_K18"].round().value),
        })

    main_table["letter"] = letter
    main_table["coord"] = SkyCoord(main_table["ra"], main_table["dec"], main_table["d_A"])

    for colname in main_table.colnames:
        main_table_[colname] = main_table[colname]
    lib.write_master_table(main_table_)

    props_fiducial = lib.read_properties(rmax=1.0, relationship="K18")

    props_fiducial.pop("halo_dm_profiles")
    props_fiducial.pop("do_mc")

    dm_igm = main_properties["dm_igm_avg"] = props_fiducial.pop("dm_igm")
    dm_igm_err = main_properties["dm_igm_avg_err"] = 0.2 * dm_igm
    # main_properties["dm_cosmic_avg"] = props_fiducial.pop("dm_cosmic")
    main_properties["dm_residual_avg_fiducial"] = props_fiducial.pop("dm_excess_avg")
    main_properties["dm_residual_emp_fiducial"] = props_fiducial.pop("dm_excess_emp")
    main_properties.update(props_fiducial)

    dm_exgal = main_properties["dm_exgal"]
    dm_exgal_err = main_properties["dm_exgal_err"]

    dm_budget = dm_exgal - dm_igm - lib.dm_host_ism_lb
    dm_budget_err = np.sqrt(dm_exgal_err ** 2 + dm_igm_err ** 2 + lib.dm_host_ism_lb_err ** 2)

    dm_budget_flimflam = dm_exgal - lib.dm_igm_flimflam - lib.dm_host_ism_lb
    dm_budget_flimflam_err_plus = np.sqrt(
        dm_exgal_err ** 2 + lib.dm_igm_flimflam_err_plus ** 2 + lib.dm_host_ism_lb_err ** 2
    )
    dm_budget_flimflam_err_minus = np.sqrt(
        dm_exgal_err ** 2 + lib.dm_igm_flimflam_err_minus ** 2 + lib.dm_host_ism_lb_err ** 2
    )

    main_properties["dm_host_fiducial"] = main_properties.pop("dm_host")
    main_properties["dm_halo_host_fiducial"] = main_properties.pop("dm_halo_host")
    main_properties["dm_excess"] = lib.fld.frb.dm - main_properties['dm_mw'] - main_properties['dm_cosmic_avg']
    main_properties["dm_excess_rest"] = main_properties["dm_excess"] * (1 + lib.fld.frb.host_galaxy.z)
    main_properties["dm_budget"] = dm_budget
    main_properties["dm_budget_err"] = dm_budget_err
    main_properties["dm_budget_flimflam"] = dm_budget_flimflam
    main_properties["dm_budget_flimflam_err_plus"] = dm_budget_flimflam_err_plus
    main_properties["dm_budget_flimflam_err_minus"] = dm_budget_flimflam_err_minus
    main_properties["dm_halos_spec_fiducial"] = np.sum(
        main_table[["pz" not in row["id"] and "HG" not in row["id"] for row in main_table]]["dm_halo_K18"]
    )
    main_properties["dm_halos_pz_fiducial"] = np.sum(
        main_table[["pz" in row["id"] and "HG" not in row["id"] for row in main_table]]["dm_halo_K18"]
    )
    main_properties["dm_modelled_fiducial"] = main_properties[
                                                  "dm_mw"] + main_properties[
                                                  "dm_cosmic_emp_K18_fiducial"] + main_properties[
                                                  "dm_host_fiducial"]
    main_properties["dm_modelled_flimflam_fiducial"] = main_properties["dm_mw"] + main_properties[
        "dm_cosmic_emp_flimflam_K18_fiducial"] + \
                                                       main_properties["dm_host_fiducial"]
    main_properties["dm_unmodelled_fiducial"] = lib.fld.frb.dm - main_properties["dm_modelled_fiducial"]
    main_properties["dm_unmodelled_flimflam_fiducial"] = lib.fld.frb.dm - main_properties["dm_modelled_flimflam_fiducial"]

    lib.write_master_properties(main_properties)

    comm_dict.update({
        "FRBDMmw": int(np.round(main_properties["dm_mw"].value)),
        "FRBDMmwerr": int(np.round(main_properties["dm_mw_err"].value)),
        "FRBDMmwismNE": int(np.round(main_properties["dm_ism_mw_ne2001"].value)),
        "FRBDMmwismYMW": int(np.round(main_properties["dm_ism_mw_ymw16"].value)),
        "FRBDMmwhalo": int(np.round(main_properties["dm_halo_mw_pz19"].value)),
        "FRBDMexgal": int(np.round(dm_exgal.value)),
        "FRBDMexgalErr": int(np.round(dm_exgal_err.value)),
        # "FRBDMcosmicmod": int(np.round(main_properties["dm_cosmic_emp"].value)),
        "FRBDMcosmicavg": int(np.round(main_properties["dm_cosmic_avg"].value)),
        "FRBDMigm": int(np.round(dm_igm.value)),
        "FRBDMigmerr": int(np.round(dm_igm_err.value)),
        "FRBDMigmFLIMFLAM": int(np.round(lib.dm_igm_flimflam.value)),
        "FRBDMigmFLIMFLAMerrPlus": int(np.round(lib.dm_igm_flimflam_err_plus.value)),
        "FRBDMigmFLIMFLAMerrMinus": int(np.round(lib.dm_igm_flimflam_err_minus.value)),
        "FRBDMhalosFLIMFLAM": int(np.round(lib.dm_halos_flimflam.value)),
        "FRBDMhalosFLIMFLAMerrPlus": int(np.round(lib.dm_halos_flimflam_err_plus.value)),
        "FRBDMhalosFLIMFLAMerrMinus": int(np.round(lib.dm_halos_flimflam_err_minus.value)),
        "FRBDMhalosavg": int(np.round(main_properties["dm_halos_avg"].value)),
        # "FRBDMhalosmod": int(np.round(main_properties["dm_halos_emp"].value)),
        # "FRBDMhalosspec": int(np.round(main_properties["dm_halos_spec"].value)),
        # "FRBDMhalospz": int(np.round(main_properties["dm_halos_pz"].value)),
        "FRBDMhalosFiducial": int(np.round(main_properties["dm_halos_emp_K18_fiducial"].value)),
        "FRBDMbudget": int(np.round(dm_budget.value)),
        "FRBDMbudgetErr": int(np.round(dm_budget_err.value)),
        "FRBDMbudgetFLIMFLAM": int(np.round(dm_budget_flimflam.value)),
        "FRBDMbudgetFLIMFLAMErrPlus": int(np.round(dm_budget_flimflam_err_plus.value)),
        "FRBDMbudgetFLIMFLAMErrMinus": int(np.round(dm_budget_flimflam_err_plus.value)),
        "FRBDMhostFiducial": int(np.round(main_properties["dm_host_fiducial"].value)),
        "FRBDMhosthaloFiducial": int(np.round(main_properties["dm_halo_host_fiducial"].value)),
        "FRBDMhostism": lib.dm_host_ism_lb.value,
        "FRBDMhostismerr": lib.dm_host_ism_lb_err.value,
        "FRBDMexcess": int(np.round(main_properties["dm_excess"].value)),
        "FRBDMexcessrest": int(np.round(main_properties["dm_excess_rest"].value)),
        "FRBDMmodelledFiducial": int(np.round(main_properties["dm_modelled_fiducial"].value)),
        "FRBDMmodelledFLIMFLAMFiducial": int(np.round(main_properties["dm_modelled_flimflam_fiducial"].value)),
        "FRBDMunmodelledFiducial": int(np.round(main_properties["dm_unmodelled_fiducial"].value)),
        "FRBDMunmodelledFLIMFLAMFiducial": int(np.round(main_properties["dm_unmodelled_flimflam_fiducial"].value)),
        "NFg": len(main_table) - 1,
        "NMUSE": len(main_table["sample"] == "MUSE") - 1,
        "N2DF": len(main_table["sample"] == "2DF"),
        "Nphot": len(main_table["sample"] == "PS1STRM"),
    })

    lib.add_commands(
        comm_dict
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
        "--do_all",
        help="Do all Rmax values (1, 2 and 3); otherwise just 1.",
        action="store_false",
    )

    parser.add_argument(
        "--skip_models",
        help="Skip all fiducial models.",
        action="store_true",
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        fid_only=args.do_all,
        skip_models=args.skip_models
    )
