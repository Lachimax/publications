#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np

from astropy import units
from astropy.cosmology import Planck18

from frb import turb_scattering

import craftutils.utils as u

import lib
from lib import dm_units

description = """
Performs the scattering analysis.
"""


def main(
        output_dir: str,
        input_dir: str,
        rmax: float = 1.
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    frb = lib.fld.frb

    halo_tbl = lib.read_master_table()
    properties = lib.read_master_properties()

    # Also: copy 190608

    # Assume a MW-like host (similar mass)
    # from Cordes 22
    z = frb.z
    # Put nu in host frame
    nu = frb.nu_scattering  # * (1 + z)
    dm_ism_host = lib.dm_host_ism_lb
    x_tau = 4

    tau_ism_host = 1.9 * 10e-7 * units.ms * (nu / units.GHz) ** - x_tau * (dm_ism_host / dm_units) ** 1.5 * (
            1 + 3.55e-5 * (dm_ism_host / dm_units) ** 3) / ((1 + z) ** (x_tau - 1))

    f = 1 * units.pc ** -(2 / 3) * units.km ** (-1 / 3)

    tau = frb.tau
    tau_err = frb.tau_err
    print("tau_obs ==", tau, "+/-", tau_err)
    tau_mw_ism, tau_mw_ism_err = frb.tau_mw()
    print("tau_mw_ism ==", tau_mw_ism, "+/-", tau_mw_ism_err,
          f"(from DM_mw_ism == {properties['dm_ism_mw_ne2001']}")
    tau_mw_halo = frb.tau_mw_halo(dm_mw_halo=properties["dm_halo_mw_pz19"])
    print("tau_mw_halo ==", tau_mw_halo, f"(from DM_mw_halo == {properties['dm_halo_mw_pz19']})")
    tau_ism_host = tau_ism_host.decompose().to("ms")
    print(f"tau_ism_host ==", tau_ism_host, f"(from DM_ism_host == {dm_ism_host}")
    tau_budget = tau - tau_mw_halo - tau_mw_ism - tau_ism_host

    for rel in ("M13", "K18"):

        taus = []
        fs_dm2 = []
        gs = []
        gs_err_plus = []
        gs_err_minus = []

        for row in halo_tbl:
            name = row['id']
            print("\n", name)
            obj = lib.fld.objects[name]
            obj.load_output_file()
            mnfw = obj.halo_model_mnfw()
            dm_halo = row[f"dm_halo_mc_{rel}"]
            print(f"\t{dm_halo=}")
            tau_this = frb.tau_from_halo(
                halo=mnfw,
                r_perp=row[f"r_perp_mc_{rel}"],
                dm_halo=row[f"dm_halo_mc_{rel}"],
                f=f
            )
            d_sl = Planck18.angular_diameter_distance_z1z2(row["z"], z)
            d_lo = row["d_A"]
            d_so = frb.host_galaxy.angular_size_distance()
            l = row[f"path_length_mc_{rel}"]
            g = u.g_scatt(
                d_sl=d_sl,
                d_lo=d_lo,
                d_so=d_so,
                l=l
            )
            g_err_plus = row[f"path_length_mean_{rel}_err_plus"] * g / l
            g_err_minus = row[f"path_length_mean_{rel}_err_minus"] * g / l
            gs.append(g)
            gs_err_plus.append(g_err_plus)
            gs_err_minus.append(g_err_minus)
            if np.isnan(tau_this):
                tau_this = 0 * units.ms
            print("\ttau_halo == ", tau_this)
            taus.append(tau_this)
            f_dm2 = f * dm_halo ** 2
            print("\tF x DM^2 == ", f_dm2)

        halo_tbl[f"tau_halo_F1_{rel}"] = taus
        halo_tbl[f"g_scatt_{rel}"] = gs
        halo_tbl[f"g_scatt_{rel}_err_plus"] = gs_err_plus
        halo_tbl[f"g_scatt_{rel}_err_minus"] = gs_err_minus

        tau_sum = np.sum(halo_tbl[f"tau_halo_F1_{rel}"]) + tau_ism_host + tau_mw_ism + tau_mw_halo
        print("Total modelled:", tau_sum)
        f_lim = frb.tau / (tau_sum / f)

        print("tau_budget ==", tau_budget)
        print("\nTotal sum modelled:", tau_sum)
        print("Using full tau: F < (", f_lim.value, "/ A_tau )", f_lim.unit)
        f_lim_budget = tau_budget / (tau_sum / f)
        print("Using budget tau: F < (", f_lim_budget.value, "/ A_tau )", f_lim_budget.unit)
        print("=" * 50)

        print(f"Second round, using F == {f_lim_budget}")

        taus = []

        ne_avgs = []
        ne_avgs_err = []

        for row in halo_tbl:
            name = row['id']
            print("\n", name)
            obj = lib.fld.objects[name]
            obj.load_output_file()
            mnfw = obj.halo_model_mnfw()
            dm_halo = row[f"dm_halo_mc_{rel}"]
            print(f"\t{dm_halo=}")
            tau_this = frb.tau_from_halo(
                halo=mnfw,
                r_perp=row[f"r_perp_mc_{rel}"],
                dm_halo=row[f"dm_halo_mc_{rel}"],
                f=f_lim_budget
            )
            if np.isnan(tau_this):
                tau_this = 0 * units.ms
            if name.startswith("HG"):
                tau_this /= 2
            print("\ttau_halo == ", tau_this)
            taus.append(tau_this)
            f_dm2 = f * dm_halo ** 2
            fs_dm2.append(f_dm2)
            print("\tF x DM^2 == ", f_dm2)

            path_length = row[f"path_length_mc_{rel}"]

            n_e_avg = (row[f"dm_halo_mc_{rel}"] / path_length).to("cm-3")
            n_e_avg_err = u.uncertainty_product(
                n_e_avg,
                (row[f"dm_halo_mc_{rel}"], row[f"dm_halo_mc_{rel}_err"]),
                (row[f"path_length_mc_{rel}"], row[f"path_length_mc_{rel}_err"])
            )

            print(f"\t<n_e> = {n_e_avg} +/- {n_e_avg_err}")

            ne_avgs.append(n_e_avg)
            ne_avgs_err.append(n_e_avg_err)
            print("\tX:")

            ne_lim_per_alpha = (2e-3 * (path_length / 50 * units.kpc) ** (-1 / 2) * (1 * units.kpc / 1 * units.kpc) ** (
                    1 / 3) * (tau_budget / 40 * units.us) ** (5 / 12)).decompose() * units.cm ** -3

            alpha_lim = ne_lim_per_alpha / n_e_avg

            print(f"\t\t<n_e> < {ne_lim_per_alpha} / alpha")
            print(f"\t\talpha < {ne_lim_per_alpha} / <n_e>")
            print(f"\t\talpha < {alpha_lim} (with <n_e> = DM / L)")

            # ne_avg_lim = 2e-3 * alpha

        print(len(taus), len(halo_tbl))
        halo_tbl[f"tau_halo_{rel}"] = taus
        halo_tbl[f"tau_halo_{rel}_err"] = [-999 * units.ms] * len(halo_tbl)
        halo_tbl[f"tau_halo_{rel}_err"][halo_tbl[f"tau_halo_{rel}"] == 0] = 0 * units.ms
        halo_tbl[f"ne_avg_{rel}"] = ne_avgs
        halo_tbl["F*DM^2"] = fs_dm2

        # From notebook, in imitation of Prochaska 2019

        fga = halo_tbl[1]
        fga_obj = lib.fld.objects["FGa_20190714A"]
        fga_halo = fga_obj.halo_model_mnfw()
        dm_ism_host = fga[f"dm_halo_{rel}"]
        r_perp = fga[f"r_perp_mc_{rel}"]
        r_perp_err = fga[f"r_perp_mc_{rel}_err"]

        ne_min = fga_halo.ne(np.array((0, 0, (r_perp + r_perp_err).value))) * units.cm ** -3
        ne = fga_halo.ne(np.array((0, 0, r_perp.value))) * units.cm ** -3
        ne_max = fga_halo.ne(np.array((0, 0, (r_perp - r_perp_err).value))) * units.cm ** -3
        print(f"{ne_min=}, {ne=}, {ne_max=}")

        # Following Prochaska 2019
        l0 = 1. * units.AU
        L0 = 0.001 * units.pc
        DL = 2 * np.sqrt((rmax * fga_halo.r200) ** 2 - r_perp ** 2)

        z_L = fga["z"]
        z_S = frb.host_galaxy.z

        nu_obs = frb.nu_scattering
        lambda_obs = nu_obs.to("m", units.spectral())
        print(f"{lambda_obs=}")

        print(f"{DL=}")

        l0 = 1e7 * units.cm
        # Assuming Kolomogorv
        turbB = turb_scattering.Turbulence(ne, l0, L0, z_L, DL=DL, lobs=lambda_obs)
        print(f"{turbB=}")
        turbB = turb_scattering.Turbulence(ne, l0, L0, z_L, DL=50 * units.kpc, lobs=lambda_obs)
        print(f"{turbB=}")
        tbroad = turbB.temporal_smearing(lambda_obs, z_S)
        print(f"{tbroad=}")
        print(f"{turbB.rdiff=}")

        # alpha_lim = 2e-3 * ()

        rel_ = rel[0]

        lib.add_commands(
            {
                f"Flim{rel_}": f_lim_budget.round(5).value,
                f"TauSumInit{rel_}": int(tau_sum.round(0).value)
            },
        )

    new_commands = {
        f"FRBTauMWISM": units.Quantity(tau_mw_ism.value).to_string(precision=4),
        f"FRBTauMWHalo": units.Quantity(tau_mw_halo.value).to_string(precision=4),
        f"FRBTauHostISM": units.Quantity(tau_ism_host.value).to_string(precision=4),
        # f"Tau": units.Quantity(tau.value).to_string(precision=2)
    }
    print(new_commands)
    lib.add_commands(
        new_commands
    )

    lib.write_master_table(tbl=halo_tbl)
    lib.write_master_properties(properties)


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
        "--rmax",
        help="R_max to use in this script, as a multiple of R_200.",
        type=float,
        default=1.
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        rmax=args.rmax
    )
