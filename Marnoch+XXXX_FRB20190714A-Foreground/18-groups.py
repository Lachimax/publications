#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np
import matplotlib.pyplot as plt

from astropy import units, constants, table
from astropy.cosmology import Planck18
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting

import frb.halos.models as halos

import craftutils.utils as u
import craftutils.params as p
import craftutils.plotting as pl
from craftutils.observation.objects.galaxy import Galaxy

import lib

description = """
Analysis of the foreground galaxy group candidates.
"""

frb = lib.fld.frb

fitter_type = fitting.TRFLSQFitter


def little_h(z):
    return Planck18.H(z=z) / (100 * units.km * units.second ** -1 * units.Mpc ** -1)


def concentration_parameter(mass, z):
    h = little_h(z)
    return 4.67 * (mass / (10 ** 14 * h ** -1 * units.solMass)) ** (-0.11)


def nfw_mass_contained(halo, r):
    return ((2 * np.pi * halo.r200 ** 3 * halo.rho0 / (halo.c ** 3)).to(units.solMass)).value * np.log(
        ((halo.c ** 2 * r ** 2) / (halo.r200 ** 2)).value + 1)


def halo_setup(z, mass=None, log_mass=None, f_hot=0.8):
    if log_mass is None:
        log_mass = np.log10(mass / units.solMass)
    elif mass is None:
        mass = (10 ** log_mass) * units.solMass

    c_halo = concentration_parameter(mass, z)

    return halos.ModifiedNFW(
        log_Mhalo=log_mass,
        c=c_halo.value,
        cosmo=Planck18,
        z=z,
        f_hot=f_hot,
    )


def ang_to_phys(theta, z):
    D_A = Planck18.angular_diameter_distance(z)
    return (D_A * theta.to(units.rad).value).to(units.kpc)


def phys_to_ang(x, z):
    D_A = Planck18.angular_diameter_distance(z)
    return (x / D_A) * units.rad


def phys_to_sky(x, y, z, ref_pos):
    x_ang = phys_to_ang(x, z)
    y_ang = phys_to_ang(y, z)

    dec = (ref_pos.dec + y_ang).to(units.deg)
    ra = (ref_pos.ra - (x_ang / np.cos(dec))).to(units.deg)

    return SkyCoord(ra, dec)


def velocity_contained_mass(M, r):
    return np.sqrt(constants.G * M / r).to(units.km / units.s)


def mass_contained(r, v):
    return (r * v ** 2 / constants.G).to(units.solMass)


def group_barycentre(group_members, name: str, mass_key: str = "mass_halo_mc_K18"):
    group_z = group_members["z"].mean()
    ref_pos = frb.position
    D_A = Planck18.angular_diameter_distance(group_z)
    group_gal_mass = group_members[mass_key].sum()
    group_members["x_ang"] = ((ref_pos.ra - group_members["ra"]) * np.cos(group_members["dec"])).to(units.arcsec)
    group_members["y_ang"] = (group_members["dec"] - ref_pos.dec).to(units.arcsec)

    group_members["x_projected"] = ang_to_phys(group_members["x_ang"], group_z)
    group_members["y_projected"] = ang_to_phys(group_members["y_ang"], group_z)

    group_centre_x = np.sum(group_members[mass_key] * group_members["x_projected"]) / group_gal_mass
    group_centre_y = np.sum(group_members[mass_key] * group_members["y_projected"]) / group_gal_mass

    group_centre = phys_to_sky(
        group_centre_x,
        group_centre_y,
        group_z,
        ref_pos
    )

    # group_centre_ra = group_centre.ra
    # group_centre_dec = group_centre.dec

    #     plt.scatter(group_members["ra"], group_members["dec"])
    #     plt.scatter(group_centre_ra, group_centre_dec)
    #     for row in group_members:
    #         plt.text(row["ra"] - 2 * units.arcsec, row["dec"], row["id_short"])
    #     plt.gca().invert_xaxis()
    #     plt.axis("equal")
    #     plt.show()

    plot_dir = f"groups/{name}"

    fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))

    ax.scatter(group_members["x_projected"], group_members["y_projected"])
    for row in group_members:
        ax.text(row["x_projected"], row["y_projected"], row["id_short"])
    ax.axis("equal")
    ax.scatter(group_centre_x, group_centre_y)
    lib.savefig(fig=fig, filename=f"{name}_projected", subdir=plot_dir)

    # group_centre = SkyCoord(group_centre_ra, group_centre_dec)
    group_members = group_quantities(
        group_members=group_members,
        group_centre_x=group_centre_x,
        group_centre_y=group_centre_y,
        group_z=group_z
    )

    fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
    ax.scatter(group_members["r_group"], group_members["v_pec"], marker="x")
    lib.savefig(fig=fig, filename=f"{name}_v_dist", subdir=plot_dir)

    fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
    ax.scatter(group_members["r_group"], group_members["mass_contained"], marker="x")
    lib.savefig(fig=fig, filename=f"{name}_r_mass", subdir=plot_dir)

    mass_group = group_members["mass_contained"].max()
    log_mass = np.log10(mass_group / units.solMass)
    group_halo = halo_setup(group_z, log_mass=log_mass)
    group_r_perp_ang = group_centre.separation(frb.position)
    group_r_perp_proj = (D_A * group_r_perp_ang.to(units.rad).value).to(units.kpc)
    dm_igrm = group_halo.Ne_Rperp(group_r_perp_proj) / (1 + group_z)

    return {
        "centre_coord": group_centre,
        "centre_x": group_centre_x,
        "centre_y": group_centre_y,
        "z": group_z,
        "member_table": group_members,
        "halo_model": group_halo,
        "halo_r200": group_halo.r200,
        "mass_fastest": mass_group,
        "log_mass_fastest": log_mass,
        "dm_igrm_outermost": dm_igrm,
        "angular_size_distance": D_A,
        "r_perp_frb": group_r_perp_proj,
        "offset_angle": group_r_perp_ang,
        "mass_sum_members": group_gal_mass,
        "dm_sum_members": group_members["dm_halo_mc_K18"].sum(),
        "log_mass_sum_members": np.log10(group_gal_mass / units.solMass),
    }


def group_quantities(
        group_members,
        group_centre_x,
        group_centre_y,
        group_z,
):
    group_members["r_group"] = np.sqrt(
        (group_members["x_projected"] - group_centre_x) ** 2 + (group_members["y_projected"] - group_centre_y) ** 2)
    group_members["v_pec"] = lib.peculiar_velocity(
        z_obs=group_members["z"],
        z_cos=group_z
    ).to(units.km / units.s)
    group_members["mass_contained"] = mass_contained(
        r=group_members["r_group"],
        v=group_members["v_pec"])
    mass_group = group_members["mass_contained"].max()
    group_members["log_mass_contained"] = np.log10(group_members["mass_contained"] / units.solMass)
    group_members["v_escape"] = np.sqrt(2 * constants.G * mass_group / group_members["r_group"]).to(units.km / units.s)
    group_members["frac_v_escape"] = group_members["v_pec"] / group_members["v_escape"]
    return group_members


# Here lies the result of my own integration.
#
# Models assuming fixed position.
#
# nfw() does analytically what nfw_2() does numerically.


group_z = 0.212
group_centre_x = 0.5
group_centre_y = 0.5


@models.custom_model
def nfw(r, mass=1e13):
    '''
    f(x)=M_contained
    '''
    halo = halo_setup(group_z, mass * units.solMass)
    M_contained = nfw_mass_contained(halo, r)
    return M_contained


@models.custom_model
def nfw_2(r, mass=1e14):
    '''
    f(x)=M_contained
    '''
    c_halo = concentration_parameter(mass * units.solMass, group_z)
    halo = halos.ModifiedNFW(
        log_Mhalo=np.log10(mass),
        c=c_halo.value,
        cosmo=Planck18,
        z=group_z,
        f_hot=0.8,
    )

    _r = np.arange(0, halo.r200.to(units.kpc).value, 0.1)
    r_frac = _r / halo.r200.value
    y = c_halo * r_frac
    rho = (halo.rho0.to(units.solMass / units.kpc ** 3)).value / (y * (1 + y ** 2))
    V = (4 * np.pi * _r ** 3 / 3)
    d_V = V - np.roll(V, 1)
    d_V[0] = 0
    M_layer = rho * d_V

    M_contained = np.zeros_like(r)
    for i, r_this in enumerate(r):
        M_contained[i] = (np.nancumsum(M_layer))[np.abs(_r - r_this).argmin() - 1]

    return M_contained


@models.custom_model
def nfw_3(
        x,
        y,
        x_centre=group_centre_x,
        y_centre=group_centre_y,
        mass=1e13
):
    '''
    f(x,y)=v_pec
    '''

    #     x_centre = group_centre_mod_x.value
    #     y_centre = group_centre_mod_y.value

    x_centre = u.dequantify(x_centre)
    y_centre = u.dequantify(y_centre)

    halo = halo_setup(group_z, mass * units.solMass)

    r = np.sqrt((x - x_centre) ** 2 + (y - y_centre) ** 2)
    M_contained = nfw_mass_contained(halo, r)
    # print(x_centre, y_centre, mass)
    v_pec = velocity_contained_mass(
        M=M_contained * units.solMass,
        r=r * units.kpc
    )
    return v_pec.value


def main(
        output_dir: str,
        input_dir: str,
        skip_grid: bool,
        quick: bool
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    m31 = halos.M31()
    dm_m31 = m31.DM_from_Galactic(scoord=lib.fld.frb.position)
    print("=" * 20)
    print("Andromeda:", dm_m31)
    print("\tSeparation:", m31.coord.separation(lib.fld.frb.position))
    print("=" * 20)

    mass_bounds = (1e10, 1e20)

    halo_tbl = lib.read_master_table()

    g1 = lib.get_by_id(halo_tbl, ["FGb", "FGc", "FGd", "FGf"])
    # g1a = lib.get_by_id(halo_tbl, ["FGb", "FGc", "FGd", "FGf", "FGi", "FGj"])
    g2 = lib.get_by_id(halo_tbl, ["FGl", "FGo", "FGh", "FGg"])
    print()

    this_script = u.latex_sanitise(os.path.basename(__file__))

    g_ = {
        "CG1": g1,
        "CG2": g2,
        # "CG1A": g1a
    }

    cg_table = {
        "id": [],
        "z": [],
        "centre": [],
        "r_perp": [],
        "sigma_v": [],
        "mass_virial": [],
        "log_mass_virial": [],
        "r_200": [],
        "dm_igrm_0.6": [],
        "dm_igrm_0.8": [],
    }

    group_dir = os.path.join(lib.output_path, "groups")
    os.makedirs(group_dir, exist_ok=True)

    for name, group_members in g_.items():
        print("=" * 50)
        print(name)
        print(group_members["id_short"])
        group_properties = group_barycentre(group_members, name)
        group_centre = group_properties["centre_coord"]
        group_members = group_properties.pop("member_table")
        group_halo = group_properties.pop("halo_model")

        # u.latexise_table(
        #     tbl=group_members[
        #         "id_short",
        #         "r_group",
        #         "v_pec"
        #     ],
        #     column_dict={
        #         "id_short": "ID",
        #         "r_group": "R_group",
        #     },
        #     sub_colnames={
        #         "r_group": "kpc",
        #         "v_pec": "km~s$^{-1}$",
        #         "ma"
        #     }
        # )

        global group_z
        global group_centre_x
        global group_centre_y

        group_centre_x = group_properties["centre_x"]
        group_centre_y = group_properties["centre_y"]
        group_z = group_properties["z"]
        # group_centre.to_string("hmsdms")

        plot_dir = f"groups/{name}"

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
        vm = max(np.abs(group_members["v_pec"].min().value), np.abs(group_members["v_pec"].max().value))
        c = ax.scatter(
            group_members["x_projected"],
            group_members["y_projected"],
            c=group_members["v_pec"],
            cmap="bwr",
            vmax=vm,
            vmin=-vm,
            facecolor="black"
        )
        cbar = fig.colorbar(c)
        for row in group_members:
            ax.text(row["x_projected"], row["y_projected"], row["id_short"])
        ax.scatter(group_centre_x, group_centre_y, marker="x", c="white")
        ax.axis("equal")
        ax.tick_params(axis="both", labelsize=pl.tick_fontsize)
        ax.set_xlabel("Projected $x$-position (kpc)", fontsize=14)
        ax.set_ylabel("Projected $y$-position (kpc)", fontsize=14)
        cbar.ax.set_ylabel("Peculiar velocity (km/s)", labelpad=15, rotation=270, fontsize=14)

        lib.savefig(
            fig=fig,
            filename=f"{name}_spatial_v",
            subdir=plot_dir
        )

        plt.style.use('default')
        pl.latex_setup()

        # ================================================
        # Virially
        # Help from https://publish.obsidian.md/astrowiki/G.+Galaxies/Velocity+Dispersion

        r_avg = np.mean(group_members["r_group"])
        sigma_v = np.std(group_members["v_pec"])
        mass_virial = (3 * r_avg * sigma_v ** 2 / constants.G).to("solMass")
        group_properties["mass_virial"] = mass_virial
        group_properties["log_mass_virial"] = np.log10(mass_virial.value)
        print("="*20)
        print(f"{name} VIRIAL MASS: {mass_virial} (10^{np.round(group_properties['log_mass_virial'], 2)})")

        for f_hot in 0.6, 0.8:
            halo_virial = halo_setup(group_z, mass_virial, f_hot=f_hot)

            group_properties["dm_igrm_virial_" + str(f_hot)] = halo_virial.Ne_Rperp(group_properties["r_perp_frb"]) / (1 + group_z)
            cg_table["dm_igrm_" + str(f_hot)].append(group_properties["dm_igrm_virial_" + str(f_hot)])
            print(f"DM, {f_hot=}:", group_properties["dm_igrm_virial_" + str(f_hot)])
            print("=" * 20)


        group_properties["r_avg"] = r_avg
        group_properties["sigma_v"] = sigma_v

        cg_table["id"].append(name[-1])
        cg_table["centre"].append(group_centre)
        cg_table["r_perp"].append(group_properties["r_perp_frb"])
        cg_table["sigma_v"].append(group_properties["sigma_v"])
        cg_table["mass_virial"].append(mass_virial)
        cg_table["log_mass_virial"].append(group_properties["log_mass_virial"])
        cg_table["r_200"].append(halo_virial.r200)
        cg_table["z"].append(group_z)

        group_members.sort("r_group")

        u.latexise_table(
            tbl=group_members[
                "id_short",
                "r_group",
                "v_pec",
                "log_mass_contained",
                # "frac_v_escape"
            ],
            round_cols=[
                "r_group",
                "v_pec",
                "log_mass_contained",
                "frac_v_escape"
            ],
            round_digits=1,
            column_dict={
                "id_short": "ID",
                "r_group": r"$R_\mathrm{group}$",
                "v_pec": r"$v_\mathrm{pec}$",
                "log_mass_contained": r"$\log_{10}(\dfrac{{\M{contained}}}{\si{\solarmass}})$",
                "frac_v_escape": r"$\dfrac{v_\mathrm{pec}}{v_\mathrm{escape}}$",
            },
            sub_colnames={
                "r_group": r"kpc",
                "v_pec": r"\si{\m\s^{-1}}",
            },
            output_path=os.path.join(lib.latex_table_path, f"{name}.tex"),
            second_path=os.path.join(lib.latex_table_path_db, f"{name}.tex"),
            label=f"tab:{name}",
            short_caption=fr"Proposed members of {name}",
            caption=fr"Proposed members of {name}. $R_\mathrm{{group}}$ is the projected distance from the group barycentre."
                    r" \tabscript{" + this_script + "}",
        )

        group_dir_this = os.path.join(group_dir, name)
        os.makedirs(group_dir_this, exist_ok=True)

        if not quick:

            # ================================================
            # Stellarly (GordonProspector)

            group = Galaxy(z=group_z)

            group_properties["mass_stellar"] = group_members["mass_stellar"].sum()
            group_properties["log_mass_stellar"] = np.log10(group_properties["mass_stellar"].value)
            group.log_mass_stellar = group_properties["log_mass_stellar"]
            for rel in ("K18", "M13"):
                mass_shmr, log_mass_shmr = group.halo_mass(relationship=rel)
                halo_shmr = halo_setup(group_z, mass_shmr)
                group_properties[f"mass_{rel}"] = mass_shmr.value
                group_properties[f"log_mass_{rel}"] = log_mass_shmr
                group_properties[f"dm_igrm_{rel}"] = halo_shmr.Ne_Rperp(group_properties["r_perp_frb"]) / (1 + group_z)

                print("=" * 20)
                print(f"{name} {rel} MASS: {mass_shmr} (10^{np.round(log_mass_shmr, 2)})")
                print("\tDM:", group_properties[f"dm_igrm_{rel}"])
                print("\tM*:", "10^", group.log_mass_stellar)
                print("=" * 20)

             # ================================================
            # Test fitting using nfw()
            r = np.arange(0, np.max(group_halo.r200.value), 0.1) * units.kpc
            m_noise = nfw_mass_contained(group_halo, r) + np.random.normal(0, 1, r.shape) * 1e12
            m_noise = m_noise[np.isfinite(m_noise)]
            r = r[np.isfinite(m_noise)]
            fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
            ax.scatter(r.value, m_noise)
            nfw_model = nfw(bounds={"mass": mass_bounds})
            fitter = fitter_type()
            # try:
            fit = fitter(nfw_model, r.value, m_noise)
            ax.plot(r.value, fit(r.value), c="red")
            lib.savefig(
                fig=fig,
                filename=f"{name}_test_fit_1",
                subdir=plot_dir
            )
            mass_fitted = fit.mass
            print("\tTest fit mass", mass_fitted)
            print("\tInput mass", group_properties["mass_fastest"])
            # except fitting.NonFiniteValueError:
            #     pass

            # ================================================
            # Actual fit using nfw() and data (position fixed)
            nfw_model = nfw(bounds={"mass": (1e10, 1e20)})
            m = group_members["mass_contained"].value
            r_g = group_members["r_group"].value
            plt.scatter(r_g, m)

            print()

            fitter = fitter_type()
            # try:
            fit = fitter(nfw_model, r_g, m)
            r = np.linspace(0 * units.kpc, np.max(r_g * units.kpc) * 1.5, 1000)

            halo_fitted = halo_setup(group_z, fit.mass * units.solMass)

            fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
            ax.plot(r.value, fit(r.value))
            ax.scatter(r_g, m)
            # ax.plot([halo_fitted.r200.value, halo_fitted.r200.value], [0, 5e12])
            ax.set_xlim(0, np.max(r_g) * 1.1)
            ax.set_ylim(0, np.max(m) * 1.1)
            # print("\t", halo_fitted.r200)
            mass_fitted = halo_fitted.M_halo.to("solMass")
            print(f"{mass_fitted=}")

            lib.savefig(
                fig=fig,
                filename=f"{name}_fit_1",
                subdir=plot_dir
            )

            print("\tFit mass 1", mass_fitted, np.log10(mass_fitted.value))
            group_properties["mass_fitted_1"] = mass_fitted
            group_properties["log_mass_fitted_1"] = np.log10(mass_fitted.value)
            group_properties["dm_igrm_fitted_1"] = halo_fitted.Ne_Rperp(group_properties["r_perp_frb"]) / (1 + group_z)

            # =================================================
            # Fit to data using grid

            if not skip_grid:

                group_centre_mod_x = group_centre_mod_y = 0 * units.kpc

                @models.custom_model
                def nfw_g(
                        x,
                        y,
                        mass=1e12
                ):
                    '''
                    f(x,y)=v_pec
                    '''

                    x_centre = group_centre_mod_x.value
                    y_centre = group_centre_mod_y.value

                    halo = halo_setup(z=group_z, mass=mass * units.solMass)

                    r = np.sqrt((x - x_centre) ** 2 + (y - y_centre) ** 2)
                    M_contained = nfw_mass_contained(halo, r)
                    #     print(x_centre, y_centre, mass)
                    v_pec = velocity_contained_mass(M=M_contained * units.solMass, r=r * units.kpc).value
                    return v_pec

                # x_grid = np.arange(0, 4, 1) * units.kpc
                # y_grid = np.arange(0, 4, 1) * units.kpc
                x_grid = np.arange(0, 100, 1) * units.kpc
                y_grid = np.arange(-150, 1, 1) * units.kpc

                masses = []
                xs = []
                ys = []
                fits = []
                err = []

                for i, group_centre_mod_x in enumerate(x_grid):
                    print(100 * i / len(x_grid), "%")
                    for group_centre_mod_y in y_grid:
                        xs.append(group_centre_mod_x)
                        ys.append(group_centre_mod_y)

                        nfw_model = nfw_g(bounds={"mass": (1e12, 1e14)})
                        v = np.abs(group_members["v_pec"].value)
                        x = group_members["x_projected"].value
                        y = group_members["y_projected"].value

                        fitter = fitting.LevMarLSQFitter()
                        fit_g = fitter(nfw_model, x, y, v)

                        # r_g = np.sqrt((x - x_centre) ** 2 + (y - y_centre) ** 2)

                        residuals = v - fit_g(x, y)
                        rms = np.sqrt(np.mean(residuals ** 2))

                        masses.append(fit_g.mass.value)
                        err.append(rms * units.m / units.s)

                fit_table = table.QTable({
                    "mass_group_halo": np.array(masses) * units.solMass,
                    "log_mass_group_halo": np.log10(masses),
                    "x_group_halo": xs,
                    "y_group_halo": ys,
                    "rms": err,
                })

                best_fit = fit_table[fit_table["rms"].argmin()]

                print("\tFit mass 2", best_fit)
                fit_table.write(os.path.join(group_dir_this, "grid_fit.ecsv"), overwrite=True)

                xmax = 120
                x_centre_grid = best_fit["x_group_halo"].value
                y_centre_grid = best_fit["y_group_halo"].value

                group_centre_grid = phys_to_sky(
                    x=best_fit["x_group_halo"],
                    y=best_fit["y_group_halo"],
                    z=group_z,
                    ref_pos=frb.position
                )

                v = np.abs(group_members["v_pec"].value)
                x = group_members["x_projected"].value
                y = group_members["y_projected"].value

                r_g = np.sqrt((x - x_centre_grid) ** 2 + (y - y_centre_grid) ** 2)
                group_members["r_group_fitted"] = r_g * units.kpc

                x_coord = np.linspace(x_centre_grid, x_centre_grid + halo_fitted.r200.value, 1000)
                y_coord = np.ones_like(x_coord) * y_centre_grid
                r_coord = np.sqrt((x_coord - x_centre_grid) ** 2 + (y_coord - y_centre_grid) ** 2)

                group_mass = best_fit["mass_group_halo"]

                c_fitted = concentration_parameter(mass=group_mass, z=group_z)
                halo_grid_fitted = halos.ModifiedNFW(
                    log_Mhalo=np.log10(group_mass.value),
                    c=c_fitted.value,
                    cosmo=Planck18,
                    z=group_z,
                    f_hot=0.8,
                )

                group_members["mass_contained_r_fitted"] = mass_contained(
                    r=group_members["r_group_fitted"],
                    v=group_members["v_pec"]
                )
                group_members["log_mass_contained_r_fitted"] = np.log10(
                    group_members["mass_contained_r_fitted"] / units.solMass)
                group_members["mass_contained_fit_predict"] = nfw_mass_contained(halo_grid_fitted, r_g) * units.solMass
                group_members["log_mass_contained_fit_predict"] = np.log10(
                    group_members["mass_contained_fit_predict"] / units.solMass)
                m_coord = nfw_mass_contained(halo_grid_fitted, r_coord)

                fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth * 2 / 3))

                ax.scatter(r_g, v, marker='x', c='red')
                ax.plot(r_coord, velocity_contained_mass(M=m_coord * units.solMass, r=r_coord * units.kpc).value)
                ax.set_xlim(0, xmax)
                ax.set_xlabel("$R_\mathrm{group}$ (kpc)")
                ax.set_ylabel("$v_\mathrm{pec}$ (km~s$^{-1}$)")
                lib.savefig(
                    fig=fig,
                    filename=f"{name}_fit_grid_rv",
                    subdir=plot_dir
                )

                fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth * 2 / 3))
                ax.scatter(r_g, group_members["mass_contained_r_fitted"].value, marker='x', c='red')
                ax.plot(r_coord, m_coord)
                plt.xlim(0, xmax)
                ax.set_ylim(0, 1.25e13)
                ax.set_xlabel("$R_\mathrm{group}$ (kpc)")
                ax.set_ylabel("$M_\mathrm{contained}$ (km~s$^{-1}$)")
                lib.savefig(
                    fig=fig,
                    filename=f"{name}_fit_grid_rm",
                    subdir=plot_dir
                )

                # for row in group_members:
                #     plt.text(row["x_projected"], row["y_projected"], row["id_short"])
                # plt.axis("equal")

                group_members["v_escape_fitted"] = np.sqrt(
                    2 * constants.G * group_members["mass_contained_fit_predict"] / group_members["r_group_fitted"]).to(
                    units.km / units.s)
                group_members["frac_v_escape_fitted"] = group_members["v_pec"] / group_members["v_escape_fitted"]

                D_A = Planck18.angular_diameter_distance(z=group_z)
                group_r_perp_ang = group_centre_grid.separation(frb.position)
                rperp = (D_A * group_r_perp_ang.to(units.rad).value).to(units.kpc)
                dm = halo_grid_fitted.Ne_Rperp(rperp)

                print("Mass from grid fit:", best_fit["log_mass_group_halo"])
                print("\tDM_IGrM:", dm)
                for key in fit_table.colnames:
                    group_properties[f"{key}_grid"] = best_fit[key]

                group_properties[f"centre_coord_grid"] = group_centre_grid
                group_properties[f"r_perp_frb_grid"] = rperp
                group_properties[f"dm_igrm_grid"] = dm

            # # ================================================
            # # Actual fit using nfw_2() and data (position fixed)
            #
            # nfw_model = nfw_2(bounds={"mass": mass_bounds})
            # m = group_members["mass_contained"].value
            # r_g = group_members["r_group"].value
            #
            # fitter = fitter_type()
            # fit_2 = fitter(nfw_model, r_g, m)
            #
            # halo_fitted = halo_setup(group_z, fit.mass * units.solMass)
            # mass_fitted = halo_fitted.M_halo.to("solMass")
            #
            # r = np.linspace(0 * units.kpc, halo_fitted.r200, 5000)
            # vals = fit_2(r.value)
            # vals[0] = 0
            #
            # fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
            # ax.scatter(r_g, m)
            # ax.plot(r.value, vals)
            # ax.set_xlim(0, np.max(r_g) * 1.1)
            # ax.set_ylim(0, np.max(m) * 1.1)
            # # plt.plot([halo_fitted.r200.value, halo_fitted.r200.value], [0, 5e12])
            #
            # lib.savefig(
            #     fig=fig,
            #     filename=f"{name}_fit_2",
            #     subdir=plot_dir
            # )
            #
            # print("\tFit mass 2", mass_fitted, np.log10(mass_fitted.value))
            # group_properties["mass_fitted_2"] = mass_fitted
            # group_properties["log_mass_fitted_2"] = np.log10(mass_fitted.value)
            # group_properties["dm_fitted_2"] = halo_fitted.Ne_Rperp(group_properties["r_perp_frb"]) / (1 + group_z)
            #
            # # ================================================
            # # Actual fit using nfw_3() and data, with position fitted
            #
            # v = np.abs(group_members["v_pec"])
            # x = group_members["x_projected"]
            # y = group_members["y_projected"]
            #
            # nfw_model = nfw_3(
            #     bounds={
            #         "mass": mass_bounds,
            #         "x_centre": (x.min(), x.max()),
            #         "y_centre": (y.min(), y.max())
            #     },
            # )
            #
            # fitter = fitting.LevMarLSQFitter()
            # fit_3 = fitter(nfw_model, x.value, y.value, v.value)
            #
            # halo_fitted = halo_setup(group_z, fit_3.mass * units.solMass)
            # mass_fitted = halo_fitted.M_halo.to("solMass")
            #
            # x_fitted = fit_3.x_centre * units.kpc
            # y_fitted = fit_3.y_centre * units.kpc
            #
            # # x_centre = fit_3.x_centre
            # # y_centre = fit_3.y_centre
            #
            # r_g = np.sqrt((x - x_fitted) ** 2 + (y - y_fitted) ** 2)
            # m = mass_contained(r_g, group_members["v_pec"])
            #
            # x_coord = np.linspace(x_fitted.value, (x_fitted + r_g.max()).value + 40, 1000) * units.kpc
            # y_coord = np.ones_like(x_coord) * y_fitted.value
            # r_coord = np.sqrt((x_coord - x_fitted) ** 2 + (y_coord - y_fitted) ** 2)
            # v_coord = fit_3(x_coord.value, y_coord.value) * units.km / units.s
            # m_coord = mass_contained(r_coord, v_coord)
            #
            # fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
            #
            # ax.scatter(r_g, m, marker='x', c='red')
            # ax.plot(r_coord, m_coord)
            # ax.set_ylabel(f"Contained mass ({units.solMass.to_string('latex')})")
            # ax.set_xlabel(f"Distance to group centre ({units.kpc.to_string('latex')})")
            # lib.savefig(
            #     fig=fig,
            #     filename=f"{name}_fit_3",
            #     subdir=plot_dir
            # )
            #
            # fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
            # ax.scatter(r_g, v, marker='x', c='red')
            # ax.plot(r_coord, v_coord)
            # ax.set_ylabel("Peculiar velocity (kms$^{-1}$)")
            # ax.set_xlabel(f"Distance to group centre ({units.kpc.to_string('latex')})")
            # lib.savefig(
            #     fig=fig,
            #     filename=f"{name}_fit_3_v_r",
            #     subdir=plot_dir
            # )
            # # plt.xlim(0, 200)
            # # plt.ylim(-0.1e13, 2e13)
            #
            # # plt.plot([halo_fitted.r200.value, halo_fitted.r200.value], [0, 5e12])
            #
            # v_predict = fit_3(x.value, y.value) * units.km / units.s
            #
            # residuals = v - v_predict
            # rms = np.sqrt(np.mean(residuals ** 2))
            #
            # fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
            # ax.scatter(r_g, residuals)
            # lib.savefig(
            #     fig=fig,
            #     filename=f"{name}_fit_3_v_residuals",
            #     subdir=plot_dir
            # )
            #
            # fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
            # residuals_m = m - mass_contained(r_g, v_predict)
            # ax.scatter(r_g, residuals_m)
            # lib.savefig(
            #     fig=fig,
            #     filename=f"{name}_fit_3_m_residuals",
            #     subdir=plot_dir
            # )
            #
            # plt.style.use('dark_background')
            # fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
            # vm = max(np.abs(group_members["v_pec"].min().value), np.abs(group_members["v_pec"].max().value))
            # sc = ax.scatter(
            #     x,
            #     y,
            #     c=group_members["v_pec"],
            #     cmap="bwr",
            #     vmax=vm,
            #     vmin=-vm,
            #     facecolor="black"
            # )
            # cbar = plt.colorbar(sc)
            # # for row in group_members:
            # #     plt.text(row["x_projected"], row["y_projected"], row["id_short"])
            # ax.axis("equal")
            # ax.scatter(x_fitted, y_fitted, marker="x", c="green")
            # ax.scatter(group_centre_x.value, group_centre_y.value, marker="x", c="violet")
            # ax.tick_params(axis="both", labelsize=pl.tick_fontsize)
            # ax.set_xlabel("Projected $x$-position (kpc)", fontsize=14)
            # ax.set_ylabel("Projected $y$-position (kpc)", fontsize=14)
            # cbar.ax.set_ylabel("Peculiar velocity (km/s)", labelpad=15, rotation=270, fontsize=14)
            #
            # for row in group_members:
            #     plt.text(row["x_projected"] - 2.5 * units.kpc, row["y_projected"] - 6 * units.kpc, row["id_short"],
            #              fontsize=14)
            #
            # lib.savefig(
            #     fig=fig,
            #     filename=f"{name}_positions_fit_3",
            #     subdir=plot_dir
            # )
            #
            # plt.style.use('default')
            # pl.latex_setup()
            #
            # print("\tFit mass 3", mass_fitted, np.log10(mass_fitted.value))
            # group_properties["mass_fitted_3"] = mass_fitted
            # group_properties["log_mass_fitted_3"] = np.log10(mass_fitted.value)
            # group_properties["dm_fitted_3"] = halo_fitted.Ne_Rperp(group_properties["r_perp_frb"]) / (1 + group_z)
            # group_properties["x_fitted"] = x_fitted
            # group_properties["y_fitted"] = y_fitted

            hst = lib.load_image("hst-ir")
            vlt = lib.load_image("vlt-fors2_g")

            for img in (hst, vlt):
                # fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))

                cmp, fig, ax = lib.label_objects(
                    tbl=group_members,
                    img=img,
                    output=f"group_{name}",
                    ellipse_colour="v_pec",
                    short_labels=True,
                    text_colour="white",
                    do_cut=True,
                    show_frb=True,
                    imshow_kwargs={"cmap": "binary_r"},
                    save=False,
                    ellipse_kwargs={"cbar_label": "Peculiar velocity (km/s)", "shrink": 0.7},
                    figsize=(lib.figwidth * 0.46, lib.figwidth * 0.46),
                    # text_position="below"
                    # factor=3,
                    height_factor=-1,
                )
                ra, dec = ax.coords
                ra.set_ticklabel(
                    fontsize=pl.tick_fontsize,
                    exclude_overlapping=True
                    # rotation=45,
                    # pad=50
                )
                # fig.colorbar(cmp)
                # fig.tight_layout(h_pad=0.1)
                x_centre, y_centre = img.world_to_pixel(group_centre)
                ax.scatter(x_centre, y_centre, marker="x", c="violet")
                if not skip_grid:
                    x_centre_grid, y_centre_grid = img.world_to_pixel(group_centre_grid)
                    ax.scatter(x_centre_grid, y_centre_grid, marker="x", c="limegreen")
                lib.savefig(
                    fig=fig,
                    filename=f"{name}_{img.name}",
                    subdir=plot_dir,
                    # tight=True
                )

        p.save_params(os.path.join(group_dir_this, "group_properties"), group_properties)
        group_members.write(os.path.join(group_dir_this, "group_members.ecsv"), format="ascii.ecsv", overwrite=True)
        group_members_terse = group_members[
            "id_short",
            "x_projected", "y_projected",
            "r_group", "v_pec", "mass_contained", "log_mass_contained", "v_escape", "frac_v_escape"
        ]
        group_members_terse.write(
            os.path.join(group_dir_this, "group_members_terse.csv"),
            format="ascii.csv",
            overwrite=True
        )


    cg_table = table.QTable(cg_table)
    cg_table["ra"] = cg_table["centre"].ra
    cg_table["dec"] = cg_table["centre"].dec
    cg_table.write(os.path.join(group_dir, "candidate_group_table.ecsv"), overwrite=True)

    u.latexise_table(
        tbl=cg_table[
            "id",
            "z",
            "ra",
            "dec",
            "r_perp",
            "sigma_v",
            "log_mass_virial",
            "r_200",
            "dm_igrm_0.8",
        ],
        round_dict={
            "dm_igrm_0.8": 0,
            "z": 3
        },
        round_cols=[
            "z",
            "r_perp",
            "sigma_v",
            "log_mass_virial",
            "r_200",
            "dm_igrm_0.8",
        ],
        round_digits=1,
        column_dict={
            "id": "CG",
            "z": "$z_\mathrm{group}$",
            "ra": r"$\alpha$",
            "dec": r"$\delta$",
            "r_perp": r"$R_\perp$",
            "sigma_v": r"$\sigma_v$",
            "log_mass_virial": r"$\log_{10}(\dfrac{\Mhalo}{\si{\solarmass}})$",
            "r_200": r"$R_{200}$",
            "dm_igrm_0.8": r"\DM{IGrM}"
        },
        sub_colnames={
            "r_perp": r"kpc",
            "sigma_v": r"\si{\m\s^{-1}}",
            "r_200": r"kpc",
            "dm_igrm_0.8": r"\dmunits"
        },
        ra_col="ra",
        dec_col="dec",
        output_path=os.path.join(lib.latex_table_path, "groups.tex"),
        second_path=os.path.join(lib.latex_table_path_db, "groups.tex"),
        label="tab:groups",
        short_caption="Candidate Group properties.",
        caption="Candidate Group properties."
                r" \tabscript{" + this_script + "}",
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
        "--skip_grid",
        help="Skip grid calculations.",
        action="store_true",
    )
    parser.add_argument(
        "--quick",
        help="Gotta go fast!",
        action="store_true",
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        skip_grid=args.skip_grid,
        quick=args.quick,
    )
